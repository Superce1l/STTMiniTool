"""
diarize.py
─────────────────────────────────────────────────────────────────────
说话者分离（Speaker Diarization）引擎

两个 ONNX 模型（共 32.5 MB，位于 ov_models/diarization/）：
  segmentation-community-1.onnx  — 检测语音段落 + 粗略说话者类别
  embedding_model.onnx           — 提取说话者声纹向量（WeSpeaker）

聚类算法（2026-10 改版）：
  三阶段设计 — 长段落先切成约 3 秒子窗逐窗提声纹（避免段内换人
  被单一声纹摊平），再对全部子窗做一次全局聚类，最后把短段落
  指派给最近的群（不再丢弃，修正漏字）。

  ┌─ 自动模式（n_speakers=None）
  │   Average-linkage 层次树上自深而浅搜索：取「所有群中心两两
  │   cosine 距离 ≥ SPLIT_SURE_DIST」的最深切割；没有任何切割
  │   达标 → 整档仅一位说话者。群心距离是实测中区分「同一人被
  │   拆」与「真正不同人」最可靠的量（同一人子群 ≤0.37、不同人
  │   ≥0.39，界限两侧都有真实样本），silhouette 与特征值间隙
  │   都在这两类真实样本上翻过车（见 _cluster_auto 文档）。
  └─ 指定人数（n_speakers=N）
      强制分成 N 组（maxclust criterion）
      → 适合已知说话者数量的情境，避免过度分割

使用方式：
  from diarize import DiarizationEngine
  eng = DiarizationEngine(model_dir / "diarization")
  segments = eng.diarize(audio_float32_16khz, n_speakers=2)
  # → [(0.40, 4.55, "说话者1"), (4.85, 9.28, "说话者2"), ...]
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _require_scipy():
    """说话者分离的聚类依赖 scipy；缺失时给中文指引而非裸 ImportError。"""
    try:
        import scipy  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "说话者分离需要 scipy 组件但当前环境未包含。请重新安装完整版程序，"
            "或执行 pip install scipy 后重试。") from e


# ── 常数（从 pyannote-rs 源码 segment.rs 取得）─────────────────
SAMPLE_RATE    = 16_000
WINDOW_SAMPLES = SAMPLE_RATE * 10          # 160,000 samples = 10 秒
FRAME_SIZE     = 270                       # samples per output frame
FRAME_START    = 721                       # initial sample offset
MIN_SEG_SEC    = 0.8                       # 过短段落过滤门槛（秒）
MERGE_GAP_SEC  = 0.30                      # 相邻同说话者合并间距（秒）

# 自动模式的 cosine distance 切割阈值
# cosine_dist = 1 - cosine_sim；值越小 → 聚类越严格（说话者数越多）
# 2026-08 起仅作为「样本数 < 4」时的 fallback；主要人数判定见 _cluster_auto
AUTO_DIST_THRESH = 0.38

# ── 聚类质量改良参数（2026-08）───────────────────────────────────
EMB_WIN_SEC       = 3.0    # 长段落切子窗提声纹的窗口长度（秒）
KEEP_MIN_SEC      = 0.30   # 段落保留下限：低于此秒数视为杂讯丢弃
MAX_AUTO_SPEAKERS = 6      # 自动模式最多尝试的说话者数
# 「确为不同说话者」的群心距离门槛（cosine 距离）。实测校准：
#   同一人被拆出的子群中心距 ≤ 0.368（真实对唱歌曲 k=3 切割、
#   TTS 双人子簇 0.34）；真正不同人 ≥ 0.39（TTS 校准值）与
#   0.417（真实男女对唱）。0.38 落在两者之间的空隙里。
SPLIT_SURE_DIST = 0.38
# 少于此数的子窗自成一群 → 视为碎群（间奏/换气/伴奏窗），并入最近大群。
# 3 窗 ≈ 4.5 秒语音；真实说话者在 ≥2 分钟音频里几乎必然超过这个量。
TINY_CLUSTER_WINDOWS = 3


def _subwindows(t0: float, t1: float,
                win: float = EMB_WIN_SEC, hop: float | None = None) -> list[tuple[float, float]]:
    """把 [t0, t1] 切成 win 秒、相邻 50% 重叠的子窗（默认 hop=win/2）。

    重叠窗让声纹样本密度翻倍——人数估计（特征值间隙）与聚类稳定性都受益。
    短段（≤1.5×win）退回等分（不产生碎窗）；尾部不足一个 hop 时并入贴尾窗。
    """
    hop = (win / 2.0) if hop is None else float(hop)
    dur = t1 - t0
    if dur <= win * 1.5:                      # 短段：等分 1~2 窗
        n = max(1, int(round(dur / win)))
        edges = np.linspace(t0, t1, n + 1)
        return list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    starts = np.arange(t0, t1 - win + 1e-6, hop).tolist()
    if not starts:
        starts = [t0]
    if starts[-1] + win < t1 - 1e-6:          # 贴尾补一窗，覆盖段尾
        starts.append(t1 - win)
    dedup = [starts[0]]
    for s in starts[1:]:
        if s - dedup[-1] > hop * 0.4:         # 贴尾窗与前一窗过近时只留一个
            dedup.append(s)
    return [(s, s + win) for s in dedup]


def _absorb_tiny(labels: np.ndarray, emb: np.ndarray,
                 min_size: int = TINY_CLUSTER_WINDOWS) -> np.ndarray:
    """把少于 min_size 个子窗的碎群并入 cosine 最近的够大群。

    间奏、换气、伴奏残留等杂讯窗在深切割下会自成一群，把「群心两两
    够远」的判定干扰成假说话者；先吸收再判定。一个够大群都没有时
    原样返回（交由调用端的 <2 群分支跳过该切割）。
    """
    labels = labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)
    big = {int(c) for c, n in zip(uniq, counts) if n >= min_size}
    if not big or len(big) == len(uniq):
        return labels
    cents = {}
    for c in big:
        v = emb[labels == c].mean(axis=0)
        nrm = np.linalg.norm(v)
        cents[c] = v / nrm if nrm > 1e-9 else v
    for i, l in enumerate(labels):
        if l not in big:
            labels[i] = max(cents, key=lambda c: float(emb[i] @ cents[c]))
    return labels


def _min_centroid_dist(emb: np.ndarray, labels: np.ndarray) -> float:
    """各群中心两两 cosine 距离的最小值（群心越远 → 越像不同的人）。"""
    cents = []
    for c in np.unique(labels):
        v = emb[labels == c].mean(axis=0)
        nrm = np.linalg.norm(v)
        cents.append(v / nrm if nrm > 1e-9 else v)
    return min(1.0 - float(a @ b)
               for i, a in enumerate(cents) for b in cents[i + 1:])


class DiarizationEngine:
    """
    说话者分离引擎（thread-safe）。

    属性：
        ready : bool  — 模型已加载且可用
    """

    def __init__(self, diar_dir: Path):
        self._lock    = threading.Lock()
        self.ready    = False
        self.seg_sess = None
        self.emb_sess = None
        self._diar_dir = diar_dir
        self._load()

    # ── 模型加载 ──────────────────────────────────────────────────

    def _load(self):
        seg_path = self._diar_dir / "segmentation-community-1.onnx"
        emb_path = self._diar_dir / "embedding_model.onnx"
        if not seg_path.exists() or not emb_path.exists():
            return   # 静默失败，self.ready 保持 False

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.log_severity_level   = 3   # 隐藏 ONNX Runtime 警告
        self.seg_sess = ort.InferenceSession(
            str(seg_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.emb_sess = ort.InferenceSession(
            str(emb_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.ready = True

    # ── 公开 API ─────────────────────────────────────────────────

    def diarize(
        self,
        audio: np.ndarray,
        n_speakers: int | None = None,
    ) -> list[tuple[float, float, str]]:
        """
        对 16kHz float32 音频执行说话者分离。

        n_speakers : 指定说话者总人数（None = 自动检测）。
                     已知人数时强烈建议指定，可避免过度分割。

        返回：[(start_sec, end_sec, "说话者N"), ...]
        已过滤静音与过短段落，可直接作为 ASR 的分段依据。
        """
        with self._lock:
            raw  = self._segment(audio)
            return self._embed_and_cluster(audio, raw, n_speakers=n_speakers)

    def diarize_detailed(
        self,
        audio: np.ndarray,
        n_speakers: int | None = None,
    ) -> tuple[list[tuple[float, float, int]], dict[int, np.ndarray]]:
        """diarize() 的扩展版：标签用整数，并返回各说话者的声纹中心。

        供长音频切片转录做跨片说话者映射：每片独立聚类的编号互不相干，
        唯有声纹中心可比（webview_backend._match_speaker 据此对齐编号）。

        返回 (segments, centroids)：
          segments  : [(start_sec, end_sec, label_int), ...]（label 从 1 起）
          centroids : {label_int: L2 正规化声纹向量}；个别标签可能缺
                      （该说话者只有过短段落、提不出可靠声纹）
        """
        with self._lock:
            raw = self._segment(audio)
            return self._embed_and_cluster(audio, raw, n_speakers=n_speakers,
                                           return_centroids=True)

    # ── 分割（Segmentation Model）────────────────────────────────

    def _segment(
        self, audio: np.ndarray
    ) -> list[tuple[float, float, int]]:
        """
        返回：[(start_sec, end_sec, local_class), ...]
        local_class: 1-6（非 0 静音）
        """
        input_name = self.seg_sess.get_inputs()[0].name   # "input_values"

        n   = len(audio)
        pad = (WINDOW_SAMPLES - n % WINDOW_SAMPLES) % WINDOW_SAMPLES
        padded = np.pad(audio.astype(np.float32), (0, pad))

        raw_segs: list[tuple[float, float, int]] = []

        for win_start in range(0, len(padded), WINDOW_SAMPLES):
            window = padded[win_start: win_start + WINDOW_SAMPLES]
            inp    = window[np.newaxis, np.newaxis, :]
            logits = self.seg_sess.run(None, {input_name: inp})[0]
            frame_labels = np.argmax(logits[0], axis=-1)

            def _frame_to_sec(fi: int, _ws: int = win_start) -> float:
                return (_ws + FRAME_START + fi * FRAME_SIZE) / SAMPLE_RATE

            cur_lbl   = int(frame_labels[0])
            seg_start = 0
            for fi in range(1, len(frame_labels)):
                lbl = int(frame_labels[fi])
                if lbl != cur_lbl:
                    if cur_lbl != 0:
                        raw_segs.append((_frame_to_sec(seg_start),
                                         _frame_to_sec(fi), cur_lbl))
                    seg_start = fi
                    cur_lbl   = lbl
            if cur_lbl != 0:
                raw_segs.append((_frame_to_sec(seg_start),
                                 _frame_to_sec(len(frame_labels)), cur_lbl))

        # 裁剪 + 过滤 + 合并相邻同类段落
        total_dur = n / SAMPLE_RATE
        merged: list[tuple[float, float, int]] = []
        for t0, t1, lbl in raw_segs:
            t0 = min(t0, total_dur)
            t1 = min(t1, total_dur)
            if t1 - t0 < 0.1:
                continue
            if (merged and merged[-1][2] == lbl
                    and t0 - merged[-1][1] < MERGE_GAP_SEC):
                merged[-1] = (merged[-1][0], t1, lbl)
            else:
                merged.append((t0, t1, lbl))

        return merged

    # ── 嵌入提取（Embedding Model）──────────────────────────────

    def _kaldi_fbank(self, samples_f32: np.ndarray) -> np.ndarray:
        """Kaldi-style 80 维 mel filter bank（WeSpeaker 标准前处理）。"""
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq       = float(SAMPLE_RATE)
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.frame_shift_ms  = 10.0
        opts.mel_opts.num_bins          = 80
        opts.frame_opts.dither          = 0.0

        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(
            float(SAMPLE_RATE),
            (samples_f32 * 32768.0).tolist(),
        )
        fbank.input_finished()

        n_frames = fbank.num_frames_ready
        if n_frames == 0:
            return np.zeros((1, 80), dtype=np.float32)

        feats = np.array(
            [fbank.get_frame(i) for i in range(n_frames)],
            dtype=np.float32,
        )
        feats -= feats.mean(axis=0, keepdims=True)   # CMN 正规化
        return feats

    def _get_embedding(
        self, audio: np.ndarray, t0: float, t1: float
    ) -> np.ndarray | None:
        """从 [t0, t1] 秒音频提取 L2 正规化后的 256 维说话者向量。"""
        s0    = int(t0 * SAMPLE_RATE)
        s1    = int(t1 * SAMPLE_RATE)
        chunk = audio[s0:s1]
        if len(chunk) < SAMPLE_RATE * 0.5:
            return None

        feats = self._kaldi_fbank(chunk)
        out   = self.emb_sess.run(
            None, {"fbank_features": feats[np.newaxis, :]}
        )[0][0]
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-9 else out

    # ── 两阶段聚类 ───────────────────────────────────────────────

    def _embed_and_cluster(
        self,
        audio: np.ndarray,
        raw_segs: list[tuple[float, float, int]],
        n_speakers: int | None = None,
        return_centroids: bool = False,
    ):
        """
        三阶段设计（2026-08 改版）：
          Stage 1  — 长段落切成约 EMB_WIN_SEC 秒的子窗逐窗提声纹，
                     避免长段落单一声纹把段内换人摊平成「混合声纹」
          Stage 2  — 对所有子窗声纹统一聚类（自动模式以 silhouette
                     选最佳人数），并做段内标签平滑
          Stage 3  — 短段落不再丢弃：有声纹者指派最近群中心，
                     无声纹者跟随时间上最近的邻居（修正旧版
                     < 0.8 秒短句整段从 ASR 消失的漏字问题）
        """
        # Stage 1：子窗声纹。units: (seg_id, w0, w1, emb)
        units:  list[tuple[int, float, float, np.ndarray]] = []
        shorts: list[tuple[float, float, np.ndarray | None]] = []
        for seg_id, (t0, t1, _) in enumerate(raw_segs):
            if t1 - t0 < KEEP_MIN_SEC:
                continue                       # 过短杂讯，仍然丢弃
            if t1 - t0 < MIN_SEG_SEC:          # 短段：留待 Stage 3 指派
                shorts.append((t0, t1, self._get_embedding(audio, t0, t1)))
                continue
            for w0, w1 in _subwindows(t0, t1):
                emb = self._get_embedding(audio, w0, w1)
                if emb is not None:
                    units.append((seg_id, w0, w1, emb))

        if not units:
            # 没有任何可靠长段落：短段全数视为单一说话者输出
            if not return_centroids:
                return [(t0, t1, "说话者1") for t0, t1, _ in shorts]
            segs = [(t0, t1, 1) for t0, t1, _ in shorts]
            embs = [e for (_t0, _t1, e) in shorts if e is not None]
            cent: dict[int, np.ndarray] = {}
            if embs:
                c = np.mean(embs, axis=0)
                nrm = np.linalg.norm(c)
                cent[1] = c / nrm if nrm > 1e-9 else c
            return segs, cent

        embeddings = [u[3] for u in units]

        # Stage 2：聚类取得说话者标签
        if len(embeddings) == 1:
            labels = [1]
        elif n_speakers is not None:
            labels = self._cluster_fixed_n(embeddings, n_speakers)
        else:
            labels = self._cluster_auto(embeddings)

        # 段内平滑：单一子窗与前后邻居（同一原始段落）不同
        # → 跟随邻居，消除声纹抖动造成的假换人
        for i in range(1, len(units) - 1):
            if (units[i - 1][0] == units[i][0] == units[i + 1][0]
                    and labels[i - 1] == labels[i + 1] != labels[i]):
                labels[i] = labels[i - 1]

        # Stage 3：短段落指派
        centroids: dict[int, np.ndarray] = {}
        for lbl in set(labels):
            c = np.mean([e for e, l in zip(embeddings, labels) if l == lbl],
                        axis=0)
            nrm = np.linalg.norm(c)
            centroids[lbl] = c / nrm if nrm > 1e-9 else c

        placed: list[tuple[float, float, int]] = [
            (w0, w1, lbl) for (_, w0, w1, _), lbl in zip(units, labels)
        ]
        for t0, t1, emb in shorts:
            if emb is not None:                # ≥0.5 秒：最近群中心
                lbl = min(centroids,
                          key=lambda k: 1.0 - float(emb @ centroids[k]))
            else:                              # <0.5 秒：时间上最近的邻居
                mid = (t0 + t1) / 2.0
                lbl = min(placed,
                          key=lambda p: abs((p[0] + p[1]) / 2.0 - mid))[2]
            placed.append((t0, t1, lbl))
        placed.sort(key=lambda p: p[0])

        # 标签依首次出现顺序重编为 1..N（「说话者1」永远最先开口）
        remap: dict[int, int] = {}
        for _, _, lbl in placed:
            if lbl not in remap:
                remap[lbl] = len(remap) + 1

        # 合并相邻的同说话者段落（内部先以整数标签累积，字符串化放最后）
        merged: list[tuple[float, float, int]] = []
        for t0, t1, lbl in placed:
            new = remap[lbl]
            if (merged and merged[-1][2] == new
                    and t0 - merged[-1][1] < MERGE_GAP_SEC):
                merged[-1] = (merged[-1][0], t1, new)
            else:
                merged.append((t0, t1, new))

        if not return_centroids:
            return [(t0, t1, f"说话者{lbl}") for (t0, t1, lbl) in merged]

        # 各说话者声纹中心（长段落子窗的 embedding 平均，L2 正规化）
        centroids: dict[int, np.ndarray] = {}
        for local_lbl, new_lbl in remap.items():
            vecs = [e for e, l in zip(embeddings, labels) if l == local_lbl]
            if not vecs:
                continue
            c = np.mean(vecs, axis=0)
            nrm = np.linalg.norm(c)
            centroids[new_lbl] = c / nrm if nrm > 1e-9 else c
        return merged, centroids

    def _cluster_fixed_n(
        self, embeddings: list[np.ndarray], n: int
    ) -> list[int]:
        """
        指定人数模式：Average-linkage 层次聚类，强制分成 n 组。
        返回 1-indexed 标签列表。
        """
        _require_scipy()
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        n = max(1, min(n, len(embeddings)))  # 限制在合理范围
        if len(embeddings) <= n:
            return list(range(1, len(embeddings) + 1))

        emb_mat = np.stack(embeddings)           # (M, 256)
        sim     = emb_mat @ emb_mat.T            # cosine similarity
        dist    = np.clip(1.0 - sim, 0.0, 2.0)  # cosine distance
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)

        Z      = linkage(condensed, method="average")
        labels = fcluster(Z, t=n, criterion="maxclust")   # 1-indexed
        return labels.tolist()

    def _cluster_auto(
        self, embeddings: list[np.ndarray]
    ) -> list[int]:
        """
        自动模式（2026-10 二次改版）：在层次树上自深而浅搜索，取
        「碎群吸收后，所有群中心两两 cosine 距离 ≥ SPLIT_SURE_DIST」
        的最深切割；一个达标的都没有 → 整档仅一位说话者。

        前两版都在真实音频上翻过车，教训值得留档：
        • argmax-silhouette（08 版）：紧致大簇继续切分数反而升高，
          双人 TTS 被切成 3 人（k=3 sil 0.864 > 正确 k=2 的 0.776）。
        • 谱特征值间隙（10 月一版）：相似度矩阵的 λ1 被「全局公共
          成分」占据（同语言+共享伴奏 → 所有窗两两 ~0.3 相似，
          真实对唱 λ=[39.6, 13.0, 6.6,…]，λ1-λ2 间隙 26.6 一家独大
          → 恒判 1 人）；且 0.25 的 silhouette 门槛把真实双人的
          0.206 也误杀。合成测试因跨群相似度 ≈0 而全数通过——
          正是这种「过干净」让两版缺陷都漏网。

        群心距离是唯一在全部实测样本上界限分明的量（同一人子群
        ≤0.368 / 不同人 ≥0.39-0.417），故以它为唯一判定依据。
        返回 1-indexed 标签列表。
        """
        _require_scipy()
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        M = len(embeddings)
        if M < 2:
            return [1] * M

        emb_mat = np.stack(embeddings)
        dist    = np.clip(1.0 - emb_mat @ emb_mat.T, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")

        if M < 4:   # 样本太少 → 退回固定阈值
            return fcluster(Z, t=AUTO_DIST_THRESH,
                            criterion="distance").tolist()

        for k in range(min(MAX_AUTO_SPEAKERS, M - 1), 1, -1):
            labels = _absorb_tiny(
                np.asarray(fcluster(Z, t=k, criterion="maxclust")), emb_mat)
            if len(np.unique(labels)) < 2:
                continue
            if _min_centroid_dist(emb_mat, labels) >= SPLIT_SURE_DIST:
                return labels.tolist()
        return [1] * M
