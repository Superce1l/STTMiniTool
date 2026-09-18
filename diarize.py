"""
diarize.py
─────────────────────────────────────────────────────────────────────
说话者分离（Speaker Diarization）引擎

两个 ONNX 模型（共 32.5 MB，位于 ov_models/diarization/）：
  segmentation-community-1.onnx  — 检测语音段落 + 粗略说话者类别
  embedding_model.onnx           — 提取说话者声纹向量（WeSpeaker）

聚类算法（2026-08 改版）：
  三阶段设计 — 长段落先切成约 3 秒子窗逐窗提声纹（避免段内换人
  被单一声纹摊平），再对全部子窗做一次全局聚类，最后把短段落
  指派给最近的群（不再丢弃，修正漏字）。

  ┌─ 自动模式（n_speakers=None）
  │   Average-linkage 层次聚类；对 2..6 人各切一次树状图，
  │   以 silhouette 分数挑分离度最佳的人数；整体分数过低
  │   → 判定整档仅一位说话者。（取代旧版固定 0.38 距离阈值——
  │   固定阈值对相近音色会并人、对环境变化大会拆人）
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

# ── 常数（从 pyannote-rs 源码 segment.rs 取得）─────────────────
SAMPLE_RATE    = 16_000
WINDOW_SAMPLES = SAMPLE_RATE * 10          # 160,000 samples = 10 秒
FRAME_SIZE     = 270                       # samples per output frame
FRAME_START    = 721                       # initial sample offset
MIN_SEG_SEC    = 0.8                       # 过短段落过滤门槛（秒）
MERGE_GAP_SEC  = 0.30                      # 相邻同说话者合并间距（秒）

# 自动模式的 cosine distance 切割阈值
# cosine_dist = 1 - cosine_sim；值越小 → 聚类越严格（说话者数越多）
# 2026-08 起仅作为「样本数 < 4」时的 fallback；主要人数判定改用 silhouette
AUTO_DIST_THRESH = 0.38

# ── 聚类质量改良参数（2026-08）───────────────────────────────────
EMB_WIN_SEC       = 3.0    # 长段落切子窗提声纹的窗口长度（秒）
KEEP_MIN_SEC      = 0.30   # 段落保留下限：低于此秒数视为杂讯丢弃
MAX_AUTO_SPEAKERS = 6      # 自动模式最多尝试的说话者数
# 以下两个阈值由 test_audio 实测校准（真1人 sil≤0.17/中心距≤0.25；
# 真多人 sil≥0.41/中心距≥0.39，中间有明确空隙）：
SIL_SINGLE_THRESH   = 0.25  # silhouette 低于此值 → 判定整档仅一位说话者
CENTROID_MERGE_DIST = 0.30  # 群中心 cosine 距离低于此值 → 两群视为同一人合并


def _subwindows(t0: float, t1: float,
                win: float = EMB_WIN_SEC) -> list[tuple[float, float]]:
    """把 [t0, t1] 均分成每窗约 win 秒的子窗（尾窗并入均分，不产生残段）。"""
    n = max(1, int(round((t1 - t0) / win)))
    edges = np.linspace(t0, t1, n + 1)
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def _silhouette_from_dist(dist: np.ndarray, labels) -> float:
    """由预先算好的距离矩阵计算平均 silhouette 分数。

    单元素群的 silhouette 依标准定义记 0（避免噪音子窗自成一群
    反而拉高分数）。返回值域约 [-1, 1]，越高代表群间分得越开。
    """
    labels = np.asarray(labels)
    uniq   = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    scores = np.zeros(len(labels))
    for i in range(len(labels)):
        same    = labels == labels[i]
        same_i  = same.copy()
        same_i[i] = False
        if not same_i.any():          # 单元素群 → 0
            continue
        a = dist[i, same_i].mean()
        b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
        m = max(a, b)
        scores[i] = (b - a) / m if m > 1e-12 else 0.0
    return float(scores.mean())


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
    ) -> list[tuple[float, float, str]]:
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
            return [(t0, t1, "说话者1") for t0, t1, _ in shorts]

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

        # 合并相邻的同说话者段落
        merged: list[tuple[float, float, str]] = []
        for t0, t1, lbl in placed:
            spk = f"说话者{remap[lbl]}"
            if (merged and merged[-1][2] == spk
                    and t0 - merged[-1][1] < MERGE_GAP_SEC):
                merged[-1] = (merged[-1][0], t1, spk)
            else:
                merged.append((t0, t1, spk))

        return merged

    def _cluster_fixed_n(
        self, embeddings: list[np.ndarray], n: int
    ) -> list[int]:
        """
        指定人数模式：Average-linkage 层次聚类，强制分成 n 组。
        返回 1-indexed 标签列表。
        """
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
        自动模式（2026-08 改版）：对 2..MAX_AUTO_SPEAKERS 人各切一次
        层次聚类树，以 silhouette 分数挑分离度最佳的人数；整体分数
        低于 SIL_SINGLE_THRESH（群间根本分不开）→ 判定仅一位说话者。

        取代旧版固定距离阈值（AUTO_DIST_THRESH）切割：固定阈值对
        「同性别相近音色」会把两人并成一人、对「录音环境变化大」
        会把同一人拆成多人——正是实际使用回报的两大症状。
        固定阈值仅保留给样本数 < 4 的极短音频当 fallback。

        返回 1-indexed 标签列表。
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        M = len(embeddings)
        if M < 2:
            return [1] * M

        emb_mat = np.stack(embeddings)
        dist    = np.clip(1.0 - emb_mat @ emb_mat.T, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")

        if M < 4:   # 样本太少 silhouette 不可靠 → 退回固定阈值
            return fcluster(Z, t=AUTO_DIST_THRESH,
                            criterion="distance").tolist()

        best_sil, best_labels = -1.0, None
        for k in range(2, min(MAX_AUTO_SPEAKERS, M - 1) + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            sil = _silhouette_from_dist(dist, labels)
            if sil > best_sil:
                best_sil, best_labels = sil, labels.tolist()

        if best_labels is None or best_sil < SIL_SINGLE_THRESH:
            return [1] * M      # 群间分不开 → 整档视为一位说话者

        # 第二重防护：群中心距离过近（同一人被拆）→ 反复合并最近的两群。
        # 实测同一人的子群中心距 ≤0.25、不同人 ≥0.39，0.30 落在空隙中间。
        labels = np.asarray(best_labels)
        while True:
            uniq = np.unique(labels)
            if len(uniq) < 2:
                break
            cents = {}
            for c in uniq:
                v = emb_mat[labels == c].mean(axis=0)
                nrm = np.linalg.norm(v)
                cents[c] = v / nrm if nrm > 1e-9 else v
            pair, mind = None, 2.0
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    d = 1.0 - float(cents[uniq[i]] @ cents[uniq[j]])
                    if d < mind:
                        pair, mind = (uniq[i], uniq[j]), d
            if mind >= CENTROID_MERGE_DIST:
                break
            labels[labels == pair[1]] = pair[0]
        return labels.tolist()
