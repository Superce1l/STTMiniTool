"""make_icon.py — 由用户提供的 Streamline 图标（语音转文字：麦克风→A）生成程序图标。

来源：assets/icon_src.svg（StreamlinePlumpColorVoiceTypingWordConvertFlat，
麦克风 + 转换箭头 + 字母 A 的彩色扁平图标）。

管线：SVG --(PyMuPDF 光栅化)--> 透明 PNG --> 加边距居中 --> 多尺寸 ICO。
说明：svglib/cairosvg 在无 cairo DLL 的 Windows 环境不可用，故直接用
PyMuPDF 打开 SVG 光栅化（其内置 MuPDF 渲染器，无外部依赖）。

产出：
  icon.png  —  512px 透明背景（网页 favicon / 窗口图标用）
  icon.ico  —  多尺寸 (16/32/48/64/128/256)（PyInstaller --icon / 窗口）
  webview/favicon.png — 64px（WebView 界面标签页图标）
"""
from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image

ASSETS = Path(__file__).resolve().parent
SRC = ASSETS / "icon_src.svg"
S = 512                       # 主图标尺寸


def render_svg(size: int) -> Image.Image:
    """SVG → 透明背景 RGBA（按 viewBox 等比放大）。"""
    doc = pymupdf.open(str(SRC))
    page = doc[0]
    zoom = size / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
    return Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)


def square_pad(img: Image.Image, size: int, pad_ratio: float = 0.06) -> Image.Image:
    """裁到内容外接框，四周留 pad_ratio 边距，居中放到 size×size 画布。"""
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    side = max(img.size)
    pad = int(side * pad_ratio)
    canvas = Image.new("RGBA", (side + 2 * pad, side + 2 * pad), (0, 0, 0, 0))
    canvas.paste(img, ((canvas.width - img.width) // 2,
                       (canvas.height - img.height) // 2), img)
    return canvas.resize((size, size), Image.LANCZOS)


def main():
    if not SRC.exists():
        raise SystemExit(f"找不到来源 SVG：{SRC}")

    icon = square_pad(render_svg(S * 2), S)
    icon.save(ASSETS / "icon.png", "PNG")
    print(f"saved {ASSETS / 'icon.png'} ({S}px)")

    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.save(ASSETS / "icon.ico", format="ICO", sizes=sizes)
    print(f"saved {ASSETS / 'icon.ico'} sizes={[s for s, _ in sizes]}")

    icon.resize((64, 64), Image.LANCZOS).save(
        ASSETS.parent / "webview" / "favicon.png", "PNG")
    print(f"saved {ASSETS.parent / 'webview' / 'favicon.png'} (64px)")


if __name__ == "__main__":
    main()
