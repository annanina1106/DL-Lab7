"""
src/postprocess/caption.py — 文案疊字
====================================

把 LLM 產生的中文文案疊到貼圖底部,做出真正「有字的貼圖」。

技術重點:
  * CJK 字型偵測:自動搜尋系統常見中文字型 (Noto / 思源 / 微軟正黑 / 蘋方…),
    或使用 FONT_PATH 指定;找不到時記錄警告並略過疊字 (UI 仍會顯示文案文字)。
  * 可讀性:文字加白底黑邊 (stroke),避免在任何底色上看不清楚。
  * 自動縮放:依文字長度與圖寬自動調整字級,避免超出邊界。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.utils import get_logger

logger = get_logger("postprocess.caption")


# 常見 CJK 字型候選路徑 (跨 Linux / macOS / Windows)
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Bold.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/source-han-sans/SourceHanSansTC-Bold.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


@lru_cache(maxsize=1)
def find_cjk_font() -> Optional[str]:
    """回傳可用的 CJK 字型路徑;找不到回傳 None。"""
    # 1. 環境變數指定優先
    env_path = os.getenv("FONT_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. 專案內 assets/fonts/ 自帶字型
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fonts_dir = os.path.join(here, "assets", "fonts")
    if os.path.isdir(fonts_dir):
        for f in sorted(os.listdir(fonts_dir)):
            if f.lower().endswith((".ttf", ".ttc", ".otf")):
                return os.path.join(fonts_dir, f)

    # 3. 系統常見路徑
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path

    logger.warning(
        "找不到 CJK 字型,將略過疊字。可執行 `sudo apt install fonts-noto-cjk`,"
        "或把 .ttf/.otf 放到 assets/fonts/,或設定 FONT_PATH。"
    )
    return None


def _fit_font(text: str, max_width: int, base_px: int, font_path: str):
    """自動縮小字級,直到文字寬度不超過 max_width。失敗回傳 None。"""
    px = base_px
    try:
        while px > 12:
            font = ImageFont.truetype(font_path, px)
            bbox = font.getbbox(text)
            if (bbox[2] - bbox[0]) <= max_width:
                return font
            px -= 2
        return ImageFont.truetype(font_path, 12)
    except Exception as e:  # noqa: BLE001
        logger.warning("字型無法載入 (%s: %s),略過疊字。", font_path, e)
        return None


def overlay_caption(
    img: Image.Image,
    text: str,
    font_path: Optional[str] = None,
) -> Image.Image:
    """在圖片底部置中疊上文案 (白字黑邊)。"""
    if not text:
        return img

    font_path = font_path or find_cjk_font()
    if not font_path:
        return img  # 無字型,直接回傳原圖

    img = img.convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    base_px = max(20, W // 9)
    font = _fit_font(text, int(W * 0.9), base_px, font_path)
    if font is None:
        return img  # 字型載入失敗,略過疊字

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=max(2, base_px // 10))
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - tw) / 2 - bbox[0]
    y = H - th - max(8, H // 20) - bbox[1]

    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=max(2, base_px // 10),
        stroke_fill=(40, 40, 40, 255),
    )
    return img
