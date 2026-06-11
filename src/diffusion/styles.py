"""
src/diffusion/styles.py — 貼圖風格 preset
========================================

每個風格提供:
  - label         : UI 顯示用中文名稱
  - en            : 英文風格名
  - prompt_suffix : 附加到每張 visual_prompt 的共用風格關鍵字 (確保整組畫風一致)

新增風格只要在此字典加一筆即可,UI 會自動帶出。
"""

from __future__ import annotations

from typing import Dict

STYLE_PRESETS: Dict[str, Dict[str, str]] = {
    "kawaii": {
        "label": "可愛卡通",
        "en": "kawaii cartoon",
        "prompt_suffix": "kawaii cartoon sticker, soft pastel colors, thick clean outline, simple cel shading, cute big eyes",
    },
    "chibi": {
        "label": "Q版大頭",
        "en": "chibi",
        "prompt_suffix": "chibi style sticker, big head small body, adorable, bold outline, flat vibrant colors",
    },
    "doodle": {
        "label": "手繪塗鴉",
        "en": "hand-drawn doodle",
        "prompt_suffix": "hand-drawn doodle sticker, marker pen style, rough sketchy outline, playful, minimal flat color",
    },
    "lineart": {
        "label": "線條簡約",
        "en": "minimal line art",
        "prompt_suffix": "minimal line art sticker, single weight black outline, mostly white, very simple, clean negative space",
    },
    "pixel": {
        "label": "像素風",
        "en": "pixel art",
        "prompt_suffix": "pixel art sticker, 8-bit retro game style, crisp pixels, limited color palette, bold silhouette",
    },
    "watercolor": {
        "label": "水彩手帳",
        "en": "watercolor",
        "prompt_suffix": "soft watercolor sticker, gentle gradients, paper texture, pastel washes, dreamy cozy mood",
    },
    "3d": {
        "label": "3D 軟糖",
        "en": "3d glossy",
        "prompt_suffix": "glossy 3d render sticker, soft studio lighting, rounded clay-like forms, vibrant candy colors, cute",
    },
}

DEFAULT_STYLE = "kawaii"


def get_style(key: str) -> Dict[str, str]:
    return STYLE_PRESETS.get(key, STYLE_PRESETS[DEFAULT_STYLE])
