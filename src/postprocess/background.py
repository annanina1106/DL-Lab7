"""
src/postprocess/background.py — 自動去背 + 貼圖描邊
================================================

技術點 (Pipeline 客製化的後處理階段):
  1. 自動去背:使用 rembg (底層 U2Net) 將生成圖去除背景 → 透明 PNG (RGBA)。
     這是讓輸出「可直接當貼圖使用」的關鍵步驟。
  2. 貼圖描邊:模擬 LINE 貼圖經典的白色模切外框 (die-cut border) ——
     對 alpha 遮罩做膨脹後填白,再把原圖疊上,產生立體貼紙感。

穩健性:
  若未安裝 rembg,remove_background() 會記錄警告並回傳原圖 (RGBA),流程不中斷。
"""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageFilter

from src.utils import get_logger

logger = get_logger("postprocess.background")

_REMBG_SESSION = None  # 重用 session 以加速多張處理


def _get_session(model_name: str):
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session

        _REMBG_SESSION = new_session(model_name)
    return _REMBG_SESSION


def remove_background(img: Image.Image, model_name: str = "u2net") -> Image.Image:
    """去除背景,回傳 RGBA 圖。失敗時回傳原圖轉 RGBA。"""
    try:
        from rembg import remove
    except (ImportError, SystemExit, Exception) as e:
        logger.warning("rembg 無法載入 (%s),略過去背。請執行 `pip install \"rembg[gpu]\"` 或 `pip install onnxruntime`。", e)
        return img.convert("RGBA")

    try:
        session = _get_session(model_name)
        out = remove(img.convert("RGB"), session=session)
        if not isinstance(out, Image.Image):
            # 某些版本回傳 bytes
            import io

            out = Image.open(io.BytesIO(out))
        return out.convert("RGBA")
    except Exception as e:  # noqa: BLE001
        logger.warning("去背失敗 (%s),回傳原圖", e)
        return img.convert("RGBA")


def add_die_cut_border(
    img: Image.Image,
    border_size: int = 12,
    color: tuple = (255, 255, 255, 255),
) -> Image.Image:
    """為去背後的 RGBA 圖加上白色模切外框。"""
    if border_size <= 0:
        return img
    img = img.convert("RGBA")
    alpha = img.split()[3]

    # 以 MaxFilter 對 alpha 做膨脹 (kernel 需為奇數)
    k = border_size * 2 + 1
    try:
        dilated = alpha.filter(ImageFilter.MaxFilter(k))
    except ValueError:
        # MaxFilter 對過大的 kernel 會報錯,改用多次小核膨脹
        dilated = alpha
        small = ImageFilter.MaxFilter(9)
        for _ in range(max(1, border_size // 4)):
            dilated = dilated.filter(small)

    # 建立純白邊框層,套用膨脹後的 alpha 形狀
    border_layer = Image.new("RGBA", img.size, color)
    border_layer.putalpha(dilated)

    # 邊框在下、原圖在上
    out = Image.alpha_composite(border_layer, img)
    return out
