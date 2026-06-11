"""
scripts/check_setup.py — 環境自我檢查
====================================

啟動前快速確認各元件是否就緒,並給出修正建議。

用法:
    python scripts/check_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _check(label, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, str(e)
    icon = "✅" if ok else "⚠️ "
    print(f"{icon} {label:<22} {detail}")
    return ok


def main():
    from config import load_config

    cfg = load_config()
    print("=" * 64)
    print(" 隨手貼 StickerForge — 環境檢查")
    print("=" * 64)

    # --- 核心套件 ---
    def _pillow():
        import PIL
        return True, f"Pillow {PIL.__version__}"

    def _numpy():
        import numpy
        return True, f"numpy {numpy.__version__}"

    def _openai():
        import openai
        return True, f"openai {openai.__version__}"

    def _gradio():
        import gradio
        return True, f"gradio {gradio.__version__}"

    _check("Pillow", _pillow)
    _check("numpy", _numpy)
    _check("openai (LLM client)", _openai)
    _check("gradio (UI)", _gradio)

    # --- RAG 嵌入 ---
    def _st():
        import sentence_transformers
        return True, f"sentence-transformers {sentence_transformers.__version__} (語意檢索可用)"
    if not _check("sentence-transformers", _st):
        print("     → 未安裝則 RAG 自動降級為關鍵字檢索 (仍可運作)")

    # --- 擴散後端 ---
    def _torch():
        import torch
        cuda = torch.cuda.is_available()
        dev = "CUDA" if cuda else ("MPS" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "CPU only")
        return True, f"torch {torch.__version__} | 裝置: {dev}"
    if not _check("torch", _torch):
        print("     → 未安裝則影像生成使用佔位生成器 (流程可跑,非真實作畫)")

    def _diffusers():
        import diffusers
        return True, f"diffusers {diffusers.__version__}"
    _check("diffusers", _diffusers)

    # --- 後處理 ---
    def _rembg():
        import rembg  # noqa: F401
        return True, "rembg 可用 (自動去背)"
    if not _check("rembg (去背)", _rembg):
        print("     → 未安裝則略過去背,輸出為不透明 PNG")

    def _font():
        from src.postprocess.caption import find_cjk_font
        p = find_cjk_font()
        return (p is not None), (p or "找不到 CJK 字型,將略過疊字")
    if not _check("CJK 字型", _font):
        print("     → 執行 `sudo apt install fonts-noto-cjk` 或設定 FONT_PATH")

    # --- LLM 連線 ---
    print("-" * 64)
    print(f" LLM 後端  : {cfg.llm.backend}")
    print(f" Base URL  : {cfg.llm.base_url}")
    print(f" Model     : {cfg.llm.model}")
    key_set = cfg.llm.api_key not in ("", "EMPTY")
    print(f" API Key   : {'已設定' if key_set else '⚠️  未設定 (請在 .env 設定 LLM_API_KEY)'}")

    if key_set:
        try:
            from src.llm import LLMClient
            client = LLMClient(cfg.llm)
            print(f" 連線測試  : {'✅ 成功' if client.ping() else '🔴 失敗 (檢查 key / base_url / model)'}")
        except Exception as e:  # noqa: BLE001
            print(f" 連線測試  : 🔴 失敗 ({e})")

    print("=" * 64)


if __name__ == "__main__":
    main()
