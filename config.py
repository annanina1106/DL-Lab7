"""
config.py — 隨手貼 StickerForge 中央設定模組
==========================================

所有可調參數集中於此,並透過環境變數 (.env) 覆寫。

設計原則:
  1. LLM 後端可插拔 —— Ollama / OpenRouter / Big Pickle 皆為 OpenAI 相容 API,
     只需切換 base_url + api_key + model 三個欄位即可無縫切換。
  2. Diffusion 設定支援 SD1.5 / SDXL,並可選擇性掛載貼圖風格 LoRA。
  3. 後處理 (去背 / 描邊 / 疊字) 皆可獨立開關。
  4. 任何欄位皆有合理預設值;即使未設定環境變數,程式仍可用「佔位生成器」啟動,
     方便在沒有 GPU 的環境先驗證整條流程。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv 為選用
    pass


# ----------------------------------------------------------------------------
# 環境變數讀取輔助函式
# ----------------------------------------------------------------------------
def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ----------------------------------------------------------------------------
# 路徑
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
CACHE_DIR = ROOT_DIR / ".cache"
OUTPUT_DIR = ASSETS_DIR / "outputs"

for _d in (ASSETS_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# LLM 設定
# ----------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """大型語言模型設定 (OpenAI 相容介面)。"""

    backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "openrouter"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "anthropic/claude-3.5-sonnet"))
    temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.8))
    max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 2048))
    timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 60.0))

    # RAG 嵌入向量。"local" 走 sentence-transformers,離線可用且免費。
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "local"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))


# ----------------------------------------------------------------------------
# Diffusion 設定
# ----------------------------------------------------------------------------
@dataclass
class DiffusionConfig:
    """擴散模型設定。"""

    enabled: bool = field(default_factory=lambda: _bool("DIFFUSION_ENABLED", True))
    use_sdxl: bool = field(default_factory=lambda: _bool("USE_SDXL", False))

    # SD1.5 預設使用社群重新上傳的權重;改用 SDXL 請設 USE_SDXL=true
    model_id: str = field(
        default_factory=lambda: _env("DIFFUSION_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    )

    # 貼圖風格 LoRA (本機 .safetensors 路徑或 HuggingFace repo);留空則不掛載
    lora_path: str = field(default_factory=lambda: _env("LORA_PATH", ""))
    lora_scale: float = field(default_factory=lambda: _float("LORA_SCALE", 0.85))

    # 硬體與推論加速
    device: str = field(default_factory=lambda: _env("DIFFUSION_DEVICE", "auto"))   # auto / cuda / mps / cpu
    dtype: str = field(default_factory=lambda: _env("DIFFUSION_DTYPE", "float16"))  # float16 / float32
    scheduler: str = field(default_factory=lambda: _env("DIFFUSION_SCHEDULER", "dpmpp"))  # dpmpp / euler_a / default
    enable_xformers: bool = field(default_factory=lambda: _bool("ENABLE_XFORMERS", False))

    # 生成參數 (貼圖通常為正方形)
    num_inference_steps: int = field(default_factory=lambda: _int("NUM_INFERENCE_STEPS", 26))
    guidance_scale: float = field(default_factory=lambda: _float("GUIDANCE_SCALE", 7.5))
    image_size: int = field(default_factory=lambda: _int("IMAGE_SIZE", 768))


# ----------------------------------------------------------------------------
# 後處理設定
# ----------------------------------------------------------------------------
@dataclass
class PostProcessConfig:
    remove_background: bool = field(default_factory=lambda: _bool("REMOVE_BACKGROUND", True))
    rembg_model: str = field(default_factory=lambda: _env("REMBG_MODEL", "u2net"))
    add_border: bool = field(default_factory=lambda: _bool("ADD_BORDER", True))
    border_size: int = field(default_factory=lambda: _int("BORDER_SIZE", 12))
    overlay_caption: bool = field(default_factory=lambda: _bool("OVERLAY_CAPTION", True))
    # 指定 CJK 字型路徑;留空則自動偵測系統字型
    font_path: str = field(default_factory=lambda: _env("FONT_PATH", ""))


# ----------------------------------------------------------------------------
# 應用程式總設定
# ----------------------------------------------------------------------------
@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)

    data_dir: Path = DATA_DIR
    assets_dir: Path = ASSETS_DIR
    cache_dir: Path = CACHE_DIR
    output_dir: Path = OUTPUT_DIR
    scenarios_file: Path = DATA_DIR / "sticker_scenarios.json"

    default_count: int = field(default_factory=lambda: _int("DEFAULT_STICKER_COUNT", 6))
    max_count: int = field(default_factory=lambda: _int("MAX_STICKER_COUNT", 9))

    server_name: str = field(default_factory=lambda: _env("GRADIO_SERVER_NAME", "127.0.0.1"))
    server_port: int = field(default_factory=lambda: _int("GRADIO_SERVER_PORT", 7860))
    share: bool = field(default_factory=lambda: _bool("GRADIO_SHARE", False))


def load_config() -> AppConfig:
    return AppConfig()


if __name__ == "__main__":
    import json

    cfg = load_config()
    print(json.dumps(
        {
            "llm": cfg.llm.__dict__,
            "diffusion": cfg.diffusion.__dict__,
            "postprocess": cfg.postprocess.__dict__,
            "server": {"name": cfg.server_name, "port": cfg.server_port},
        },
        ensure_ascii=False,
        indent=2,
    ))
