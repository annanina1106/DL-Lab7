"""
src/diffusion/pipeline.py — 貼圖影像生成
======================================

提供兩種生成器,透過工廠函式 build_generator() 自動選擇:

  1. StickerDiffusionPipeline — 真實的 Stable Diffusion / SDXL 生成器
     技術點:
       * LoRA 權重掛載 (load_lora_weights) —— 套用貼圖風格
       * Pipeline 客製化 —— 切換 scheduler、安全檢查器關閉
       * 推論加速 —— fp16 半精度、DPM-Solver++ 排程器、attention slicing、(選用) xformers
       * 角色一致性 —— 可固定 base seed,僅以 index 微調,讓整組貼圖角色更穩定

  2. PlaceholderGenerator — 純 PIL 佔位生成器 (無需 GPU / torch)
     當 DIFFUSION_ENABLED=false、缺少 torch/diffusers、或無可用裝置時自動啟用。
     會畫出帶有情緒文字的彩色佔位圖,讓你在沒有 GPU 的環境也能跑完整條流程、驗證
     LLM → 概念 → 後處理 → 打包 的端到端管線。

兩者都實作相同介面:generate(concept: dict, seed: int) -> PIL.Image (RGB)。
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Protocol

from PIL import Image, ImageDraw, ImageFont

from config import DiffusionConfig
from src.utils import get_logger

logger = get_logger("diffusion.pipeline")


class StickerGenerator(Protocol):
    backend_name: str

    def generate(self, concept: Dict, seed: int) -> Image.Image: ...


# ============================================================================
# 工廠函式
# ============================================================================
def build_generator(cfg: DiffusionConfig, image_size: int) -> StickerGenerator:
    """依設定與環境自動選擇生成器。"""
    if not cfg.enabled:
        logger.info("DIFFUSION_ENABLED=false → 使用佔位生成器")
        return PlaceholderGenerator(image_size)

    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError:
        logger.warning("未安裝 torch / diffusers → 退回佔位生成器 (僅供流程驗證)")
        return PlaceholderGenerator(image_size)

    try:
        return StickerDiffusionPipeline(cfg, image_size)
    except Exception as e:  # noqa: BLE001
        logger.warning("初始化擴散管線失敗 (%s) → 退回佔位生成器", e)
        return PlaceholderGenerator(image_size)


# ============================================================================
# 真實擴散生成器
# ============================================================================
class StickerDiffusionPipeline:
    backend_name = "diffusion"

    def __init__(self, cfg: DiffusionConfig, image_size: int):
        import torch
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
        )

        self.cfg = cfg
        self.image_size = image_size
        self.device = self._resolve_device(cfg.device)
        self.dtype = torch.float16 if (cfg.dtype == "float16" and self.device == "cuda") else torch.float32

        logger.info("載入擴散模型: %s (device=%s, dtype=%s)", cfg.model_id, self.device, self.dtype)

        PipeCls = StableDiffusionXLPipeline if cfg.use_sdxl else StableDiffusionPipeline

        # SDXL 的 pipeline 不接受 safety_checker 參數,僅 SD1.5 需要 (並關閉以避免誤判貼圖)
        from_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }
        if not cfg.use_sdxl:
            from_kwargs["safety_checker"] = None
            from_kwargs["requires_safety_checker"] = False

        pipe = PipeCls.from_pretrained(cfg.model_id, **from_kwargs)

        # --- Pipeline 客製化:切換排程器 (推論加速) ---
        if cfg.scheduler == "dpmpp":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
            )
        elif cfg.scheduler == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(self.device)

        # --- LoRA 權重掛載:套用貼圖風格 ---
        if cfg.lora_path:
            try:
                pipe.load_lora_weights(cfg.lora_path)
                pipe.fuse_lora(lora_scale=cfg.lora_scale)
                logger.info("已掛載 LoRA: %s (scale=%.2f)", cfg.lora_path, cfg.lora_scale)
            except Exception as e:  # noqa: BLE001
                logger.warning("LoRA 掛載失敗 (%s),改用基礎模型", e)

        # --- 記憶體 / 推論最佳化 ---
        if self.device == "cuda":
            pipe.enable_attention_slicing()
            if cfg.enable_xformers:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("已啟用 xformers")
                except Exception as e:  # noqa: BLE001
                    logger.warning("xformers 啟用失敗: %s", e)

        self.pipe = pipe
        self._torch = torch

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(pref: str) -> str:
        import torch

        if pref != "auto":
            return pref
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    def generate(self, concept: Dict, seed: int) -> Image.Image:
        generator = self._torch.Generator(device=self.device).manual_seed(int(seed))
        result = self.pipe(
            prompt=concept["visual_prompt"],
            negative_prompt=concept.get("negative_prompt", ""),
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            height=self.image_size,
            width=self.image_size,
            generator=generator,
        )
        return result.images[0].convert("RGB")


# ============================================================================
# 佔位生成器 (無需 GPU)
# ============================================================================
class PlaceholderGenerator:
    backend_name = "placeholder"

    _PALETTE = [
        (255, 214, 165), (255, 173, 173), (202, 255, 191), (160, 196, 255),
        (189, 178, 255), (255, 198, 255), (253, 255, 182), (155, 246, 255),
    ]

    def __init__(self, image_size: int):
        self.image_size = image_size
        logger.info("使用佔位生成器 (image_size=%d) — 僅供流程驗證,非真實 AI 生成圖", image_size)

    def generate(self, concept: Dict, seed: int) -> Image.Image:
        size = self.image_size
        # 以情緒文字決定底色,讓每張不同但可重現
        key = (concept.get("emotion", "") + str(seed)).encode("utf-8")
        idx = int(hashlib.md5(key).hexdigest(), 16) % len(self._PALETTE)
        bg = self._PALETTE[idx]

        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        # 畫一個簡單的圓形「角色」佔位
        margin = size // 6
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=(255, 255, 255),
            outline=(60, 60, 60),
            width=max(3, size // 100),
        )
        # 兩個眼睛 + 嘴
        eye_y = size // 2 - size // 12
        for ex in (size // 2 - size // 9, size // 2 + size // 9):
            draw.ellipse([ex - size // 40, eye_y - size // 40, ex + size // 40, eye_y + size // 40], fill=(40, 40, 40))
        draw.arc([size // 2 - size // 10, size // 2, size // 2 + size // 10, size // 2 + size // 8], 200, 340, fill=(40, 40, 40), width=max(2, size // 120))

        # 情緒標籤 (置於頂部,避免與底部文案重疊)
        emotion = concept.get("emotion", "")
        font = _load_placeholder_font(size // 16)
        if emotion:
            _draw_centered_text(draw, f"[{emotion}]", (size // 2, margin // 2), font, fill=(120, 120, 120))

        return img


# ----------------------------------------------------------------------------
# 字型輔助 (佔位生成器用,簡化版;正式疊字在 postprocess/caption.py)
# ----------------------------------------------------------------------------
def _load_placeholder_font(px: int) -> ImageFont.FreeTypeFont:
    from src.postprocess.caption import find_cjk_font

    path = find_cjk_font()
    if path:
        try:
            return ImageFont.truetype(path, px)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def _draw_centered_text(draw, text, center, font, fill):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:  # noqa: BLE001
        w, h = len(text) * 10, 14
    draw.text((center[0] - w / 2, center[1] - h / 2), text, font=font, fill=fill)
