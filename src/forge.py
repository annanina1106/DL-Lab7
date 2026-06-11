"""
src/forge.py — StickerForge 主流程協調器
======================================

把四個階段串成一條端到端管線:

    主題 ──▶ [RAG 檢索] ──▶ [LLM 概念生成] ──▶ [擴散生成] ──▶ [後處理] ──▶ 貼圖組
              情境語料        Prompt Eng.        SD/LoRA       去背/描邊/疊字

對外提供:
  - StickerForge.generate(...) : 產生一整組貼圖,逐張 yield 進度
  - StickerForge.pack(...)     : 把貼圖打包成 zip
"""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from config import AppConfig
from src.diffusion import build_generator
from src.llm import ConceptEngine, LLMClient, ScenarioRetriever
from src.postprocess import add_die_cut_border, overlay_caption, remove_background
from src.utils import get_logger

logger = get_logger("forge")


@dataclass
class StickerResult:
    index: int
    emotion: str
    caption: str
    scene: str
    visual_prompt: str
    image: Image.Image          # 最終 RGBA 貼圖
    path: Optional[Path] = None  # 存檔路徑


@dataclass
class ForgeStatus:
    rag_mode: str = ""
    llm_ok: bool = False
    diffusion_backend: str = ""
    messages: List[str] = field(default_factory=list)


class StickerForge:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.status = ForgeStatus()

        # --- LLM (概念生成必須) ---
        self.llm: Optional[LLMClient] = None
        self.concept_engine: Optional[ConceptEngine] = None
        try:
            self.llm = LLMClient(cfg.llm)
            self.concept_engine = ConceptEngine(self.llm)
        except Exception as e:  # noqa: BLE001
            logger.error("LLM 初始化失敗: %s", e)
            self.status.messages.append(f"LLM 初始化失敗: {e}")

        # --- RAG ---
        self.retriever = ScenarioRetriever(cfg.scenarios_file, cfg.llm, cfg.cache_dir)
        self.status.rag_mode = self.retriever.mode

        # --- 擴散生成器 (真實或佔位) ---
        self.generator = build_generator(cfg.diffusion, cfg.diffusion.image_size)
        self.status.diffusion_backend = self.generator.backend_name

    # ------------------------------------------------------------------
    def health_check(self) -> ForgeStatus:
        if self.llm is not None:
            self.status.llm_ok = self.llm.ping()
        return self.status

    # ------------------------------------------------------------------
    def generate(
        self,
        theme: str,
        character: str = "",
        style_key: str = "kawaii",
        count: int = 6,
        seed: int = -1,
        lock_character: bool = True,
        remove_bg: Optional[bool] = None,
        add_border: Optional[bool] = None,
        overlay_text: Optional[bool] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> List[StickerResult]:
        """產生一整組貼圖。

        Args:
            seed:           基礎亂數種子;-1 表示隨機。
            lock_character: True 時每張用 base_seed + index,讓角色更一致;
                            False 時每張完全隨機。
            remove_bg / add_border / overlay_text: None 表示沿用 config 預設。
        """
        import random

        if self.concept_engine is None:
            raise RuntimeError("LLM 未就緒,無法生成概念。請檢查 .env 的 LLM_API_KEY / LLM_BASE_URL。")

        pp = self.cfg.postprocess
        remove_bg = pp.remove_background if remove_bg is None else remove_bg
        add_border = pp.add_border if add_border is None else add_border
        overlay_text = pp.overlay_caption if overlay_text is None else overlay_text

        base_seed = random.randint(0, 2**31 - 1) if seed < 0 else int(seed)
        logger.info("base_seed=%d, lock_character=%s", base_seed, lock_character)

        def _p(frac, msg):
            if progress:
                progress(frac, msg)
            logger.info("[%3d%%] %s", int(frac * 100), msg)

        # --- 階段一: RAG 檢索靈感 ---
        _p(0.05, f"檢索常見貼圖情境 (RAG: {self.retriever.mode})…")
        retrieved = self.retriever.retrieve(theme, k=min(8, max(4, count)))

        # --- 階段二: LLM 概念生成 ---
        _p(0.15, "LLM 正在發想貼圖概念 (情緒 + 文案 + 提示詞)…")
        concepts = self.concept_engine.generate_concepts(
            theme=theme, character=character, style_key=style_key, count=count, retrieved=retrieved
        )

        # --- 階段三+四: 逐張生成 + 後處理 ---
        results: List[StickerResult] = []
        n = len(concepts)
        for i, c in enumerate(concepts):
            frac = 0.2 + 0.75 * (i / max(1, n))
            _p(frac, f"生成第 {i+1}/{n} 張：{c['emotion']}「{c['caption']}」")

            img_seed = (base_seed + i) if lock_character else random.randint(0, 2**31 - 1)
            t0 = time.time()
            raw = self.generator.generate(c, img_seed)
            logger.info("第 %d 張生成耗時 %.2fs", i + 1, time.time() - t0)

            # 後處理鏈
            img = raw
            if remove_bg:
                img = remove_background(img, pp.rembg_model)
            if add_border:
                img = add_die_cut_border(img, pp.border_size)
            if overlay_text and c["caption"]:
                img = overlay_caption(img, c["caption"], pp.font_path or None)
            img = img.convert("RGBA")

            results.append(StickerResult(
                index=c["index"], emotion=c["emotion"], caption=c["caption"],
                scene=c.get("scene", ""), visual_prompt=c["visual_prompt"], image=img,
            ))

        _p(1.0, "完成！")
        return results

    # ------------------------------------------------------------------
    def save_results(self, results: List[StickerResult], out_dir: Optional[Path] = None) -> List[StickerResult]:
        """把每張貼圖存成透明 PNG。"""
        out_dir = out_dir or (self.cfg.output_dir / f"pack_{int(time.time())}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            safe = f"{r.index:02d}_{r.emotion}".replace("/", "_").replace(" ", "")
            path = out_dir / f"{safe}.png"
            r.image.save(path)
            r.path = path
        logger.info("已存檔 %d 張貼圖至 %s", len(results), out_dir)
        return results

    def pack(self, results: List[StickerResult], out_dir: Optional[Path] = None) -> Path:
        """存檔並打包成 zip,回傳 zip 路徑。"""
        out_dir = out_dir or (self.cfg.output_dir / f"pack_{int(time.time())}")
        self.save_results(results, out_dir)
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                if r.path:
                    zf.write(r.path, arcname=r.path.name)
        logger.info("已打包: %s", zip_path)
        return zip_path
