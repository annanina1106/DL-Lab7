"""
src/llm/rag.py — 貼圖情境 RAG 檢索
================================

流程:
  1. 載入 data/sticker_scenarios.json (常見貼圖情緒 / 文案模式語料庫)
  2. 用 sentence-transformers (預設 BAAI/bge-small-zh-v1.5,中文最佳化、體積小)
     將每個情境編碼成向量,快取到 .cache/ 避免重複計算
  3. 給定使用者主題,以餘弦相似度取出最相關的 top-k 個貼圖情境,
     作為 LLM 概念生成的 grounding context (RAG)

穩健性:
  若 sentence-transformers 無法載入 (未安裝 / 離線),自動降級為
  「關鍵字重疊比對」,確保系統仍可運作。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import LLMConfig
from src.utils import get_logger

logger = get_logger("llm.rag")


class ScenarioRetriever:
    def __init__(self, scenarios_file: Path, cfg: LLMConfig, cache_dir: Path):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.scenarios: List[Dict] = json.loads(Path(scenarios_file).read_text(encoding="utf-8"))
        logger.info("已載入 %d 個貼圖情境", len(self.scenarios))

        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._mode = "semantic"

        if cfg.embedding_backend == "local":
            self._try_build_semantic_index()
        else:
            logger.warning("embedding_backend=%s 尚未實作,改用關鍵字檢索", cfg.embedding_backend)
            self._mode = "keyword"

    # ------------------------------------------------------------------
    def _signature(self) -> str:
        h = hashlib.sha256()
        h.update(self.cfg.embedding_model.encode("utf-8"))
        for s in self.scenarios:
            h.update((s["id"] + s["caption"]).encode("utf-8"))
        return h.hexdigest()[:16]

    def _embed_text(self, s: Dict) -> str:
        return f"{s['emotion']} {s['caption']} {s['usage']} {' '.join(s.get('keywords', []))}"

    def _try_build_semantic_index(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("未安裝 sentence-transformers,降級為關鍵字檢索")
            self._mode = "keyword"
            return

        cache_file = self.cache_dir / f"scenario_emb_{self._signature()}.npy"
        try:
            self._model = SentenceTransformer(self.cfg.embedding_model)
            if cache_file.exists():
                self._embeddings = np.load(cache_file)
                logger.info("從快取載入嵌入向量: %s", cache_file.name)
            else:
                texts = [self._embed_text(s) for s in self.scenarios]
                self._embeddings = self._model.encode(texts, normalize_embeddings=True)
                np.save(cache_file, self._embeddings)
                logger.info("已計算並快取嵌入向量 (%d 筆)", len(texts))
            self._mode = "semantic"
        except Exception as e:  # noqa: BLE001
            logger.warning("建立語意索引失敗 (%s),降級為關鍵字檢索", e)
            self._mode = "keyword"

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        if self._mode == "semantic" and self._model is not None and self._embeddings is not None:
            return self._semantic_retrieve(query, k)
        return self._keyword_retrieve(query, k)

    def _semantic_retrieve(self, query: str, k: int) -> List[Dict]:
        q_vec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ q_vec
        top = np.argsort(-scores)[:k]
        out = []
        for idx in top:
            s = dict(self.scenarios[int(idx)])
            s["_score"] = round(float(scores[int(idx)]), 4)
            out.append(s)
        return out

    def _keyword_retrieve(self, query: str, k: int) -> List[Dict]:
        q_chars = set(query)
        scored = []
        for s in self.scenarios:
            text = "".join(s.get("keywords", [])) + s["caption"] + s["usage"] + s["emotion"]
            scored.append((len(q_chars & set(text)), s))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, s in scored[:k]:
            item = dict(s)
            item["_score"] = score
            out.append(item)
        return out

    @property
    def mode(self) -> str:
        return self._mode
