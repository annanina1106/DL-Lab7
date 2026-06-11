"""
src/llm/client.py — 可插拔的 LLM 客戶端
=====================================

Ollama、OpenRouter、Big Pickle (OpenCode) 三者皆提供 OpenAI 相容的 REST API,
因此這裡只用一個 `openai` SDK 物件,透過 base_url / api_key / model 切換後端。

提供:
  - chat()      : 一般文字對話
  - chat_json() : 強制回傳 JSON 並穩健解析 (用於結構化的貼圖概念生成)
  - ping()      : 健康檢查
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from config import LLMConfig
from src.utils import get_logger

logger = get_logger("llm.client")


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("需要 openai 套件,請執行 `pip install openai`。") from e

        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "EMPTY",
            timeout=cfg.timeout,
        )
        logger.info("LLM 後端=%s | 模型=%s | endpoint=%s", cfg.backend, cfg.model, cfg.base_url)

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Any:
        """要求模型回傳 JSON,並以容錯方式解析 (可為 dict 或 list)。"""
        raw = self.chat(messages, temperature=temperature)
        return _extract_json(raw)

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            self.chat([{"role": "user", "content": "ping"}], temperature=0.0, max_tokens=5)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 連線測試失敗: %s", e)
            return False


# ----------------------------------------------------------------------------
# 從可能含雜訊的文字中抽取 JSON (支援物件或陣列)
# ----------------------------------------------------------------------------
def _extract_json(text: str) -> Any:
    # 1. 移除 ```json ... ``` 圍欄
    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    # 2. 直接解析
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 3. 退而求其次:抓出第一個 [ 或 { 到對應結尾的區間
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = candidate.find(open_ch), candidate.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    logger.error("無法解析 LLM 回傳的 JSON,原始內容前 300 字: %s", text[:300])
    raise ValueError("LLM 未回傳有效 JSON")
