"""
src/llm/concept_engine.py — 貼圖概念生成引擎
==========================================

這是本專題 **Prompt Engineering 的核心**。

輸入: 使用者主題 + 角色描述 + 風格 + 數量 + RAG 檢索到的常見貼圖情境
輸出: N 組結構化貼圖概念,每組包含
        - emotion        情緒 (中文)
        - caption        文案 (繁體中文,口語、有梗,4–10 字)
        - scene          動作/場景描述 (中文)
        - visual_prompt  給擴散模型的英文提示詞 (含一致角色 + 情緒 + 貼圖風格)
        - negative_prompt 負面提示詞

設計重點 (確保「一組風格一致的貼圖」):
  1. 角色一致性 —— 系統提示要求每張 visual_prompt 都嵌入相同的角色外觀描述。
  2. 情緒多樣性 —— 每張要有明顯不同的情緒/動作,避免重複。
  3. 貼圖構圖 —— 強制單一主體、置中、乾淨/純色背景 (利於後續自動去背)。
  4. 風格錨定 —— style preset 提供共用的風格描述字串,附加到每張提示詞。
  5. few-shot 範例 + 嚴格 JSON 輸出格式,確保可被程式穩定解析。
"""

from __future__ import annotations

from typing import Dict, List

from src.diffusion.styles import STYLE_PRESETS, get_style
from src.llm.client import LLMClient
from src.utils import get_logger

logger = get_logger("llm.concept_engine")


SYSTEM_PROMPT = """你是一位專業的 LINE 貼圖與梗圖設計師,擅長把一個主題發想成「一整組風格一致、情緒各異」的貼圖。

你的任務是根據使用者提供的【主題】【角色設定】【風格】,設計指定數量的貼圖概念。

必須嚴格遵守以下規則:
1. 角色一致性:所有貼圖都是「同一個角色」。你會在每張的 visual_prompt 中完整重複相同的角色外觀描述 (character sheet),只改變表情、動作與情緒。
2. 情緒多樣:每張貼圖的情緒與動作都要明顯不同 (例如:開心、崩潰、無奈、得意、想睡…),不可重複。
3. 文案 (caption):用繁體中文、台灣口語、要有梗或共鳴,長度約 4–10 個字,適合日常聊天時使用。
4. 貼圖構圖:visual_prompt 必須描述「單一主體、置中、表情清楚、乾淨純色背景 (例如純白背景)」,方便之後自動去背。不要有複雜場景或多個角色。
5. visual_prompt 用英文撰寫 (擴散模型對英文反應較好),包含:角色外觀 + 當下表情/動作 + 「sticker design, die-cut sticker, bold clean outline, flat color, centered, white background, high quality」之類的貼圖風格關鍵字。
6. negative_prompt 用英文,排除常見缺陷,例如:"text, watermark, multiple characters, blurry, low quality, extra limbs, realistic photo, complex background"。

你只能輸出一個 JSON 陣列,不要有任何其他文字、說明或 markdown 圍欄。
每個元素的格式為:
{
  "emotion": "情緒(中文)",
  "caption": "文案(繁體中文)",
  "scene": "動作或場景的中文描述",
  "visual_prompt": "english prompt with consistent character + emotion + sticker style keywords",
  "negative_prompt": "english negative prompt"
}"""


# 提供一個 few-shot 範例,錨定輸出格式與品質
FEWSHOT_USER = """【主題】一隻愛喝咖啡的上班族柴犬的日常
【角色設定】一隻米白色的柴犬,圓圓的臉,戴細框眼鏡,穿白襯衫
【風格】可愛卡通 (kawaii cartoon)
【共用風格關鍵字】kawaii cartoon sticker, soft pastel colors, thick clean outline, simple shading
【數量】2
【可參考的常見貼圖情境】累到升天 / 謝謝老闆"""

FEWSHOT_ASSISTANT = """[
  {
    "emotion": "疲憊",
    "caption": "我需要咖啡續命",
    "scene": "柴犬癱在辦公桌上,手裡緊抓一杯咖啡,眼神死",
    "visual_prompt": "a cream-white shiba inu dog with a round face, thin-framed glasses, white shirt, exhausted dead-eyed expression slumped over a desk hugging a coffee cup, kawaii cartoon sticker, soft pastel colors, thick clean outline, simple shading, die-cut sticker, centered, white background, high quality",
    "negative_prompt": "text, watermark, multiple characters, blurry, low quality, extra limbs, realistic photo, complex background"
  },
  {
    "emotion": "感激",
    "caption": "老闆我愛你",
    "scene": "柴犬雙手捧著咖啡,眼睛閃閃發亮,一臉感動",
    "visual_prompt": "a cream-white shiba inu dog with a round face, thin-framed glasses, white shirt, sparkling teary grateful eyes holding a cup of coffee with both paws, kawaii cartoon sticker, soft pastel colors, thick clean outline, simple shading, die-cut sticker, centered, white background, high quality",
    "negative_prompt": "text, watermark, multiple characters, blurry, low quality, extra limbs, realistic photo, complex background"
  }
]"""


class ConceptEngine:
    def __init__(self, client: LLMClient):
        self.client = client

    # ------------------------------------------------------------------
    def generate_concepts(
        self,
        theme: str,
        character: str,
        style_key: str,
        count: int,
        retrieved: List[Dict],
    ) -> List[Dict]:
        """產生 N 組貼圖概念。

        Args:
            theme:     使用者主題/情境
            character: 角色外觀描述 (可留空,會請 LLM 自行設計並保持一致)
            style_key: 風格 preset 鍵值 (見 styles.py)
            count:     貼圖數量
            retrieved: RAG 檢索到的常見貼圖情境,作為靈感參考
        """
        style = get_style(style_key)
        character = character.strip() or "(未指定,請你設計一個討喜的角色,並在每張中保持完全一致)"

        # 把 RAG 結果整理成精簡的靈感清單
        inspirations = " / ".join(
            f"{s['caption']}({s['emotion']})" for s in retrieved
        ) or "(無)"

        user_prompt = (
            f"【主題】{theme}\n"
            f"【角色設定】{character}\n"
            f"【風格】{style['label']} ({style['en']})\n"
            f"【共用風格關鍵字】{style['prompt_suffix']}\n"
            f"【數量】{count}\n"
            f"【可參考的常見貼圖情境】{inspirations}\n\n"
            f"請設計 {count} 張情緒各異、但角色與畫風完全一致的貼圖。只輸出 JSON 陣列。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("向 LLM 請求 %d 組貼圖概念 (主題=%s, 風格=%s)", count, theme, style_key)
        data = self.client.chat_json(messages)

        concepts = self._normalize(data, style, count)
        logger.info("成功取得 %d 組概念", len(concepts))
        return concepts

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(data, style: Dict, count: int) -> List[Dict]:
        """容錯處理 LLM 回傳結果,補齊缺漏欄位並確保風格關鍵字存在。"""
        if isinstance(data, dict):
            # 萬一模型包了一層 {"stickers": [...]} 之類
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            raise ValueError("概念生成結果非陣列格式")

        suffix = style["prompt_suffix"]
        default_neg = (
            "text, watermark, multiple characters, blurry, low quality, "
            "extra limbs, realistic photo, complex background"
        )

        out: List[Dict] = []
        for i, item in enumerate(data[:count]):
            if not isinstance(item, dict):
                continue
            vp = str(item.get("visual_prompt", "")).strip()
            # 確保風格關鍵字與貼圖構圖關鍵字一定存在
            if suffix.lower() not in vp.lower():
                vp = f"{vp}, {suffix}" if vp else suffix
            if "white background" not in vp.lower():
                vp += ", die-cut sticker, centered, white background, high quality"

            out.append({
                "index": i + 1,
                "emotion": str(item.get("emotion", "")).strip() or "情緒",
                "caption": str(item.get("caption", "")).strip() or "…",
                "scene": str(item.get("scene", "")).strip(),
                "visual_prompt": vp,
                "negative_prompt": str(item.get("negative_prompt", "")).strip() or default_neg,
            })

        if not out:
            raise ValueError("未能從 LLM 回傳中解析出任何有效概念")
        return out


def available_styles() -> Dict[str, str]:
    """回傳 {key: label} 供 UI 下拉選單使用。"""
    return {k: v["label"] for k, v in STYLE_PRESETS.items()}
