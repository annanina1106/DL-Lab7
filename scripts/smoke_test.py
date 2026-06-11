"""
scripts/smoke_test.py — 端到端煙霧測試
=====================================

用「假的 LLM 回應 (canned JSON)」+「佔位生成器」跑完整條管線,
驗證 RAG → 概念 → 生成 → 後處理 → 打包 全部串得起來。
不需要 GPU,也不需要真實 LLM API key。

用法:
    python scripts/smoke_test.py
"""

import os
import sys
from pathlib import Path

# 讓 import 找得到專案根目錄
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 強制使用佔位生成器,避免測試時嘗試載入 GPU 模型
os.environ.setdefault("DIFFUSION_ENABLED", "false")

from config import load_config           # noqa: E402
from src.forge import StickerForge        # noqa: E402
from src.llm.concept_engine import ConceptEngine  # noqa: E402


# --- 假的 LLM 客戶端:回傳固定的概念 JSON ---
class MockLLMClient:
    def chat_json(self, messages, temperature=None):
        return [
            {"emotion": "開心", "caption": "今天超讚der", "scene": "貓咪舉手歡呼",
             "visual_prompt": "a chubby orange cat cheering with paws up, kawaii cartoon sticker, white background",
             "negative_prompt": "text, blurry, multiple characters"},
            {"emotion": "崩潰", "caption": "我裂開了", "scene": "貓咪抱頭崩潰",
             "visual_prompt": "a chubby orange cat holding head in despair, kawaii cartoon sticker, white background",
             "negative_prompt": "text, blurry, multiple characters"},
            {"emotion": "想睡", "caption": "讓我再睡5分鐘", "scene": "貓咪趴著打哈欠",
             "visual_prompt": "a chubby orange cat yawning sleepily lying down, kawaii cartoon sticker, white background",
             "negative_prompt": "text, blurry, multiple characters"},
            {"emotion": "得意", "caption": "就是這麼狂", "scene": "貓咪叉腰得意笑",
             "visual_prompt": "a chubby orange cat smug with paws on hips, kawaii cartoon sticker, white background",
             "negative_prompt": "text, blurry, multiple characters"},
        ]

    def ping(self):
        return True


def main():
    cfg = load_config()
    forge = StickerForge(cfg)

    # 注入 mock,繞過真實 LLM 呼叫
    forge.concept_engine = ConceptEngine(MockLLMClient())
    forge.llm = MockLLMClient()

    print("=" * 60)
    print(f"RAG 模式        : {forge.status.rag_mode}")
    print(f"擴散後端        : {forge.status.diffusion_backend}")
    print("=" * 60)

    results = forge.generate(
        theme="一隻慵懶橘貓的日常心情",
        character="一隻胖橘貓,圓臉大眼,短毛",
        style_key="kawaii",
        count=4,
        seed=42,
        lock_character=True,
    )

    print(f"\n共生成 {len(results)} 張貼圖:")
    for r in results:
        print(f"  [{r.index}] {r.emotion:<4} 「{r.caption}」  size={r.image.size} mode={r.image.mode}")

    zip_path = forge.pack(results)
    print(f"\n打包輸出: {zip_path}")
    assert zip_path.exists(), "zip 未產生"
    assert all(r.path and r.path.exists() for r in results), "PNG 未全部產生"
    assert all(r.image.mode == "RGBA" for r in results), "貼圖應為 RGBA 透明圖"
    print("\n✅ 煙霧測試通過：端到端管線運作正常。")


if __name__ == "__main__":
    main()
