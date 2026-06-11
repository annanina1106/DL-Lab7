"""
app.py — 隨手貼 StickerForge 互動式介面 (階段四:介面封裝)
====================================================

以 Gradio 封裝整條生成管線為可互動的 Web App:
  輸入主題/角色/風格/數量 ──▶ 一鍵生成一組風格一致的貼圖 ──▶ 預覽 + 下載 zip

執行:
    python app.py
預設開在 http://127.0.0.1:7860
"""

from __future__ import annotations

import gradio as gr

from config import load_config
from src.forge import StickerForge
from src.llm import available_styles
from src.utils import get_logger

logger = get_logger("app")

# Gradio 6.0 起 theme/css 從 Blocks() 移到 launch();偵測主版本以相容 4.x/5.x/6.x
try:
    _GRADIO_MAJOR = int(gr.__version__.split(".")[0])
except Exception:  # noqa: BLE001
    _GRADIO_MAJOR = 4

CFG = load_config()
FORGE = StickerForge(CFG)
STYLES = available_styles()  # {key: label}
STYLE_CHOICES = [(label, key) for key, label in STYLES.items()]

EXAMPLES = [
    ["一隻慵懶橘貓的日常心情", "一隻胖橘貓,圓臉大眼,短毛", "可愛卡通", 6],
    ["過勞上班族的內心戲", "一個黑眼圈很重的小人,穿皺襯衫", "手繪塗鴉", 6],
    ["努力減肥中的倉鼠", "一隻圓滾滾的金色倉鼠,鼓鼓的臉頰", "Q版大頭", 6],
    ["戀愛中的少女心情", "一個短髮女孩,臉頰有腮紅", "水彩手帳", 6],
]


def _status_md() -> str:
    s = FORGE.status
    diff = "🟢 真實擴散模型" if s.diffusion_backend == "diffusion" else "🟡 佔位生成器 (未偵測到 GPU/diffusers)"
    rag = "🟢 語意檢索" if s.rag_mode == "semantic" else "🟡 關鍵字檢索 (未安裝 sentence-transformers)"
    llm = "🟢 已連線" if s.llm_ok else ("🔴 未連線/未設定" if FORGE.llm else "🔴 未初始化")
    return (
        f"**系統狀態**　LLM: {llm}　|　RAG: {rag}　|　影像生成: {diff}　"
        f"|　後端模型: `{CFG.llm.model}`"
    )


# 把 forge.StickerResult 轉成 Gradio Gallery 接受的 (image, caption) 並組出說明表
def _to_gallery(results):
    gallery = [(r.image, f"{r.emotion}｜{r.caption}") for r in results]
    rows = [[r.index, r.emotion, r.caption, r.scene, r.visual_prompt] for r in results]
    return gallery, rows


def run_generation(theme, character, style_label, count,
                   seed, lock_character, remove_bg, add_border, overlay_text,
                   progress=gr.Progress()):
    """Gradio 回呼:執行生成並回傳 (gallery, 概念表, zip 路徑, 狀態)。"""
    if not theme or not theme.strip():
        raise gr.Error("請先輸入主題/情境！")

    style_key = STYLES_INV.get(style_label, "kawaii")

    def _cb(frac, msg):
        progress(frac, desc=msg)

    try:
        results = FORGE.generate(
            theme=theme.strip(),
            character=character.strip(),
            style_key=style_key,
            count=int(count),
            seed=int(seed),
            lock_character=lock_character,
            remove_bg=remove_bg,
            add_border=add_border,
            overlay_text=overlay_text,
            progress=_cb,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("生成失敗")
        raise gr.Error(f"生成失敗：{e}")

    zip_path = FORGE.pack(results)
    gallery, rows = _to_gallery(results)
    return gallery, rows, str(zip_path), _status_md()


STYLES_INV = {label: key for key, label in STYLES.items()}


CUSTOM_CSS = """
.gradio-container {max-width: 1180px !important;}
#title-row h1 {margin-bottom: 0;}
.sticker-gallery {min-height: 360px;}
footer {visibility: hidden;}
"""


def build_ui():
    # Gradio < 6:theme/css 放在 Blocks();>= 6:放在 launch()
    blocks_kwargs = {"title": "隨手貼 StickerForge"}
    if _GRADIO_MAJOR < 6:
        blocks_kwargs["theme"] = gr.themes.Soft()
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Row(elem_id="title-row"):
            gr.Markdown(
                "# 🎨 隨手貼 StickerForge\n"
                "輸入一個主題,AI 幫你生成 **一整組風格一致、情緒各異** 的貼圖 "
                "(LLM 發想文案與概念 → 擴散模型作畫 → 自動去背 + 描邊 + 疊字)"
            )
        status = gr.Markdown(_status_md())

        with gr.Row():
            # ---------------- 左側:輸入 ----------------
            with gr.Column(scale=2):
                theme = gr.Textbox(
                    label="主題 / 情境 ✨",
                    placeholder="例如:一隻慵懶橘貓的日常心情",
                    lines=2,
                )
                character = gr.Textbox(
                    label="角色設定 (選填,留空由 AI 設計並保持一致)",
                    placeholder="例如:一隻胖橘貓,圓臉大眼,短毛",
                    lines=2,
                )
                with gr.Row():
                    style = gr.Dropdown(
                        choices=list(STYLES.values()),
                        value=list(STYLES.values())[0],
                        label="貼圖風格",
                    )
                    count = gr.Slider(
                        minimum=1, maximum=CFG.max_count, value=CFG.default_count, step=1,
                        label="貼圖數量",
                    )

                with gr.Accordion("進階設定", open=False):
                    seed = gr.Number(value=-1, label="亂數種子 (-1 = 隨機)", precision=0)
                    lock_character = gr.Checkbox(value=True, label="固定角色 (同基礎種子,讓整組角色更一致)")
                    with gr.Row():
                        remove_bg = gr.Checkbox(value=CFG.postprocess.remove_background, label="自動去背")
                        add_border = gr.Checkbox(value=CFG.postprocess.add_border, label="白色描邊")
                        overlay_text = gr.Checkbox(value=CFG.postprocess.overlay_caption, label="疊上文案")

                run_btn = gr.Button("🚀 開始生成貼圖", variant="primary", size="lg")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[theme, character, style, count],
                    label="範例 (點一下帶入)",
                )

            # ---------------- 右側:輸出 ----------------
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="生成結果", columns=3, height=440, object_fit="contain",
                    elem_classes="sticker-gallery", show_label=True,
                )
                zip_out = gr.File(label="📦 下載貼圖包 (透明 PNG, .zip)")
                concepts = gr.Dataframe(
                    headers=["#", "情緒", "文案", "場景", "Visual Prompt (給擴散模型)"],
                    datatype=["number", "str", "str", "str", "str"],
                    label="🧠 LLM 生成的概念明細 (展示 Prompt Engineering 過程)",
                    wrap=True,
                    column_widths=["5%", "10%", "15%", "25%", "45%"],
                )

        run_btn.click(
            fn=run_generation,
            inputs=[theme, character, style, count, seed, lock_character, remove_bg, add_border, overlay_text],
            outputs=[gallery, concepts, zip_out, status],
        )

        gr.Markdown(
            "---\n"
            "💡 **提示**：若「影像生成」顯示為佔位生成器,代表尚未偵測到 GPU 或 diffusers。"
            "你仍可看到完整的 LLM 概念生成與後處理流程;接上 GPU 並安裝 `torch`/`diffusers`、"
            "於 `.env` 設定 `LORA_PATH` 後即會切換為真實貼圖作畫。"
        )

    return demo


if __name__ == "__main__":
    # 啟動前先做一次健康檢查 (確認 LLM 是否連得上)
    FORGE.health_check()
    app = build_ui()

    launch_kwargs = {
        "server_name": CFG.server_name,
        "server_port": CFG.server_port,
        "share": CFG.share,
    }
    if _GRADIO_MAJOR >= 6:
        launch_kwargs["theme"] = gr.themes.Soft()
        launch_kwargs["css"] = CUSTOM_CSS

    app.queue().launch(**launch_kwargs)
