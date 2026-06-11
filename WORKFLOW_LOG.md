# 🤖 Agent 協作紀錄 (Workflow Log)

> 專題:隨手貼 StickerForge — 個人化 AI 貼圖生成系統
> 本文件記錄以 AI Agent 輔助開發的全週期:關鍵 Prompt、工具組合,以及 Agent 協助解決的技術問題。

---

## 🛠️ 使用的工具組合

| 用途 | 工具 |
|---|---|
| 規劃 / 程式碼生成 Agent | Claude (對話式)、claude cli / open code (終端機自動化) |
| LLM 推論後端 (專題執行時) | OpenRouter(開發)、Ollama(離線驗證)、Big Pickle(備援) |
| 影像生成 | Stable Diffusion 1.5 + 貼圖風格 LoRA (diffusers) |
| 去背 | rembg (U2Net) |
| RAG 嵌入 | sentence-transformers (BAAI/bge-small-zh-v1.5) |
| 介面 | Gradio |
| 版本控制 | Git / GitHub |

---

## 階段一:發想與企劃

**目標:** 給 Agent 初始 context,請其產出符合「LLM + Diffusion 結合」的專題提案。

**關鍵 Prompt(節錄):**
> 「我要做一份深度生成模型的期末專題,必須結合 LLM 與 Diffusion,最終要有可互動的 App。請提出幾個難度適中、demo 效果好、技術點齊全的題目,並標明各自的 LLM 用途與 Diffusion 用途。」

**Agent 產出與決策:**
Agent 提出多個方向(詩詞水墨、書法臨摹、AI 繪本、邊緣端即時生成、貼圖生成…)。最終選定 **個人化貼圖生成器**,理由:
- 開發成本低、展示討喜,適合在有限時間內完成可運行系統。
- 技術點完整:RAG + Prompt Engineering(LLM)、LoRA + Pipeline 客製化 + 推論加速(Diffusion),外加「自動去背」這個有記憶點的後處理。
- 輸出(透明 PNG 貼圖)直覺好懂,錄影 demo 一目了然。

**確立的專題目標:** 輸入主題 → 自動生成「一整組風格一致、情緒各異」的貼圖包。

---

## 階段二:架構設計與任務拆解

**目標:** 請 Agent 把題目拆成具體模組,定義系統架構與資料交換格式。

**關鍵 Prompt(節錄):**
> 「把這個貼圖生成器拆成可獨立開發、可測試的模組。LLM 後端要能在 Ollama / OpenRouter / Big Pickle 之間切換。定義各階段之間的資料契約。並且:就算沒有 GPU,也要能跑完整條流程做開發驗證。」

**Agent 提出的架構決策:**
1. **四階段管線**:RAG 檢索 → LLM 概念生成 → 擴散生成 → 後處理,由 `src/forge.py` 統一協調。
2. **資料契約**:LLM 階段一律輸出結構化 JSON 陣列
   `{emotion, caption, scene, visual_prompt, negative_prompt}`,作為擴散階段的輸入介面 —— 前後端以此 schema 解耦。
3. **後端抽象**:因 Ollama / OpenRouter / Big Pickle 皆為 OpenAI 相容 API,設計單一 `LLMClient`,只切換 `base_url / api_key / model`。
4. **可測試性設計(關鍵決策)**:擴散階段拆成「真實生成器」與「佔位生成器」雙實作,用工廠函式依環境自動選擇。讓沒有 GPU 的開發機也能驗證 LLM→概念→後處理→打包 的完整鏈路。

**任務拆解(交付給程式碼生成階段):**
- [x] `config.py` — 環境變數驅動的中央設定
- [x] `llm/client.py` — 可插拔 LLM 客戶端 + 穩健 JSON 解析
- [x] `llm/rag.py` — RAG 檢索 + 降級備援
- [x] `llm/concept_engine.py` — 概念生成 system prompt / few-shot
- [x] `diffusion/pipeline.py` — SD/SDXL + LoRA + 佔位生成器
- [x] `postprocess/` — 去背、描邊、疊字
- [x] `forge.py` — 串接協調器
- [x] `app.py` — Gradio 介面
- [x] `scripts/` — 環境檢查與煙霧測試

---

## 階段三:程式碼生成與實作

**目標:** 以 Agent 實作各模組,開發者擔任系統規劃者,提供精確 context 並處理除錯。

### 關鍵 Prompt 範例

**(a) 概念生成的 Prompt Engineering**
> 「寫一個 system prompt,讓 LLM 根據主題產出 N 組貼圖概念。硬性要求:① 所有貼圖是同一角色(每張 visual_prompt 都重複相同角色外觀);② 情緒各異不重複;③ 文案用台灣口語、4–10 字、要有梗;④ 構圖必須單一主體、置中、純色背景(方便去背);⑤ 只輸出 JSON 陣列。再附一組 few-shot 範例錨定格式。」

**(b) 推論加速**
> 「擴散管線要加上推論加速:fp16 半精度、DPM-Solver++ 排程器、attention slicing,xformers 設成可選。LoRA 用 load_lora_weights + fuse_lora 掛載,scale 可調。」

**(c) 後處理**
> 「用 rembg 去背成透明 PNG,再做 LINE 風格的白色模切外框(對 alpha 膨脹後填白再疊回)。中文文案疊在底部,白字黑邊,字級依長度自動縮放。所有步驟在缺套件/字型時要能優雅略過。」

### 🔧 Agent 協助解決的技術問題(實際除錯紀錄)

> 這一段是整份 log 的重點 —— 記錄開發過程中真正踩到、並由 Agent 協助解決的問題。

1. **Gradio 6.0 API 變動**
   - 問題:測試時跳出 `UserWarning: theme/css have been moved from the Blocks constructor to the launch() method in Gradio 6.0`。直接寫 `gr.Blocks(theme=..., css=...)` 在新版會被忽略。
   - 解法:在 `app.py` 偵測 `gr.__version__` 主版本號,`< 6` 時把 theme/css 放 `Blocks()`,`>= 6` 時改放 `launch()`,讓程式同時相容 4.x / 5.x / 6.x。

2. **沒有 GPU 也要能開發**
   - 問題:擴散模型需要 GPU,但開發機沒有,無法驗證整條流程。
   - 解法:設計 `PlaceholderGenerator`(純 PIL),與真實生成器實作相同 `generate()` 介面,由 `build_generator()` 工廠依環境自動切換。後續用 `smoke_test.py` 證明端到端管線在無 GPU、無 API key 下可完整跑通並輸出 PNG。

3. **LLM 回傳 JSON 不穩定**
   - 問題:不同模型(尤其開源模型)常把 JSON 包在 ```json 圍欄、或前後加說明文字,直接 `json.loads` 會失敗。
   - 解法:`client._extract_json()` 三層容錯:先抓 ```json 圍欄 → 直接解析 → 退而抓第一個 `[`/`{` 到對應結尾的區間。並在 `concept_engine._normalize()` 補齊缺漏欄位、強制注入風格與構圖關鍵字。

4. **RAG 在離線/未安裝套件時崩潰**
   - 問題:sentence-transformers 未安裝或無法下載權重時,整個系統會起不來。
   - 解法:`ScenarioRetriever` 加入「關鍵字重疊比對」降級模式,語意索引建立失敗時自動切換,並把目前模式顯示在 UI 狀態列。嵌入向量也快取到 `.cache/` 避免每次重算。

5. **中文文案疊字的字型問題**
   - 問題:PIL 預設字型不含中文,且不同作業系統字型路徑不同。
   - 解法:`find_cjk_font()` 依序搜尋 `FONT_PATH` → `assets/fonts/` → 各系統常見 Noto/思源/微軟正黑/蘋方路徑;找不到時記錄警告並略過疊字(不中斷),UI 仍會以文字顯示文案。

6. **角色一致性**
   - 問題:同一組貼圖角色長相飄移。
   - 解法:雙管齊下 —— (i) Prompt 端要求每張重複完整角色外觀描述;(ii) 影像端提供「固定角色」選項,以 `base_seed + index` 取代完全隨機種子。並在文件標註進階方向(角色 LoRA / IP-Adapter)。

7. **白色描邊的 kernel 上限**
   - 問題:`ImageFilter.MaxFilter` 對過大的 kernel 會丟出 `ValueError`。
   - 解法:`add_die_cut_border()` 捕捉例外,改用「多次小核膨脹」近似大範圍膨脹,確保任意 `border_size` 都能運作。

---

## 階段四:介面封裝與總結

**目標:** 指示 Agent 以 Gradio 封裝後端為可互動 App,並輔助撰寫技術文件。

**關鍵 Prompt(節錄):**
> 「用 Gradio Blocks 做介面:左側輸入(主題/角色/風格/數量 + 進階設定),右側顯示貼圖 Gallery、LLM 概念明細表、zip 下載。生成過程要有進度條。狀態列要即時顯示 LLM / RAG / 影像生成三者的就緒狀態,讓 demo 時一眼看出系統組態。」

**產出:**
- `app.py`:含進度回呼、系統狀態列、範例一鍵帶入、概念明細表(展示 Prompt Engineering 過程)、貼圖包 zip 下載。
- 由 Agent 協助撰寫 `README.md`(架構圖、技術對應表、安裝/執行步驟、疑難排解)與本 `WORKFLOW_LOG.md`。

**驗證:** `python scripts/smoke_test.py` 通過,確認四階段端到端串接無誤;`build_ui()` 在 Gradio 6.x 下成功建立介面。

---

## 💡 心得與反思

- **「系統規劃者」的角色**:Agent 能快速產出程式碼,但架構決策(雙生成器、資料契約、降級策略)需要開發者主導,否則容易得到能跑卻不穩健的結果。最有價值的 prompt 往往是「限制條件」而非「請幫我寫」。
- **穩健性是設計出來的**:本專題每個外部依賴(GPU、嵌入模型、rembg、字型)都預設了降級路徑,這讓系統在任何環境都能展示,也大幅降低 demo 翻車風險。
- **可測試性優先**:先把「無 GPU 也能跑」這個約束放進架構,使得開發、除錯、展示都更順暢。

> 📝 備註:本 log 為開發主軸的真實紀錄。實際繳交時,請依你自己執行時的對話、額外嘗試的 prompt、以及接上真實 GPU/LoRA 後遇到的狀況再行補充,使其完整反映個人的開發歷程。
