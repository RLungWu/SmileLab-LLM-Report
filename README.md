# SmileLab-LLM-Report

這是一個用來批次評估 USMLE 選擇題的小工具。強烈建議使用本機 Ollama 模型（亦可選擇 OpenAI）。整體使用流程盡量保持「一步到位、少參數」。

**最簡啟動（Ollama 推薦）**
- 安裝 Ollama（https://ollama.com/）
- 下載任一模型（選一個你機器跑得動的）
  - `ollama pull llama3.1:8b`
  - 或 `ollama pull gemma2:2b`
  - 或 `ollama pull qwen2.5:3b`
- 準備資料檔 `data/USMLE.jsonl`（JSONL，每行包含 `question`、`options`、`answer_idx`）
- 安裝依賴並執行（Windows PowerShell）
  - `pip install -r requirements.txt`
  - `python main.py --provider ollama --ollama-model llama3.1:8b --limit 5 --metrics`

以上指令會：讀入資料前 5 題、呼叫本機模型推論、並顯示整體準確率，輸出結果到專案根目錄。

## 常用指令（Ollama）
- 跑全部題目（顯示進度條）
  - `python main.py --provider ollama --ollama-model llama3.1:8b`
- 指定輸入與輸出檔名
  - `python main.py --provider ollama --ollama-model llama3.1:8b --input ./data/USMLE.jsonl --output ./results.json`
- 控制輸出詳略
  - `--quiet` 只印出產生的結果檔路徑
  - `--verbose` 顯示每題題幹、選項、正解與模型回覆
  - `--no-progress` 關閉進度條
- 準確率與評分欄位
  - `--metrics` 跑完印出整體準確率
  - `--add-eval` 在結果中加入 `pred`（模型答案 A–E）與 `is_correct`（是否答對）
  - `--metrics-out metrics.json` 另存統計 JSON（含簡易混淆表）

提示：`--ollama-model` 請換成你本機已下載的模型名稱；以上列舉僅為常見範例。

## 資料格式（JSONL）
每行一筆題目 JSON，必要欄位：
- `question`: 題目文字（string）
- `options`: 選項物件（如 A..E 對應文字）
- `answer_idx`: 正確選項標籤（如 "C"）

範例：
```
{"question":"...","options":{"A":"optA","B":"optB","C":"optC"},"answer_idx":"B"}
```

## OpenAI（可選）
- 建立 `.env` 並填入鍵值
```
OPENAI_API_KEY=你的_api_key
OPENAI_MODEL=gpt-5-mini  # 可改
```
- 執行
  - `python main.py --provider openai --limit 5 --metrics`

## 參數總覽（需要時再看）
- `--input, -i`：輸入 JSONL（預設 `./data/USMLE.jsonl`）
- `--output, -o`：輸出 JSON 路徑（預設自動命名）
- `--limit, -n`：最多處理筆數（0=全部）
- `--provider`：`ollama` 或 `openai`
- `--ollama-model`：Ollama 模型名稱（例 `llama3.1:8b`、`gemma2:2b`）
- `--openai-model`：OpenAI 模型（例 `gpt-5-mini`）
- `--no-env`：不讀取 `.env`
- `--quiet, -q`｜`--verbose, -v`｜`--no-progress`
- `--metrics`｜`--metrics-out`｜`--add-eval`

## 備註
- 預設輸出檔名：`resultsusmle_{provider}_{model}.json`（可用 `--output` 指定）
- 若模型回覆不是單一 A–E，程式會嘗試自動擷取第一個 A–E 作為預測
- 讀檔相容 UTF‑8 與含 BOM 的 JSONL
