# SmileLab-LLM-Report

Lightweight script to evaluate multiple‑choice USMLE questions using either OpenAI (ChatGPT) or local Ollama models. Provides a small CLI via argparse and optional .env configuration.

**Features**
- OpenAI or Ollama provider selection (auto-detect via env)
- Simple JSONL dataset loader (UTF‑8/BOM safe)
- CLI flags for input/output/limit/provider/models
- Quiet/verbose output and optional progress bar
- Writes a JSON results file with model responses

## Setup

1) Create and activate a virtual environment (Windows PowerShell):
- `python -m venv .venv`
- `.\.venv\Scripts\Activate`

2) Install dependencies:
- `pip install -r requirements.txt`

3) Configure environment (optional but recommended for OpenAI):
- Create a `.env` file with:
  - `OPENAI_API_KEY=your_api_key`
  - Optional: `OPENAI_MODEL=gpt-5-mini`
  - Optional: `PROVIDER=openai` (or leave unset for auto-detect)

## Dataset Format

The input file is JSON Lines (one JSON object per line). Each object should include:
- `question`: string
- `options`: object mapping labels (e.g., A..E) to option strings
- `answer_idx`: string or label for the correct option (e.g., `"C"`)

Example line:
```
{"question":"...","options":{"A":"optA","B":"optB","C":"optC"},"answer_idx":"B"}
```

## CLI Usage

- Basic (auto-detect provider, use OpenAI if `OPENAI_API_KEY` is set):
- `python main.py`

- Limit to first N records:
- `python main.py --limit 5`

- Specify input/output files explicitly:
- `python main.py --input ./data/USMLE.jsonl --output ./results.json`

- Force provider and set models:
- `python main.py --provider openai --openai-model gpt-5-mini`
- `python main.py --provider ollama --ollama-model gemma3`

- Skip loading .env:
- `python main.py --no-env`

- Output control:
- `python main.py --quiet` (只顯示最後輸出路徑)
- `python main.py --verbose`（列出每題題目、選項、正解與模型回覆）
- `python main.py --no-progress`（停用進度條，預設在非 quiet/verbose 時顯示）

Outputs a file named like `resultsusmle_openai_gpt-5-mini.json` by default, or the path you pass with `--output`.

## Notes

- OpenAI provider requires `OPENAI_API_KEY`.
- Ollama provider requires the local Ollama service and the requested model available.
- The loader is tolerant of UTF-8 BOM markers when reading JSONL.
