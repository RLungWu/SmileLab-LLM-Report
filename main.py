import os
import json
import argparse
try:
    import ollama  # type: ignore
    _HAS_OLLAMA = True
except Exception:
    ollama = None  # type: ignore
    _HAS_OLLAMA = False

try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE), ignores comments/empty lines."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Non-fatal if .env cannot be parsed; proceed with existing env
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USMLE QA runner with OpenAI/Ollama providers")
    parser.add_argument("--input", "-i", default="./data/USMLE.jsonl", help="Path to input JSONL dataset")
    parser.add_argument("--output", "-o", default=None, help="Path to output JSON file (default auto-named)")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Limit number of records (0=all)")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=None,
        help="Force provider (default: auto by env)",
    )
    parser.add_argument("--openai-model", default=None, help="OpenAI model name (default from env or gpt-5-mini)")
    parser.add_argument("--ollama-model", default=None, help="Ollama model name (default from env or gemma3)")
    parser.add_argument("--no-env", action="store_true", help="Do not load .env before running")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-question output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed per-question output")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar display")
    return parser.parse_args()

def main():
    args = parse_args()

    # load env (for OPENAI_API_KEY, OPENAI_MODEL, PROVIDER)
    if not args.no_env:
        load_env()

    # load dataset
    USMLE_PATH = args.input
    data_mle = []

    if not os.path.exists(USMLE_PATH):
        raise FileNotFoundError(f"Dataset not found at {USMLE_PATH}")

    with open(USMLE_PATH, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                data_mle.append(json.loads(line))

    result = {}

    # Provider/model selection
    provider = args.provider or os.getenv("PROVIDER")
    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY") else ("ollama" if _HAS_OLLAMA else "none")

    OPENAI_MODEL = args.openai_model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    OLLAMA_MODEL = args.ollama_model or os.getenv("OLLAMA_MODEL", "gemma3")

    # send question to model and get response
    max_count = len(data_mle) if args.limit <= 0 else min(args.limit, len(data_mle))

    # verbosity & progress
    verbose = bool(args.verbose)
    quiet = bool(args.quiet and not verbose)
    use_progress = _HAS_TQDM and (not args.no_progress) and (not verbose) and (not quiet)
    pbar = tqdm(total=max_count, desc="Processing", unit="q") if use_progress else None

    for idx, item in enumerate(data_mle[:max_count]):
        question = item["question"]
        options = item["options"]
        gt_answer_idx = item["answer_idx"]

        if verbose:
            print("Question:", question)
            print("Options:", options)
            print("Ground truth index:", gt_answer_idx)
            print("")

        PROMPT = (
            "Please directly answer the question with A, B, C, D, or E. "
            "Do not provide any explanation.\n"
            f"Question: {question}\n"
            f"Options: {options}\n"
            "Answer:"
        )

        model_used = None
        model_answer = ""

        if provider == "openai":
            model_used = OPENAI_MODEL
            if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                try:
                    client = OpenAI()
                    resp = client.chat.completions.create(
                        model=model_used,
                        messages=[{"role": "user", "content": PROMPT}],
                    )
                    model_answer = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    model_answer = f"<error: {e}>"
            else:
                if not _HAS_OPENAI:
                    model_answer = "<error: openai package not installed>"
                elif not os.getenv("OPENAI_API_KEY"):
                    model_answer = "<error: OPENAI_API_KEY missing>"
        elif provider == "ollama":
            model_used = OLLAMA_MODEL
            if _HAS_OLLAMA:
                try:
                    response = ollama.chat(
                        model=model_used,
                        messages=[{"role": "user", "content": PROMPT}],
                    )
                    model_answer = response.get("message", {}).get("content", "").strip()
                except Exception as e:
                    model_answer = f"<error: {e}>"
            else:
                model_answer = "<error: ollama package not installed>"
        else:
            model_used = "none"
            model_answer = "<error: no provider configured>"

        if verbose:
            print("Model Response:", model_answer)

        result[idx] = {
            "provider": provider,
            "model": model_used,
            "question": question,
            "options": options,
            "answer": gt_answer_idx,
            "model_response": model_answer,
        }

        if pbar is not None:
            pbar.update(1)

    # save response to output file
    suffix = OPENAI_MODEL if provider == "openai" else OLLAMA_MODEL if provider == "ollama" else "none"
    out_path = args.output or f"resultsusmle_{provider}_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    if pbar is not None:
        pbar.close()
    if not quiet:
        print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
