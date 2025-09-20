import os
import json
import argparse
import re
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
    parser.add_argument("--metrics", action="store_true", help="Compute and print accuracy metrics")
    parser.add_argument("--metrics-out", default=None, help="Write metrics JSON to this path")
    parser.add_argument("--add-eval", action="store_true", help="Add 'pred' and 'is_correct' to results")
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

    def normalize_choice(text: str | None) -> str | None:
        if not isinstance(text, str):
            return None
        s = text.strip().upper()
        # find standalone A-E first
        m = re.search(r"\b([A-E])\b", s)
        if m:
            return m.group(1)
        # tolerate formats like "C)" or "C." or "Answer: C"
        m = re.search(r"\b([A-E])(?=[\s\)\].,:;!?-])", s)
        if m:
            return m.group(1)
        return None

    # metrics containers
    total = 0
    correct = 0
    labels = ["A", "B", "C", "D", "E"]
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}

    # verbosity & progress
    verbose = bool(args.verbose)
    quiet = bool(args.quiet and not verbose)
    use_progress = _HAS_TQDM and (not args.no_progress) and (not verbose) and (not quiet)
    pbar = tqdm(total=max_count, desc="Processing", unit="q") if use_progress else None

    for idx, item in enumerate(data_mle[:max_count]):
        question = item["question"]
        options = item["options"]
        gt_answer_idx = str(item["answer_idx"]).strip().upper()

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

        # evaluation
        pred = normalize_choice(model_answer)
        is_correct = (pred == gt_answer_idx)

        if gt_answer_idx in confusion and pred in confusion.get(gt_answer_idx, {}):
            confusion[gt_answer_idx][pred] += 1
        total += 1
        if is_correct:
            correct += 1

        result[idx] = {
            "provider": provider,
            "model": model_used,
            "question": question,
            "options": options,
            "answer": gt_answer_idx,
            "model_response": model_answer,
        }

        if args.add_eval:
            result[idx]["pred"] = pred
            result[idx]["is_correct"] = is_correct

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

    # metrics reporting
    if args.metrics:
        acc = (correct / total) if total else 0.0
        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "provider": provider,
            "model": OPENAI_MODEL if provider == "openai" else OLLAMA_MODEL if provider == "ollama" else "none",
            "confusion": confusion,
        }
        if not quiet:
            print(f"Accuracy: {correct}/{total} = {acc:.2%}")
        if args.metrics_out:
            with open(args.metrics_out, "w", encoding="utf-8") as mf:
                json.dump(metrics, mf, indent=2, ensure_ascii=False)
            if not quiet:
                print(f"Saved metrics to {args.metrics_out}")

if __name__ == "__main__":
    main()
