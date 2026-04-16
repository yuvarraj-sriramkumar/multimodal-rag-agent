"""
evaluate.py — Ablation study: RAG OFF vs RAG ON on merve/vqav2-small.

Uses merve/vqav2-small which bundles PIL images directly — no URL fetching.

CLI:
    python src/evaluate.py --n_samples 100 --ablation
    python src/evaluate.py --n_samples 50 --use_rag          # single condition
    python src/evaluate.py --n_samples 50                    # RAG OFF only
"""

import argparse
import json
import re
import string
import sys
import tempfile
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Ensure src/ is importable when called as a script
_src = str(Path(__file__).parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from rag import format_context, retrieve
from vision import analyze_image

# ---------------------------------------------------------------------------
# Text normalisation helpers (VQA official evaluation style)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = _ARTICLES.sub("", text)
    text = text.translate(_PUNCTUATION)
    return " ".join(text.split())


def extract_final_answer(cot_output: str) -> str:
    """
    Parse the **Answer** section from a chain-of-thought response.
    Falls back to the last non-empty line if the section header is absent.
    """
    # Split on the Answer header (bold markdown or plain)
    header_pattern = re.compile(r"\*{0,2}answer\*{0,2}", re.IGNORECASE)
    parts = header_pattern.split(cot_output)

    if len(parts) >= 2:
        # Take everything after the last Answer header
        after = parts[-1].strip()
        # Stop at the next section header (**Something** or ## Something)
        stop = re.search(r"\n\s*(\*\*\w|\#{1,3}\s)", after)
        section = after[: stop.start()] if stop else after
        # Return the first non-empty line
        lines = [l.strip() for l in section.splitlines() if l.strip()]
        if lines:
            return lines[0]

    # Fallback: last non-empty line
    lines = [l.strip() for l in cot_output.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute word-overlap F1 between normalised prediction and ground truth.
    Returns 0.0 if either is empty after normalisation.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# VQA majority-vote ground truth
# ---------------------------------------------------------------------------

def _majority_answer(sample: dict) -> str:
    """
    Return the most frequent answer from the 'answers' list.
    Falls back to 'multiple_choice_answer' if the list is unavailable.
    """
    raw_answers = sample.get("answers") or []
    if raw_answers:
        counts: Counter = Counter()
        for a in raw_answers:
            ans = a.get("answer", "") if isinstance(a, dict) else str(a)
            counts[ans] += 1
        return counts.most_common(1)[0][0]
    return sample.get("multiple_choice_answer", "")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    n_samples: int = 100,
    use_rag: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Run GPT-4o on *n_samples* VQAv2 validation images with or without RAG.

    Returns a dict with keys: accuracy, avg_f1, results (list of per-sample dicts).
    """
    label = "rag_on" if use_rag else "rag_off"
    if output_path is None:
        output_path = f"eval_results_{label}.json"

    print(f"\n{'='*60}")
    print(f"  Condition : {'RAG ON' if use_rag else 'RAG OFF'}")
    print(f"  Samples   : {n_samples}")
    print(f"  Output    : {output_path}")
    print(f"{'='*60}\n")

    # merve/vqav2-small ships PIL images inline — no streaming, no URL fetching
    dataset = load_dataset("merve/vqav2-small", split="validation")
    # Shuffle once for variety; fixed seed for reproducibility across conditions
    dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    results = []
    exact_matches = 0
    total_f1 = 0.0
    errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        pbar = tqdm(total=n_samples, desc=f"Evaluating ({label})")

        for i, sample in enumerate(dataset):
            question = sample["question"]
            # merve/vqav2-small only has multiple_choice_answer (no answers list)
            ground_truth = sample.get("multiple_choice_answer", "")
            image_id = sample.get("image_id", i)

            # Images are PIL objects already — just save to temp JPEG
            pil_image = sample["image"]
            img_path = str(Path(tmpdir) / f"{i}.jpg")
            pil_image.convert("RGB").save(img_path, format="JPEG")

            # Optionally retrieve RAG context
            context: str | None = None
            if use_rag:
                hits = retrieve(question)
                context = format_context(hits) if hits else None

            # Call GPT-4o
            try:
                cot_output = analyze_image(
                    image_path=img_path,
                    question=question,
                    context=context,
                )
                raw_prediction = extract_final_answer(cot_output)
            except Exception as exc:
                errors += 1
                pbar.set_postfix(errors=errors)
                results.append({
                    "id": i,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": "",
                    "exact_match": False,
                    "token_f1": 0.0,
                    "error": str(exc),
                })
                pbar.update(1)
                continue

            # Score
            norm_pred = normalize_answer(raw_prediction)
            norm_gt = normalize_answer(ground_truth)
            em = norm_pred == norm_gt
            f1 = token_f1(raw_prediction, ground_truth)

            if em:
                exact_matches += 1
            total_f1 += f1

            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": raw_prediction,
                "exact_match": em,
                "token_f1": round(f1, 4),
            })

            completed = i + 1
            pbar.set_postfix(
                EM=f"{exact_matches/completed:.1%}",
                F1=f"{total_f1/completed:.3f}",
            )
            pbar.update(1)

        pbar.close()

    n = len(results)
    accuracy = exact_matches / n if n else 0.0
    avg_f1 = total_f1 / n if n else 0.0

    summary = {
        "condition": label,
        "n_samples": n,
        "n_errors": errors,
        "accuracy": round(accuracy, 4),
        "avg_f1": round(avg_f1, 4),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExact Match Accuracy : {accuracy:.1%}  ({exact_matches}/{n})")
    print(f"Average Token F1     : {avg_f1:.4f}")
    print(f"Errors               : {errors}")
    print(f"Results saved to     : {output_path}\n")

    return summary


# ---------------------------------------------------------------------------
# Ablation comparison printer
# ---------------------------------------------------------------------------


def print_comparison(off: dict, on: dict) -> None:
    """Print a formatted side-by-side ablation table."""
    w = 38

    def row(label: str, off_val: str, on_val: str, delta: str = "") -> str:
        return f"  {label:<22} {off_val:>8}   {on_val:>8}   {delta:>8}"

    print(f"\n{'='*w*2}")
    print(f"  {'ABLATION RESULTS':^{w*2-2}}")
    print(f"{'='*w*2}")
    print(row("Metric", "RAG OFF", "RAG ON", "Δ (ON−OFF)"))
    print(f"  {'-'*72}")

    acc_off = off["accuracy"]
    acc_on = on["accuracy"]
    f1_off = off["avg_f1"]
    f1_on = on["avg_f1"]
    n_off = off["n_samples"]
    n_on = on["n_samples"]
    err_off = off["n_errors"]
    err_on = on["n_errors"]

    def fmt_delta(d: float, pct: bool = False) -> str:
        s = f"{d:+.1%}" if pct else f"{d:+.4f}"
        return s

    print(row("Exact Match Acc.", f"{acc_off:.1%}", f"{acc_on:.1%}", fmt_delta(acc_on - acc_off, pct=True)))
    print(row("Avg Token F1", f"{f1_off:.4f}", f"{f1_on:.4f}", fmt_delta(f1_on - f1_off)))
    print(row("Samples evaluated", str(n_off), str(n_on)))
    print(row("Errors", str(err_off), str(err_on)))
    print(f"{'='*w*2}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate multimodal RAG agent on VQAv2"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of VQAv2 validation samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run both RAG OFF and RAG ON and print comparison table",
    )
    parser.add_argument(
        "--use_rag", action="store_true",
        help="Enable RAG context retrieval (ignored when --ablation is set)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.ablation:
        off_results = evaluate(n_samples=args.n_samples, use_rag=False)
        on_results = evaluate(n_samples=args.n_samples, use_rag=True)
        print_comparison(off_results, on_results)
    else:
        evaluate(n_samples=args.n_samples, use_rag=args.use_rag)


if __name__ == "__main__":
    main()
