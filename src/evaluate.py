"""Evaluation metrics for all three tasks."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, classification_report

from prompts import format_judge_prompt

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---- Task 1: Missing-slot detection ----

def normalize_prediction(pred):
    """Extract a clean slot label from model output.

    Handles verbose base-model outputs (e.g., BioMistral) by checking the
    start of the response first, then falling back to keyword search with
    guards against false positives like "P-value" or "Population (P)".
    """
    raw = pred.strip()
    upper = raw.upper()

    # --- Exact match (clean outputs) ---
    for label in ["NONE", "P", "I", "O"]:
        if upper == label:
            return label.lower() if label == "NONE" else label

    # --- Check first token / first line ---
    first_line = raw.split("\n")[0].strip()
    first_token = first_line.split()[0].rstrip(".:,;") if first_line.split() else ""
    ft_upper = first_token.upper()
    if ft_upper in ("P", "I", "O"):
        return ft_upper
    if ft_upper == "NONE":
        return "none"

    # --- "none" synonyms anywhere in output ---
    lower = raw.lower()
    if any(phrase in lower for phrase in [
        "none", "no element", "all elements are present", "no pico",
        "nothing is missing", "no missing",
    ]):
        return "none"

    # --- Guarded single-letter search (skip false positives) ---
    # Only match standalone P/I/O that aren't part of common false-positive patterns
    for label in ["P", "I", "O"]:
        # Require the letter to appear as a standalone word, but reject matches
        # adjacent to "-" (P-value), or inside parenthetical explanations
        pattern = rf"(?<![A-Za-z\-]){label}(?![A-Za-z\-])"
        matches = list(re.finditer(pattern, upper))
        if matches:
            # Extra guard: reject if the match is inside a word-like context
            m = matches[0]
            context = upper[max(0, m.start() - 10):m.end() + 10]
            if re.search(rf"{label}[\-]", context):
                continue  # e.g., "P-VALUE"
            return label

    return raw  # return raw if we can't parse — tracked as parse failure


def evaluate_task1(results):
    true_labels = []
    pred_labels = []
    parse_failures = []

    valid_labels = {"P", "I", "O", "none"}

    for r in results:
        true = r["true_slot"] if r["true_slot"] else "none"
        pred = normalize_prediction(r["predicted"])
        true_labels.append(true)
        pred_labels.append(pred)
        if pred not in valid_labels:
            parse_failures.append({
                "pmid": r.get("pmid"),
                "raw": r["predicted"][:200],
                "normalized": pred[:100],
            })

    labels = ["P", "I", "O", "none"]
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)

    print(f"Accuracy: {acc:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")
    n = len(results)
    pf = len(parse_failures)
    print(f"Parse failures: {pf}/{n} ({pf/n*100:.1f}%)")
    if parse_failures:
        print("  Sample failures:")
        for pf_entry in parse_failures[:5]:
            print(f"    pmid={pf_entry['pmid']}: {pf_entry['raw'][:80]!r}")
    print()
    print(classification_report(true_labels, pred_labels, labels=labels, zero_division=0))

    return {"accuracy": acc, "macro_f1": macro_f1, "parse_failures": len(parse_failures)}


# ---- Task 2: Clarification question quality (LLM-as-judge) ----

def evaluate_task2(results, judge_model="gpt-4o"):
    """Run LLM-as-judge on Task 2 outputs. Requires API access."""
    from prompts import format_judge_prompt

    try:
        from run_experiment import call_model
    except ImportError:
        print("Cannot import call_model. Run from src/ directory.")
        return

    scores = []
    for r in results:
        if r["question"].startswith("ERROR"):
            continue

        messages = format_judge_prompt(
            r["model_input_text"], r["missing_slot"], r["question"]
        )
        try:
            judge_output = call_model(messages, judge_model, temperature=0.0)
            score = json.loads(judge_output)
            score["pmid"] = r["pmid"]
            scores.append(score)
        except Exception as e:
            print(f"Judge error for {r['pmid']}: {e}")

    if not scores:
        print("No scores collected.")
        return

    avg = {
        "slot_targeting": sum(s["slot_targeting"] for s in scores) / len(scores),
        "specificity": sum(s["specificity"] for s in scores) / len(scores),
        "no_assumptions": sum(s["no_assumptions"] for s in scores) / len(scores),
    }
    print(f"LLM-as-Judge scores (n={len(scores)}):")
    for k, v in avg.items():
        print(f"  {k}: {v:.2f}")

    return {"average_scores": avg, "individual_scores": scores}


# ---- Task 3: Downstream PICO extraction improvement ----

def extract_json_from_response(text):
    """Try to parse JSON from a model response that may contain extra text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON block in markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Find any JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def token_f1(pred_spans, gold_spans):
    """Compute token-level F1 between predicted and gold span lists."""
    pred_tokens = set()
    for span in pred_spans:
        for token in span.lower().split():
            pred_tokens.add(token)

    gold_tokens = set()
    for span in gold_spans:
        for token in span.lower().split():
            gold_tokens.add(token)

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall = tp / len(gold_tokens) if gold_tokens else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_task3(results):
    """Compare token-level PICO F1 across turn1, turn2, and oracle."""
    slots = ["P", "I", "O"]
    metrics = {"turn1": [], "turn2": [], "oracle": []}

    for r in results:
        gold = r["gold_spans"]

        for turn_key in ["turn1_response", "turn2_response", "oracle_response"]:
            parsed = extract_json_from_response(r[turn_key])
            if parsed is None:
                continue

            f1s = []
            for slot in slots:
                pred = parsed.get(slot, [])
                if isinstance(pred, str):
                    pred = [pred]
                f1s.append(token_f1(pred, gold.get(slot, [])))

            stage = turn_key.replace("_response", "")
            metrics[stage].append(sum(f1s) / len(f1s))

    print("Token-level PICO F1 (macro avg across P/I/O):")
    for stage, scores in metrics.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {stage}: {avg:.3f} (n={len(scores)})")
        else:
            print(f"  {stage}: no valid responses")

    if metrics["turn1"] and metrics["turn2"]:
        delta = (sum(metrics["turn2"]) / len(metrics["turn2"])) - (
            sum(metrics["turn1"]) / len(metrics["turn1"])
        )
        print(f"\n  Performance delta (turn2 - turn1): {delta:+.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--judge-model", type=str, default="gpt-4o", help="Model for Task 2 judging")
    args = parser.parse_args()

    with open(args.results, "r") as f:
        raw = json.load(f)
    # Support both envelope format {"metadata": ..., "results": [...]} and bare list
    if isinstance(raw, dict) and "results" in raw:
        results = raw["results"]
        meta = raw.get("metadata", {})
        print(f"Loaded {len(results)} results from {args.results}")
        if meta:
            print(f"  Run: model={meta.get('model')}, task={meta.get('task')}, "
                  f"timestamp={meta.get('timestamp_utc')}")
    else:
        results = raw
        print(f"Loaded {len(results)} results from {args.results} (legacy format)")

    if args.task == 1:
        evaluate_task1(results)
    elif args.task == 2:
        evaluate_task2(results, judge_model=args.judge_model)
    elif args.task == 3:
        evaluate_task3(results)


if __name__ == "__main__":
    main()
