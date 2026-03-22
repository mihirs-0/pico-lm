"""Main experiment runner. Calls LLM APIs for each task and saves raw outputs."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

import prompts
from prompts import (
    format_task1_prompt,
    format_task2_prompt,
    format_task3_turn1,
    format_task3_turn2,
    format_task3_oracle,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42

# Lazy-loaded HF model/tokenizer (shared across calls)
_hf_model = None
_hf_tokenizer = None


def _get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).parent.parent
        ).decode().strip()
    except Exception:
        return None


def _get_package_versions():
    versions = {"python": sys.version}
    for pkg in ["openai", "transformers", "torch"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _hash_file(path):
    """SHA-256 of a file for dataset fingerprinting."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_prompt_templates():
    """Snapshot all prompt template strings from prompts.py."""
    templates = {}
    for name in dir(prompts):
        obj = getattr(prompts, name)
        if isinstance(obj, str) and name.isupper():
            templates[name] = obj
    return templates


def build_run_metadata(args, data_path, num_examples):
    """Build a metadata dict capturing everything needed to reproduce this run."""
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": args.task,
        "model": args.model,
        "split": args.split,
        "few_shot": getattr(args, "few_shot", False),
        "limit": args.limit,
        "num_examples": num_examples,
        "seed": SEED,
        "temperature": 0.0,
        "max_tokens": 512,
        "dataset_file": str(data_path),
        "dataset_sha256": _hash_file(data_path),
        "git_hash": _get_git_hash(),
        "platform": platform.platform(),
        "package_versions": _get_package_versions(),
        "prompt_templates": _get_prompt_templates(),
    }


def call_openai(messages, model="gpt-4o", temperature=0.0, max_tokens=512):
    from openai import OpenAI
    client = OpenAI()
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** attempt  # 1, 2, 4, 8s
                time.sleep(wait)
                continue
            raise


def load_hf_model(model_name):
    """Load a HuggingFace model onto the best available device (CUDA > MPS > CPU)."""
    global _hf_model, _hf_tokenizer

    if _hf_model is not None:
        return _hf_model, _hf_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Suppress the auto_conversion background thread that spams HF API calls
    # (checks for safetensors conversion PRs — causes 429s on shared IPs)
    try:
        import transformers.safetensors_conversion
        transformers.safetensors_conversion.auto_conversion = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass

    # Fix torch version check for torch >= 2.10 (string comparison bug:
    # "2.10" < "2.6" lexicographically). Force the gate open.
    try:
        import transformers.modeling_utils as _mu
        for attr in dir(_mu):
            if "torch" in attr and "2_6" in attr:
                setattr(_mu, attr, True)
    except (ImportError, AttributeError):
        pass

    print(f"Loading {model_name} ...")
    # Models natively supported in transformers don't need trust_remote_code
    # (and their HF-hosted custom code may be stale / incompatible).
    trust = model_name not in {
        "microsoft/Phi-3-mini-4k-instruct",
    }
    _hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if _hf_tokenizer.pad_token is None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token

    if torch.cuda.is_available():
        device_map = "auto"  # spread across GPUs if needed (e.g., 70B)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"

    _hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=trust,
    )
    _hf_model.eval()
    print(f"Model loaded (device_map={device_map})")
    return _hf_model, _hf_tokenizer


def call_hf(messages, model_name, temperature=0.0, max_tokens=512):
    """Run inference with a local HuggingFace model."""
    import torch

    model, tokenizer = load_hf_model(model_name)

    # Use chat template if available, otherwise manual formatting
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        parts = []
        for m in messages:
            role = m["role"]
            if role == "system":
                parts.append(f"System: {m['content']}\n")
            elif role == "user":
                parts.append(f"User: {m['content']}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {m['content']}\n")
        parts.append("Assistant:")
        input_text = "\n".join(parts)

    # With device_map="auto", model may span devices; send inputs to first parameter's device
    first_device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(first_device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    response_tokens = output[0][input_len:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


def call_model(messages, model, **kwargs):
    """Dispatch to the right backend based on model name."""
    if model.startswith("gpt-"):
        return call_openai(messages, model=model, **kwargs)
    else:
        return call_hf(messages, model_name=model, **kwargs)


def load_data(split="test"):
    path = DATA_DIR / f"ebm_nlp_{split}_masked.json"
    with open(path, "r") as f:
        return json.load(f)


def run_task1(data, model, few_shot=False, limit=None):
    results = []
    subset = data[:limit] if limit else data

    for ex in tqdm(subset, desc="Task 1"):
        messages = format_task1_prompt(ex["model_input_text"], few_shot=few_shot)
        t0 = time.time()
        try:
            pred = call_model(messages, model)
            error = None
        except Exception as e:
            pred = f"ERROR: {e}"
            error = str(e)
        elapsed = time.time() - t0

        results.append({
            "pmid": ex["pmid"],
            "is_masked": ex["is_masked"],
            "true_slot": ex["missing_slot"],
            "predicted": pred,
            "latency_s": round(elapsed, 3),
            "error": error,
        })

    return results


def run_task2(data, model, limit=None):
    # Only run on masked examples
    masked = [ex for ex in data if ex["is_masked"]]
    subset = masked[:limit] if limit else masked
    results = []

    for ex in tqdm(subset, desc="Task 2"):
        messages = format_task2_prompt(ex["model_input_text"])
        t0 = time.time()
        try:
            question = call_model(messages, model)
            error = None
        except Exception as e:
            question = f"ERROR: {e}"
            error = str(e)
        elapsed = time.time() - t0

        results.append({
            "pmid": ex["pmid"],
            "missing_slot": ex["missing_slot"],
            "question": question,
            "model_input_text": ex["model_input_text"],
            "latency_s": round(elapsed, 3),
            "error": error,
        })

    return results


def run_task3(data, model, limit=None):
    masked = [ex for ex in data if ex["is_masked"]]
    subset = masked[:limit] if limit else masked
    results = []

    for ex in tqdm(subset, desc="Task 3"):
        timings = {}

        # Turn 1: extract from masked abstract (model may ask question or guess)
        t1_messages = format_task3_turn1(ex["model_input_text"])
        t0 = time.time()
        try:
            t1_response = call_model(t1_messages, model, max_tokens=1024)
            t1_error = None
        except Exception as e:
            t1_response = f"ERROR: {e}"
            t1_error = str(e)
        timings["turn1"] = round(time.time() - t0, 3)

        # Turn 2: provide withheld info, re-extract
        withheld = " | ".join(ex["withheld_spans"])
        t2_messages = format_task3_turn2(t1_messages, t1_response, withheld)
        t0 = time.time()
        try:
            t2_response = call_model(t2_messages, model, max_tokens=1024)
            t2_error = None
        except Exception as e:
            t2_response = f"ERROR: {e}"
            t2_error = str(e)
        timings["turn2"] = round(time.time() - t0, 3)

        # Oracle: extract from full original abstract
        oracle_messages = format_task3_oracle(ex["original_text"])
        t0 = time.time()
        try:
            oracle_response = call_model(oracle_messages, model, max_tokens=1024)
            oracle_error = None
        except Exception as e:
            oracle_response = f"ERROR: {e}"
            oracle_error = str(e)
        timings["oracle"] = round(time.time() - t0, 3)

        results.append({
            "pmid": ex["pmid"],
            "missing_slot": ex["missing_slot"],
            "turn1_response": t1_response,
            "turn2_response": t2_response,
            "oracle_response": oracle_response,
            "gold_spans": ex["all_spans"],
            "latency_s": timings,
            "errors": {"turn1": t1_error, "turn2": t2_error, "oracle": oracle_error},
        })

    return results


def load_checkpoint(out_file):
    """Load existing results for resume support."""
    if out_file.exists():
        with open(out_file, "r") as f:
            saved = json.load(f)
        if isinstance(saved, dict) and "results" in saved:
            return saved["results"]
        elif isinstance(saved, list):
            return saved
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--few-shot", action="store_true", help="Use few-shot examples (Task 1)")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to run")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if output exists")
    args = parser.parse_args()

    data = load_data(args.split)
    data_path = DATA_DIR / f"ebm_nlp_{args.split}_masked.json"
    print(f"Loaded {len(data)} examples from {args.split} split")

    fs_tag = "_fewshot" if args.few_shot else ""
    out_file = RESULTS_DIR / f"task{args.task}_{args.model.replace('/', '_')}{fs_tag}.json"

    # Resume support: skip already-completed PMIDs (exclude errored results)
    if args.resume:
        existing = load_checkpoint(out_file)
        # Keep only successful results; drop errored ones so they get retried
        def _has_error(r):
            if r.get("error"):
                return True
            if isinstance(r.get("errors"), dict) and any(r["errors"].values()):
                return True
            return False
        successful = [r for r in existing if not _has_error(r)]
        errored = [r for r in existing if _has_error(r)]
        done_pmids = {r["pmid"] for r in successful}
        original_len = len(data)
        data = [ex for ex in data if ex["pmid"] not in done_pmids]
        print(f"Resume: {len(successful)} succeeded, {len(errored)} errored (will retry), "
              f"{len(data)} to run (of {original_len})")
        existing = successful
    else:
        existing = []

    num_to_run = min(len(data), args.limit) if args.limit else len(data)
    metadata = build_run_metadata(args, data_path, num_to_run)

    run_start = time.time()
    if args.task == 1:
        results = run_task1(data, args.model, few_shot=args.few_shot, limit=args.limit)
    elif args.task == 2:
        results = run_task2(data, args.model, limit=args.limit)
    elif args.task == 3:
        results = run_task3(data, args.model, limit=args.limit)
    run_elapsed = time.time() - run_start

    all_results = existing + results
    metadata["total_results"] = len(all_results)
    metadata["new_results"] = len(results)
    metadata["total_runtime_s"] = round(run_elapsed, 2)
    metadata["num_errors"] = sum(
        1 for r in results
        if r.get("error") or (isinstance(r.get("errors"), dict) and any(r["errors"].values()))
    )

    output = {
        "metadata": metadata,
        "results": all_results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nRun complete:")
    print(f"  Results: {len(all_results)} total ({len(results)} new)")
    print(f"  Errors:  {metadata['num_errors']}")
    print(f"  Runtime: {run_elapsed:.1f}s")
    print(f"  Saved:   {out_file}")


if __name__ == "__main__":
    main()
