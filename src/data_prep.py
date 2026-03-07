"""
Download EBM-NLP dataset, parse tokens + annotations, and create masked datasets.

Fixes from teammate's original preprocessing:
- Span-level deletion (replaces spans with [REDACTED]) instead of sentence removal
- Cleaner token-file matching (strips .AGGREGATED suffix)
"""

import json
import os
import random
import tarfile
import urllib.request
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "ebm_nlp_raw"
BASE_DIR = RAW_DIR / "ebm_nlp_2_00"

SLOTS = {"P": "participants", "I": "interventions", "O": "outcomes"}
SPLIT_SUBPATHS = {"train": "train", "test": os.path.join("test", "gold")}


def download_ebm_nlp():
    """Download and extract EBM-NLP 2.0 dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if BASE_DIR.exists():
        print(f"EBM-NLP already extracted at {BASE_DIR}, skipping download.")
        return

    url = "https://raw.githubusercontent.com/bepnye/EBM-NLP/master/ebm_nlp_2_00.tar.gz"
    tar_path = DATA_DIR / "ebm_nlp_2_00.tar.gz"

    print(f"Downloading EBM-NLP from {url}...")
    urllib.request.urlretrieve(url, str(tar_path))

    print("Extracting...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(str(RAW_DIR), filter="data")

    tar_path.unlink()
    print(f"Done. Raw data at {BASE_DIR}")


def parse_ebm_nlp():
    """Parse EBM-NLP tokens and annotations into structured JSON.

    Returns list of dicts with keys: pmid, split, text, spans {P: [...], I: [...], O: [...]}.
    """
    documents_dir = BASE_DIR / "documents"
    ann_base = BASE_DIR / "annotations" / "aggregated" / "starting_spans"

    parsed = []

    for split_name, sub_path in SPLIT_SUBPATHS.items():
        # Use participants dir to discover PMIDs for this split
        p_dir = ann_base / "participants" / sub_path
        if not p_dir.exists():
            print(f"Warning: {p_dir} not found, skipping {split_name}")
            continue

        ann_files = sorted(f for f in os.listdir(p_dir) if f.endswith(".ann"))
        print(f"Parsing {split_name}: {len(ann_files)} annotation files")

        for ann_file in ann_files:
            pmid = ann_file.split(".")[0]
            token_file = documents_dir / f"{pmid}.tokens"

            if not token_file.exists():
                continue

            with open(token_file, "r", encoding="utf-8") as f:
                tokens = f.read().split()

            record = {
                "pmid": pmid,
                "split": split_name,
                "text": " ".join(tokens),
                "tokens": tokens,
                "spans": {"P": [], "I": [], "O": []},
            }

            for slot, folder in SLOTS.items():
                slot_ann = ann_base / folder / sub_path / ann_file
                if not slot_ann.exists():
                    continue

                with open(slot_ann, "r", encoding="utf-8") as f:
                    labels = f.read().split()

                # Build contiguous spans from BIO-style 0/1 labels
                current_span_tokens = []
                current_span_indices = []
                for i, (token, label) in enumerate(zip(tokens, labels)):
                    if label.strip() == "1":
                        current_span_tokens.append(token)
                        current_span_indices.append(i)
                    else:
                        if current_span_tokens:
                            record["spans"][slot].append({
                                "text": " ".join(current_span_tokens),
                                "token_start": current_span_indices[0],
                                "token_end": current_span_indices[-1],
                            })
                            current_span_tokens = []
                            current_span_indices = []
                if current_span_tokens:
                    record["spans"][slot].append({
                        "text": " ".join(current_span_tokens),
                        "token_start": current_span_indices[0],
                        "token_end": current_span_indices[-1],
                    })

            parsed.append(record)

    print(f"Total parsed: {len(parsed)} abstracts")
    return parsed


def create_masked_dataset(parsed_data, mask_ratio=0.75):
    """Create masked datasets with span-level deletion.

    For masked examples, all spans of one randomly chosen slot are replaced
    with [REDACTED] in the token sequence. This isolates the deletion to
    the target slot without removing surrounding context.

    Returns dict with 'train' and 'test' lists.
    """
    splits = {}

    for split_name in ["train", "test"]:
        data = [d for d in parsed_data if d["split"] == split_name]
        # Only abstracts with at least one non-empty slot can be masked
        valid = [d for d in data if any(d["spans"][s] for s in SLOTS)]
        random.shuffle(valid)

        target_masked = int(len(valid) * mask_ratio)
        masked_count = 0
        examples = []

        for record in valid:
            if masked_count < target_masked:
                # Pick a random non-empty slot to mask
                available = [s for s in SLOTS if record["spans"][s]]
                target_slot = random.choice(available)
                spans = record["spans"][target_slot]

                # Build set of token indices to redact
                redact_indices = set()
                for span in spans:
                    for i in range(span["token_start"], span["token_end"] + 1):
                        redact_indices.add(i)

                # Replace redacted tokens, collapsing consecutive redactions
                masked_tokens = []
                in_redaction = False
                for i, token in enumerate(record["tokens"]):
                    if i in redact_indices:
                        if not in_redaction:
                            masked_tokens.append("[REDACTED]")
                            in_redaction = True
                    else:
                        masked_tokens.append(token)
                        in_redaction = False

                withheld_texts = [sp["text"] for sp in spans]

                examples.append({
                    "pmid": record["pmid"],
                    "split": split_name,
                    "is_masked": True,
                    "missing_slot": target_slot,
                    "model_input_text": " ".join(masked_tokens),
                    "original_text": record["text"],
                    "withheld_spans": withheld_texts,
                    "all_spans": {
                        s: [sp["text"] for sp in record["spans"][s]] for s in SLOTS
                    },
                })
                masked_count += 1
            else:
                # Unmasked control example
                examples.append({
                    "pmid": record["pmid"],
                    "split": split_name,
                    "is_masked": False,
                    "missing_slot": None,
                    "model_input_text": record["text"],
                    "original_text": record["text"],
                    "withheld_spans": None,
                    "all_spans": {
                        s: [sp["text"] for sp in record["spans"][s]] for s in SLOTS
                    },
                })

        random.shuffle(examples)
        splits[split_name] = examples

        n_masked = sum(1 for e in examples if e["is_masked"])
        print(f"{split_name}: {n_masked} masked + {len(examples) - n_masked} complete = {len(examples)} total")

    return splits


def main():
    download_ebm_nlp()
    parsed = parse_ebm_nlp()

    # Save ground truth (with token indices for debugging)
    gt_file = DATA_DIR / "ebm_nlp_ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        # Strip tokens list for smaller file size in ground truth
        gt_data = []
        for r in parsed:
            gt_record = {k: v for k, v in r.items() if k != "tokens"}
            gt_record["spans"] = {
                s: [sp["text"] for sp in r["spans"][s]] for s in SLOTS
            }
            gt_data.append(gt_record)
        json.dump(gt_data, f, indent=2)
    print(f"Saved ground truth to {gt_file}")

    # Create and save masked datasets
    splits = create_masked_dataset(parsed)
    for split_name, examples in splits.items():
        out_file = DATA_DIR / f"ebm_nlp_{split_name}_masked.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
