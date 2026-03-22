# RunPod GPU Experiment Guide

## Quick Start

```bash
# On RunPod (after cloning the repo):
git clone git@github.com:mihirs-0/pico-lm.git && cd pico-lm

# 1. Install deps + prepare dataset
bash runpod/setup.sh

# 2. Run all experiments
bash runpod/run_all.sh            # all models including 70B
bash runpod/run_all.sh --skip-70b # skip 70B if on a smaller GPU

# 3. Evaluate
bash runpod/evaluate_all.sh

# 4. Copy results back to local machine
# From your LOCAL machine:
scp -r <runpod-ssh>:~/pico-lm/results/ ./results/
```

## GPU Recommendation

| GPU | VRAM | Cost (community) | Can run |
|-----|------|-------------------|---------|
| **A100 80GB** | 80 GB | ~$1.64/hr | All models including 70B |
| A40 | 48 GB | ~$0.76/hr | All 7B/8B models, NOT 70B |
| A6000 | 48 GB | ~$0.76/hr | All 7B/8B models, NOT 70B |

**Recommendation: A100 80GB** — handles everything in one session (~1 hr total).

## Models & Time Estimates (A100 80GB)

| Model | Params | VRAM | ~Time (2 runs) | Purpose |
|-------|--------|------|-----------------|---------|
| BioMistral-7B | 7B | 14 GB | ~5 min | Domain-specialized (proposal model) |
| Llama-3.1-8B-Instruct | 8B | 16 GB | ~5 min | Strong open instruct (proposal model) |
| Mistral-7B-Instruct-v0.3 | 7B | 14 GB | ~5 min | BioMistral's parent → isolates domain effect |
| Llama-3.1-70B-Instruct | 70B | 140 GB | ~40 min | Scale test (needs A100 80GB) |

Total estimated time: **~1 hour** (including model downloads).

## What these runs add

- **BioMistral vs Mistral-Instruct**: Same architecture, domain vs general pretraining.
  Directly tests whether biomedical pretraining helps prompted detection.
- **Llama-3.1-8B**: Originally proposed model. Strong 8B instruct baseline.
- **Llama-3.1-70B** (optional): Tests if scale alone closes the gap to GPT-4o for open models.

## Notes

- Each run produces a JSON file in `results/` with 189 entries.
- HuggingFace models are downloaded to `~/.cache/huggingface/` (not committed).
- Llama models require accepting the license on huggingface.co and being logged in:
  `huggingface-cli login`
- BioMistral is a **base model** (no chat template) — uses the manual prompt fallback.
