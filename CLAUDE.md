# Clarification-Seeking in Biomedical LLMs Under Controlled PICO Ambiguity

## Project Overview

Benchmark for testing whether LLMs detect missing information in biomedical RCT abstracts
and ask clarifying questions rather than guessing. Uses EBM-NLP dataset with programmatic
PICO slot deletion to create controlled ambiguity.

## Research Questions

1. **Missing-slot detection (Task 1):** Can the LLM identify which PICO slot is missing?
2. **Clarification question quality (Task 2):** Does it ask a targeted question about the missing slot?
3. **Downstream improvement (Task 3):** Does providing the withheld span improve PICO extraction?

## Key Design Decisions

- **Slots: P/I/O only** (no Comparator). EBM-NLP's "Control" subtype is noisy and inconsistently
  annotated. Scoping to 3 slots keeps evaluation clean.
- **Span-level deletion** (not sentence-level). Teammate's original approach removed entire sentences,
  which also strips other PICO elements. We replace target spans with `[REDACTED]` to isolate
  the manipulation.
- **Single dataset (EBM-NLP)** for the initial implementation. PICO-Corpus can be added later.

## Project Structure

```
llm-proj/
  CLAUDE.md
  requirements.txt
  src/
    data_prep.py        # Download EBM-NLP, parse, create masked datasets
    prompts.py          # Prompt templates for all 3 tasks
    run_experiment.py   # Main experiment runner (calls LLM APIs)
    evaluate.py         # Metrics: accuracy, F1, LLM-as-judge
  data/                 # Generated data files (gitignored)
  results/              # Experiment outputs (gitignored)
```

## Models to Evaluate

- GPT-4o (via OpenAI API)
- Llama-3.1-8B-Instruct (via local inference or API)
- BioMistral-7B (via HuggingFace / local)

## Evaluation Metrics

- **Task 1:** Accuracy, macro-F1 over {P, I, O, none}
- **Task 2:** LLM-as-judge rubric (slot targeting, specificity, no new assumptions)
- **Task 3:** Token-level PICO F1 before vs. after clarification, performance delta

## Commands

```bash
# Setup
pip install -r requirements.txt
python src/data_prep.py              # Download + preprocess EBM-NLP

# Run experiments
python src/run_experiment.py --task 1 --model gpt-4o
python src/run_experiment.py --task 2 --model gpt-4o
python src/run_experiment.py --task 3 --model gpt-4o

# Evaluate
python src/evaluate.py --task 1 --results results/task1_gpt4o.json
```

## Dev Notes

- API keys: set `OPENAI_API_KEY` env var for GPT-4o
- For local models: set `HF_MODEL_PATH` or use `--model-path` flag
- Seed: 42 everywhere for reproducibility
