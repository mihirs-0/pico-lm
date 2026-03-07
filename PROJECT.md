# Clarification-Seeking in Biomedical LLMs Under Controlled PICO Ambiguity

> Eranki Vasistha, Mihir Sahasrabudhe

---

## 1. Motivation

Large language models respond to underspecified clinical questions with confident guesses rather than asking for the missing information. In biomedical settings this is dangerous — fabricating a comparator or outcome can mislead clinical decision-making. This project reframes PICO extraction from RCT abstracts into a testbed for **clarification-seeking under ambiguity**: can a model notice something is missing, ask about it, and then do better?

## 2. Research Questions

| RQ | Question | Task |
|----|----------|------|
| RQ1 | Given an RCT abstract with exactly one PICO slot redacted, can an LLM identify *which* slot is missing? | Task 1: Missing-slot detection |
| RQ2 | When a slot is missing, can the model ask a targeted, non-leading clarifying question? | Task 2: Clarification question generation |
| RQ3 | Does providing the withheld span (simulated user answer) improve downstream PICO extraction? | Task 3: Post-clarification extraction |

## 3. Dataset

### EBM-NLP (Nye et al., 2018)

- **Source:** [github.com/bepnye/EBM-NLP](https://github.com/bepnye/EBM-NLP)
- **Contents:** ~5,000 RCT abstracts with token-level span annotations for **Participants (P)**, **Interventions (I)**, and **Outcomes (O)**.
- **Splits:** 4,792 train / 189 test (gold-annotated test set)
- **Format:** One `.tokens` file per abstract + per-slot `.ann` files with binary (0/1) labels per token.

### Why not Comparator (C)?

The original proposal included a Comparator slot. EBM-NLP does not annotate Comparators directly — they are a subtype ("Control") within the Interventions folder. In practice these labels are:

- Inconsistently applied across annotators
- Not cleanly separable from Intervention spans
- A source of noise that would confound our controlled experiments

**Decision:** We scope to **P/I/O only**. This gives us 3 well-annotated slot types, each with clean gold spans. The experimental design (detection, question generation, extraction improvement) works identically with 3 slots. Comparators can be revisited as a follow-up if the PICO-Corpus dataset (which has explicit C labels) is integrated.

## 4. Data Pipeline

### 4.1 Parsing (`src/data_prep.py`)

```
EBM-NLP raw files
  ├── documents/{pmid}.tokens        (whitespace-separated token files)
  └── annotations/aggregated/starting_spans/
        ├── participants/{split}/{pmid}.ann
        ├── interventions/{split}/{pmid}.ann
        └── outcomes/{split}/{pmid}.ann
```

For each abstract, we:

1. Read the token file to get the token sequence.
2. Read each slot's annotation file to get binary labels.
3. Build contiguous spans by grouping consecutive `1`-labeled tokens.
4. Store both span text and **token indices** (start, end) for precise redaction later.

Output: `data/ebm_nlp_ground_truth.json`

```json
{
  "pmid": "10036953",
  "split": "train",
  "text": "99 patients with H. pylori infection were ...",
  "spans": {
    "P": ["99 patients with H. pylori infection were"],
    "I": ["lansoprazole", "ranitidine", "clarithromycin ( CAM )", ...],
    "O": ["efficacy and safety", "cure rate of H. pylori infection"]
  }
}
```

### 4.2 Masking (Controlled Slot Deletion)

For each abstract to be masked:

1. **Pick a slot** — randomly select one of {P, I, O} that has at least one annotated span.
2. **Redact at the token level** — replace every token belonging to that slot's spans with `[REDACTED]`, collapsing consecutive redactions into a single marker.
3. **Preserve everything else** — all other tokens (including other PICO slots) remain intact.

**Example:**

Original:
> Women with early-stage breast cancer (n=450) were randomized to receive adjuvant chemotherapy with doxorubicin and cyclophosphamide or tamoxifen alone. The primary endpoint was disease-free survival at 5 years.

Outcome-deleted:
> Women with early-stage breast cancer (n=450) were randomized to receive adjuvant chemotherapy with doxorubicin and cyclophosphamide or tamoxifen alone. The primary endpoint was [REDACTED] at 5 years.

### Why span-level, not sentence-level?

The teammate's original approach removed entire sentences containing the target slot. Problems:

| Issue | Impact |
|-------|--------|
| **Collateral deletion** | A sentence with both P and I spans — masking I also removes P context |
| **Unrealistic input** | Abstracts with missing sentences look structurally broken; models may detect the gap from structure alone rather than from missing content |
| **Information loss** | Non-PICO context in the sentence (e.g., study design, timeframes) is lost |

Span-level redaction with `[REDACTED]` markers solves all three: it isolates the manipulation to exactly the target slot, preserves sentence structure, and gives the model a clear signal that *something* was removed (rather than silently omitting it).

### 4.3 Dataset Composition

| Split | Masked (75%) | Complete (25%) | Total |
|-------|-------------|---------------|-------|
| Train | ~3,594 | ~1,198 | 4,792 |
| Test  | ~141 | ~48 | 189 |

The 25% complete (unmasked) abstracts serve as **"none missing" controls** for Task 1 — the model must also recognize when all slots are present.

Output files: `data/ebm_nlp_train_masked.json`, `data/ebm_nlp_test_masked.json`

```json
{
  "pmid": "10036953",
  "split": "test",
  "is_masked": true,
  "missing_slot": "O",
  "model_input_text": "... [REDACTED] ...",
  "original_text": "...",
  "withheld_spans": ["efficacy and safety", "cure rate of H. pylori infection"],
  "all_spans": {"P": [...], "I": [...], "O": [...]}
}
```

## 5. Tasks and Prompting

### Task 1: Missing-Slot Detection

**Goal:** Given a (possibly redacted) abstract, classify which of {P, I, O, none} is missing.

**Prompt design:**
- System message explains PICO and the task.
- User message provides the abstract and asks for exactly one label.
- **Zero-shot variant:** No examples.
- **Few-shot variant:** 4 hand-crafted examples (one per label: P, I, O, none).

**Expected output:** A single token — `P`, `I`, `O`, or `none`.

### Task 2: Clarification Question Generation

**Goal:** Given a redacted abstract, ask one specific clarifying question about the missing information.

**Prompt design:**
- System message instructs the model to ask a single targeted question, avoid guessing, avoid multiple questions.
- User message provides the abstract.

**Expected output:** A natural-language question like *"What outcome was measured in this trial?"*

**Only run on masked examples** (asking "what's missing?" on a complete abstract is meaningless).

### Task 3: Post-Clarification PICO Extraction

**Goal:** Test whether answering the clarification question improves PICO extraction.

**Protocol (3 conditions per example):**

| Condition | Input | Purpose |
|-----------|-------|---------|
| **Turn 1** | Masked abstract | Baseline extraction (model may guess or ask a question) |
| **Turn 2** | Turn 1 conversation + withheld spans provided | Re-extraction after clarification |
| **Oracle** | Full original abstract (no masking) | Upper bound |

**Expected output:** JSON `{"P": [...], "I": [...], "O": [...]}` with extracted span lists.

## 6. Evaluation

### Task 1 Metrics

| Metric | Definition |
|--------|-----------|
| **Accuracy** | % of examples where predicted label matches true label |
| **Macro-F1** | Average F1 across all 4 classes {P, I, O, none} |
| **Per-slot breakdown** | Precision/Recall/F1 for each slot separately |

### Task 2 Metrics: LLM-as-Judge

An LLM judge (GPT-4o) scores each generated question on three criteria (1-5 scale):

| Criterion | What it measures |
|-----------|-----------------|
| **Slot targeting** | Does the question ask about the correct missing element? |
| **Specificity** | Is it specific enough that a domain expert could answer concisely? |
| **No assumptions** | Does it avoid introducing information not in the abstract? |

We spot-check a random sample against human judgment for calibration.

### Task 3 Metrics

| Metric | Definition |
|--------|-----------|
| **Token-level PICO F1** | Overlap between predicted and gold span tokens, macro-averaged across P/I/O |
| **Performance delta** | F1(Turn 2) - F1(Turn 1) — the improvement from clarification |
| **Oracle gap** | F1(Oracle) - F1(Turn 2) — how much room remains |

## 7. Models

| Model | Type | Access |
|-------|------|--------|
| GPT-4o | Closed API | OpenAI API (`OPENAI_API_KEY`) |
| Llama-3.1-8B-Instruct | Open instruct | HuggingFace / vLLM |
| BioMistral-7B | Biomedical fine-tune | HuggingFace / vLLM |

We compare general-purpose vs. domain-specialized models to test whether biomedical fine-tuning improves clarification-seeking behavior (not just extraction accuracy).

## 8. Baselines and Ablations

| Ablation | What it tests |
|----------|--------------|
| Zero-shot vs. few-shot (Task 1) | Does in-context learning help slot detection? |
| With vs. without clarification instruction | Does explicitly asking the model to clarify change behavior? |
| Slot type effects (P vs I vs O) | Which missing slot is hardest to detect / ask about? |
| Sentence-level vs. span-level masking | Does masking granularity affect detection difficulty? |

## 9. Project Structure

```
llm-proj/
  CLAUDE.md                  # Quick reference for AI assistant
  PROJECT.md                 # This file — full project documentation
  requirements.txt
  .gitignore
  sahasrabudhe-mihir-proposal.pdf   # Original proposal
  src/
    __init__.py
    data_prep.py             # Download EBM-NLP, parse, mask
    prompts.py               # All prompt templates
    run_experiment.py        # CLI experiment runner
    evaluate.py              # All evaluation metrics
  data/                      # Generated datasets (gitignored)
    ebm_nlp_raw/             # Raw EBM-NLP files
    ebm_nlp_ground_truth.json
    ebm_nlp_train_masked.json
    ebm_nlp_test_masked.json
  results/                   # Experiment outputs (gitignored)
    task1_gpt-4o.json
    task2_gpt-4o.json
    task3_gpt-4o.json
```

## 10. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Download and preprocess data
python src/data_prep.py

# 3. Run experiments (set OPENAI_API_KEY for GPT-4o)
export OPENAI_API_KEY="sk-..."

# Task 1: missing-slot detection
python src/run_experiment.py --task 1 --model gpt-4o                  # zero-shot
python src/run_experiment.py --task 1 --model gpt-4o --few-shot       # few-shot

# Task 2: clarification question generation
python src/run_experiment.py --task 2 --model gpt-4o

# Task 3: post-clarification extraction
python src/run_experiment.py --task 3 --model gpt-4o

# Use --limit N for quick test runs
python src/run_experiment.py --task 1 --model gpt-4o --limit 10

# 4. Evaluate
python src/evaluate.py --task 1 --results results/task1_gpt-4o.json
python src/evaluate.py --task 2 --results results/task2_gpt-4o.json
python src/evaluate.py --task 3 --results results/task3_gpt-4o.json
```

## 11. Deviations from Original Proposal

| Proposal | Implementation | Rationale |
|----------|---------------|-----------|
| P/I/C/O (4 slots) | P/I/O (3 slots) | EBM-NLP Comparator labels are noisy; 3 clean slots > 4 noisy ones |
| Sentence-level masking | Span-level `[REDACTED]` | Avoids collateral deletion of other PICO elements |
| EBM-NLP + PICO-Corpus | EBM-NLP only (for now) | Start with one well-understood dataset; PICO-Corpus can be added |
| BioMistral-7B | May substitute if unavailable | Fallback: Meditron or biomedical Llama variant |

## 12. Open Questions

- **Should we include abstracts where the redacted slot has many scattered spans?** (e.g., Intervention often has 10+ spans — redacting all of them leaves a very fragmented abstract. We may want to filter for examples where the slot has <= N spans.)
- **Few-shot for Tasks 2 and 3?** Currently only Task 1 has few-shot. Worth testing if few-shot examples help question quality or extraction.
- **Cost management:** Running GPT-4o on all ~140 test examples across 3 tasks (with Task 3 requiring 3 API calls each) is ~600 calls. Budget-check before full runs.
