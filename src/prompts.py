"""Prompt templates for all three tasks."""

# --- Task 1: Missing-slot detection ---

TASK1_SYSTEM = (
    "You are a biomedical research assistant. You will be given an RCT abstract "
    "where zero or one PICO element may have been redacted (replaced with [REDACTED]). "
    "PICO elements are:\n"
    "  P = Population/Participants\n"
    "  I = Intervention\n"
    "  O = Outcome\n\n"
    "Your job: identify which element is missing, or say none is missing."
)

TASK1_USER = (
    "Abstract:\n\"{abstract}\"\n\n"
    "Which PICO element is missing or redacted in this abstract? "
    "Reply with exactly one of: P, I, O, none"
)

TASK1_FEWSHOT_EXAMPLES = [
    {
        "abstract": (
            "A total of [REDACTED] were randomized to receive either aspirin 100mg daily "
            "or placebo for 12 months. The primary endpoint was reduction in cardiovascular events."
        ),
        "answer": "P",
        "reasoning": "The population/participants description is redacted.",
    },
    {
        "abstract": (
            "120 patients with type 2 diabetes were randomized to receive [REDACTED] or "
            "metformin 500mg twice daily. HbA1c levels were measured at 6 months."
        ),
        "answer": "I",
        "reasoning": "One of the interventions is redacted.",
    },
    {
        "abstract": (
            "Women aged 50-70 with early-stage breast cancer (n=300) were randomized to "
            "receive tamoxifen 20mg daily or anastrozole 1mg daily. [REDACTED] was assessed at 5 years."
        ),
        "answer": "O",
        "reasoning": "The outcome measure is redacted.",
    },
    {
        "abstract": (
            "Patients with chronic lower back pain (n=200) were randomly assigned to "
            "physical therapy twice weekly or usual care. Pain scores on the VAS scale "
            "were measured at 3 and 6 months."
        ),
        "answer": "none",
        "reasoning": "All PICO elements are present.",
    },
]


def format_task1_prompt(abstract, few_shot=False):
    messages = [{"role": "system", "content": TASK1_SYSTEM}]

    if few_shot:
        for ex in TASK1_FEWSHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": TASK1_USER.format(abstract=ex["abstract"]),
            })
            messages.append({"role": "assistant", "content": ex["answer"]})

    messages.append({
        "role": "user",
        "content": TASK1_USER.format(abstract=abstract),
    })
    return messages


# --- Task 2: Clarification question generation ---

TASK2_SYSTEM = (
    "You are a biomedical research assistant reviewing an RCT abstract. "
    "Some information has been redacted ([REDACTED]). "
    "Your job is to ask a single, specific clarifying question to recover the missing information. "
    "Do NOT guess the missing content. Do NOT ask multiple questions. "
    "Your question should be targeted enough that a domain expert could answer it concisely."
)

TASK2_USER = (
    "Abstract:\n\"{abstract}\"\n\n"
    "Ask one clarifying question about what is missing from this abstract."
)


def format_task2_prompt(abstract):
    return [
        {"role": "system", "content": TASK2_SYSTEM},
        {"role": "user", "content": TASK2_USER.format(abstract=abstract)},
    ]


# --- Task 3: Post-clarification PICO extraction ---

TASK3_SYSTEM = (
    "You are a biomedical NLP system. Extract all PICO elements from the given RCT abstract.\n"
    "Return a JSON object with keys \"P\", \"I\", \"O\", where each value is a list of strings "
    "representing the extracted spans. If an element is missing or unclear, use an empty list."
)

TASK3_USER_TURN1 = (
    "Extract the PICO elements from this abstract. If any element is missing or "
    "underspecified, ask a clarifying question instead of guessing.\n\n"
    "Abstract:\n\"{abstract}\""
)

TASK3_USER_TURN2 = (
    "Thank you for asking. Here is the missing information: \"{withheld_text}\"\n\n"
    "Now please extract all PICO elements from the complete abstract. "
    "Return JSON with keys \"P\", \"I\", \"O\"."
)

TASK3_EXTRACT_ONLY = (
    "Extract the PICO elements from this abstract.\n\n"
    "Abstract:\n\"{abstract}\"\n\n"
    "Return JSON with keys \"P\", \"I\", \"O\"."
)


def format_task3_turn1(abstract):
    return [
        {"role": "system", "content": TASK3_SYSTEM},
        {"role": "user", "content": TASK3_USER_TURN1.format(abstract=abstract)},
    ]


def format_task3_turn2(messages, model_response, withheld_text):
    """Append the clarification answer and re-extraction request."""
    messages = messages + [
        {"role": "assistant", "content": model_response},
        {"role": "user", "content": TASK3_USER_TURN2.format(withheld_text=withheld_text)},
    ]
    return messages


def format_task3_oracle(abstract):
    """Single-turn extraction on the full (unmasked) abstract — oracle upper bound."""
    return [
        {"role": "system", "content": TASK3_SYSTEM},
        {"role": "user", "content": TASK3_EXTRACT_ONLY.format(abstract=abstract)},
    ]


# --- Task 2 evaluation: LLM-as-judge ---

JUDGE_SYSTEM = (
    "You are an evaluation judge. You will be given:\n"
    "1. An RCT abstract with one PICO slot redacted\n"
    "2. Which slot was redacted (P, I, or O)\n"
    "3. A clarifying question generated by a model\n\n"
    "Score the question on three criteria (1-5 each):\n"
    "  (a) Slot targeting: Does the question ask about the correct missing slot?\n"
    "  (b) Specificity: Is it specific enough to be answered concisely?\n"
    "  (c) No assumptions: Does it avoid introducing information not in the abstract?\n\n"
    "Return JSON: {\"slot_targeting\": int, \"specificity\": int, \"no_assumptions\": int, \"reasoning\": str}"
)

JUDGE_USER = (
    "Abstract:\n\"{abstract}\"\n\n"
    "Redacted slot: {slot}\n"
    "Model's clarifying question: \"{question}\"\n\n"
    "Score this question."
)


def format_judge_prompt(abstract, slot, question):
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            abstract=abstract, slot=slot, question=question
        )},
    ]
