"""Prompt format shared by every arm of the latent-question pilot.

The prompt is the adapter's own format with the status options, exactly as in the
earlier measurements. Only the question slot differs:
  text    the question as words
  none    an empty question
  latent  K positions whose input embeddings are replaced by question vectors
The state is read with the adapter off; the adapter turns on at the end of the state.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
from measure_decision4b import LETTERS, STATUS_OPTIONS, SYSTEM  # noqa: E402

MARK = "⁣QUESTION⁣"  # replaced by the question text or by K vector positions
STATE_END = '", "question": "'
OPTIONS = [("true", "The proposition is true.", True), ("false", "The proposition is false.", False)] + [
    (key, text, status) for status, (key, text) in STATUS_OPTIONS.items() if status != "NoFeasibleChoice"
]
VALUES = [v for _, _, v in OPTIONS]


def question_text(case):
    return f"Is this proposition true? {case['proposition']}. Criterion: {case['criterion']}."


def gold_index(case):
    return VALUES.index(bool(case["p_true"]) if case["status"] == "Assessed" else case["status"])


def render(tok, case):
    """The chat prompt with MARK where the question goes."""
    task = {
        "state": "\n".join([case["prompt"], "Facts:", *case["facts"], "Rules:", *case["rules"]]),
        "question": MARK,
        "options": [{"label": LETTERS[i], "key": k, "description": d} for i, (k, d, _) in enumerate(OPTIONS)],
    }
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def tokenise(tok, case, mode, k=0, pad=0):
    """Token ids, the first adapter-on position, and where the question vectors go (latent only)."""
    before, after = render(tok, case).split(MARK)
    if mode == "latent":
        head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
        ids = head["input_ids"] + [pad] * k + tok(after, add_special_tokens=False)["input_ids"]
        offsets, q_start = head["offset_mapping"], len(head["input_ids"])
    else:
        text = before + (question_text(case) if mode == "text" else "") + after
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets, q_start = enc["input_ids"], enc["offset_mapping"], -1
    boundary = before.index(STATE_END) + 1
    gate_start = sum(end <= boundary for _, end in offsets)
    return ids, gate_start, q_start
