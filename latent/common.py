"""Prompt format shared by every arm of the latent-question pilot.

The prompt is the adapter's own format with the status options, exactly as in the
earlier measurements. Only the question slot differs:
  text    the question as words
  none    an empty question
  latent  K positions whose input embeddings are replaced by question vectors
The state is read with the adapter off; the adapter turns on at the end of the state.

Pilot 2 adds a pointer for single-prerequisite questions. The question then names
no prerequisite, and the request points at the name where the rule sentence
mentions it. The pointer reaches the fork in one of two forms:
  vector  one extra position after the K vectors (filled by the caller with the
          mean input embedding of the pointed tokens)
  span    the pointed tokens themselves, repeated after the K vectors
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import RULE  # noqa: E402
from measure_decision4b import LETTERS, STATUS_OPTIONS, SYSTEM  # noqa: E402

MARK = "⁣QUESTION⁣"  # replaced by the question text or by K vector positions
STATE_END = '", "question": "'
OPTIONS = [("true", "The proposition is true.", True), ("false", "The proposition is false.", False)] + [
    (key, text, status) for status, (key, text) in STATUS_OPTIONS.items() if status != "NoFeasibleChoice"
]
VALUES = [v for _, _, v in OPTIONS]


def pointed(case, other=False):
    """The prerequisite a single-prerequisite question is about; with other=True, the one it is not about."""
    if not (case["parsed"] and case["kind"] in ("pa", "pb")):
        return None
    _, a, b = RULE.match(case["rules"][0]).groups()
    return a if (case["kind"] == "pa") != other else b


# Pilot 3: one fixed wording per kind of question, with no name from the state in it.
CANONICAL = {
    "and": ("What the supplied rule allows or defines holds in this case", "the supplied rule applied to the supplied facts"),
    "or": ("At least one prerequisite of the supplied rule holds", "either prerequisite alone is enough"),
    "pa": ("The indicated prerequisite holds", "the indicated prerequisite alone"),
}
CANONICAL["pb"] = CANONICAL["pa"]


def question_text(case, pointer=False, canonical=False):
    """The question as words. With pointer=True a single-prerequisite question names no prerequisite.

    With canonical=True the question is one of the fixed wordings above. A single-prerequisite
    question needs its pointer to be canonical; without one it keeps its own wording.
    """
    if canonical and (case["kind"] in ("and", "or") or (pointer and pointed(case) is not None)):
        proposition, criterion = CANONICAL[case["kind"]]
        return f"Is this proposition true? {'Negation of: ' if case['negated'] else ''}{proposition}. Criterion: {criterion}."
    if pointer and pointed(case) is not None:
        target = RULE.match(case["rules"][0]).group(1)
        proposition = f"{'Negation of: ' if case['negated'] else ''}The indicated prerequisite holds for {target}"
        return f"Is this proposition true? {proposition}. Criterion: the indicated prerequisite alone."
    return f"Is this proposition true? {case['proposition']}. Criterion: {case['criterion']}."


def pointer_tokens(before, head, case, other=False):
    """Token ids of the pointed name, taken from the tokenised state where the rule sentence mentions it."""
    name = pointed(case, other)
    if name is None:
        return []
    _, a, b = RULE.match(case["rules"][0]).groups()
    escape = lambda s: json.dumps(s, ensure_ascii=False)[1:-1]  # the state sits inside a JSON string
    marker = "both prerequisites hold: "
    first = (case["kind"] == "pa") != other  # pointing at the first prerequisite named in the rule
    start = before.rindex(marker) + len(marker) + (0 if first else len(escape(a)) + 2)
    end = start + len(escape(name))
    assert before[start:end] == escape(name), "the rule sentence does not mention the name where expected"
    ids = [i for i, (s, e) in zip(head["input_ids"], head["offset_mapping"]) if e > start and s < end]
    return ids


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


def tokenise(tok, case, mode, k=0, pad=0, pointer=None, other=False, canonical=False):
    """Token ids, the first adapter-on position, and where the question vectors go (latent only).

    With pointer="vector" or "span" a fourth value is returned: the pointed token ids
    (empty when the question has no pointer). other=True points at the other prerequisite.
    """
    before, after = render(tok, case).split(MARK)
    if mode == "latent":
        head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
        span = pointer_tokens(before, head, case, other) if pointer else []
        slot = [pad] * (k + 1) if pointer == "vector" else [pad] * k + span
        ids = head["input_ids"] + slot + tok(after, add_special_tokens=False)["input_ids"]
        offsets, q_start = head["offset_mapping"], len(head["input_ids"])
    else:
        text = before + (question_text(case, canonical=canonical) if mode == "text" else "") + after
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets, q_start = enc["input_ids"], enc["offset_mapping"], -1
    boundary = before.index(STATE_END) + 1
    gate_start = sum(end <= boundary for _, end in offsets)
    return (ids, gate_start, q_start, span) if pointer else (ids, gate_start, q_start)
