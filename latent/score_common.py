"""Prompt format shared by both arms of the Score request pilot.

The prompt is the adapter's own format: state, question, options (the eleven levels,
then the status options). The arms differ in the question only:
  text    the fixed wording as words, followed by the rubric as words
  latent  K positions for the question vectors; no rubric, the state's rule holds it
The state is read with the adapter off; the adapter turns on at the end of the state.

There are two kinds of request. "score" asks for the score. "complement" asks for one
minus the score; it is derived by rule and exists so that two requests on one state
have different right answers.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
from measure_decision4b import LETTERS, LEVELS, STATUS_OPTIONS, SYSTEM, interval_distance  # noqa: E402

MARK = "⁣QUESTION⁣"
STATE_END = '", "question": "'
WORDING = {
    "score": "What score between 0 and 1 applies under the supplied rule?",
    "complement": "What is one minus the score that applies under the supplied rule?",
}
CRITERION = " Criterion: the supplied rule applied to the supplied facts."
REQUESTS = sorted(w + CRITERION for w in WORDING.values())  # the order of the request vectors in their file
STATUSES = [s for s in STATUS_OPTIONS if s != "NoFeasibleChoice"]
VALUES = LEVELS + STATUSES
TOLERANCE = 0.05 + 1e-9  # a level midway between two targets must not fall outside by rounding


def request_text(case):
    """What the fixed encoder reads: the fixed wording, nothing from the case."""
    return WORDING[case["kind"]] + CRITERION


def question_text(case):
    """What the text arm reads: the fixed wording and the case's rubric."""
    rubric = "; ".join(f"{k}: {v}" for k, v in case["rubric"].items())
    return f"{request_text(case)} Rubric: {rubric}."


def state_text(case):
    return "\n".join([case["prompt"], "Facts:", *case["facts"], "Rules:", *case["rules"]])


def right_levels(interval):
    return {x for x in LEVELS if interval_distance(x, interval) <= TOLERANCE}


def target(case):
    """Weight per option: the status, or the two levels around the target so the expected level equals it."""
    w = [0.0] * len(VALUES)
    if case["status"] != "Assessed":
        w[VALUES.index(case["status"])] = 1.0
        return w
    lo, hi = case["interval"]
    x = min(max((lo + hi) / 2, 0.0), 1.0) * (len(LEVELS) - 1)
    below = min(int(x), len(LEVELS) - 2)
    w[below], w[below + 1] = 1.0 - (x - below), x - below
    return w


def is_right(case, pick):
    picked = VALUES[pick]
    if case["status"] != "Assessed":
        return picked == case["status"]
    return not isinstance(picked, str) and picked in right_levels(case["interval"])


def render(tok, case, latent):
    options = [{"label": LETTERS[i], "key": f"{x:.1f}", "description": f"A value of {x:.1f}."} for i, x in enumerate(LEVELS)]
    options += [{"label": LETTERS[len(LEVELS) + j], "key": STATUS_OPTIONS[s][0], "description": STATUS_OPTIONS[s][1]}
                for j, s in enumerate(STATUSES)]
    task = {"state": state_text(case), "question": MARK if latent else question_text(case), "options": options}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def tokenise(tok, case, mode, k=0, pad=0):
    """Token ids, the first adapter-on position, and where the question vectors go (-1 for text)."""
    text = render(tok, case, mode == "latent")
    if mode == "text":
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets, q_start = enc["input_ids"], enc["offset_mapping"], -1
    else:
        before, after = text.split(MARK)
        head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
        offsets, q_start = head["offset_mapping"], len(head["input_ids"])
        ids = head["input_ids"] + [pad] * k + tok(after, add_special_tokens=False)["input_ids"]
    boundary = text.index(STATE_END) + 1
    gate_start = sum(end <= boundary for _, end in offsets)
    assert 0 < gate_start <= (q_start if q_start >= 0 else len(ids))
    return ids, gate_start, q_start
