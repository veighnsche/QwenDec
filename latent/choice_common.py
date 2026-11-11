"""Prompt format shared by both arms of the Choice request pilot.

The prompt is the adapter's own format: state, question, options (each alternative,
then the status options). The arms differ in two places:
  text    the question as words; each alternative with its name and properties
  latent  K positions for the question vectors; each alternative is a pointer, the
          tokens of its name taken from the state and replayed as the option's key
The state is read with the adapter off; the adapter turns on at the end of the state.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
from measure_decision4b import LETTERS, STATUS_OPTIONS, SYSTEM  # noqa: E402

MARK = "⁣QUESTION⁣"
ALT = "⁣ALT{}⁣"
STATE_END = '", "question": "'
WORDING = {
    "select": "Which of the indicated alternatives should be selected under the supplied rule?",
    "ruledout": "Which of the indicated alternatives does the supplied rule rule out?",
}
CRITERION = " Criterion: the supplied rule applied to the supplied facts."


def question_text(case):
    return WORDING[case["kind"]] + CRITERION


def state_text(case):
    return "\n".join([case["prompt"], "Facts:", *case["facts"], "Rules:", *case["rules"]])


def values(case):
    """What each option stands for: the alternatives in the order asked, then the statuses."""
    return [a["identity"] for a in case["alternatives"]] + list(STATUS_OPTIONS)


def target(case):
    """Weight per option, and whether the loss credits the weighted set as a whole."""
    v = values(case)
    if case["status"] != "Assessed":
        return [float(x == case["status"]) for x in v], False
    return [float(x in case["acceptable"]) for x in v], True


def is_right(case, pick):
    picked = values(case)[pick]
    if case["status"] != "Assessed":
        return picked == case["status"]
    return picked in case["acceptable"]


def render(tok, case, latent):
    n = len(case["alternatives"])
    if latent:
        options = [{"label": LETTERS[i], "key": ALT.format(i)} for i in range(n)]
    else:
        options = [{"label": LETTERS[i], "key": a["identity"], "description": a["properties"]}
                   for i, a in enumerate(case["alternatives"])]
    options += [{"label": LETTERS[n + j], "key": key, "description": text}
                for j, (key, text) in enumerate(STATUS_OPTIONS.values())]
    task = {"state": state_text(case), "question": MARK if latent else question_text(case), "options": options}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def span_ids(before, head, state, span):
    """Token ids covering a character range of the state, taken from the tokenised state."""
    escape = lambda s: json.dumps(s, ensure_ascii=False)[1:-1]  # the state sits inside a JSON string
    s, e = span
    start = before.index(escape(state)) + len(escape(state[:s]))
    end = start + len(escape(state[s:e]))
    assert before[start:end] == escape(state[s:e])
    return [t for t, (a, b) in zip(head["input_ids"], head["offset_mapping"]) if b > start and a < end]


def tokenise(tok, case, mode, k=0, pad=0):
    """Token ids, the first adapter-on position, where the question vectors go (-1 for text), and the pointer spans."""
    text = render(tok, case, mode == "latent")
    if mode == "text":
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets, q_start, spans = enc["input_ids"], enc["offset_mapping"], -1, []
    else:
        before, rest = text.split(MARK)
        head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
        offsets, q_start, spans = head["offset_mapping"], len(head["input_ids"]), []
        ids = head["input_ids"] + [pad] * k
        for i in range(len(case["alternatives"])):
            piece, rest = rest.split(ALT.format(i))
            spans.append(span_ids(before, head, state_text(case), case["alternatives"][i]["span"]))
            ids += tok(piece, add_special_tokens=False)["input_ids"] + spans[-1]
        ids += tok(rest, add_special_tokens=False)["input_ids"]
    boundary = text.index(STATE_END) + 1
    gate_start = sum(end <= boundary for _, end in offsets)
    assert 0 < gate_start <= (q_start if q_start >= 0 else len(ids))
    return ids, gate_start, q_start, spans
