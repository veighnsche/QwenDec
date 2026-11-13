"""What the native model reads, and how a place in the state maps to its tokens and back.

The native pass is the opening of the decision prompt up to the end of the state, read
with the adapter off. It is the same for every operation: the tokens depend on the
state alone. A pointer is a range of those tokens.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
from measure_decision4b import SYSTEM  # noqa: E402

MARK = "⁣QUESTION⁣"
STATE_END = '", "question": "'
OPERATIONS = ["Noul", "Choice", "Score"]
SLOTS = 3
PUNCTUATION = ",.;:?!'\"()"


def state_text(case):
    return "\n".join([case["prompt"], "Facts:", *case["facts"], "Rules:", *case["rules"]])


def escape(s):
    return json.dumps(s, ensure_ascii=False)[1:-1]  # the state sits inside a JSON string


def opening(tok, state):
    """The prompt up to where the question goes. Depends on the state only."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"state": state, "question": MARK}, ensure_ascii=False)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text.split(MARK)[0]


def native_pass(tok, state):
    """Token ids the native model reads, their character offsets, and which of them lie in the state.

    Returns (ids, offsets, first, before): ids[:len(ids)] are read with the adapter off;
    tokens from index `first` on belong to the state itself.
    """
    before = opening(tok, state)
    head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
    boundary = before.index(STATE_END) + 1
    n = sum(end <= boundary for _, end in head["offset_mapping"])
    start = before.index(escape(state))
    first = sum(end <= start for _, end in head["offset_mapping"][:n])
    return head["input_ids"][:n], head["offset_mapping"][:n], first, before


def tokens_of(offsets, before, state, span):
    """First and last token index covering a character range of the state."""
    s, e = span
    start = before.index(escape(state)) + len(escape(state[:s]))
    end = start + len(escape(state[s:e]))
    assert before[start:end] == escape(state[s:e])
    covered = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    return covered[0], covered[-1]


def characters_of(offsets, before, state, first, last):
    """The character range of the state that a token range covers.

    The state is written as a JSON string, where a line break takes two characters. A token
    that starts or ends inside such a pair is cut back to the whole characters it covers.
    """
    base = before.index(escape(state))
    start, end = offsets[first][0] - base, offsets[last][1] - base
    edges = [0]  # where each state character ends once the state is written as a JSON string
    for i in range(len(state)):
        edges.append(len(escape(state[: i + 1])))
    s = next(i for i, e in enumerate(edges) if e >= start)
    e = max(i for i, x in enumerate(edges) if x <= end)
    return [s, e] if s < e else None


def reads_as(text, name):
    """A produced range counts as an item if it reads exactly as the item's name.

    A token can carry a neighbouring punctuation mark, or the n of an escaped line break,
    as four of the handed development pointers do.
    """
    text = text.strip().strip(PUNCTUATION)
    return text in (name, "n" + name)
