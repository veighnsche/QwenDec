"""Requests as fixed word sequences.

A request kind is one fixed list of token ids: its wording tokenised on its own. It is
inserted as it is where the question goes, never tokenised together with what surrounds
it, so the same kind is the same ids in every prompt.
"""

_ids = {}


def fixed_ids(tok, text):
    if text not in _ids:
        _ids[text] = tok(text, add_special_tokens=False)["input_ids"]
    return _ids[text]


def insert(tok, ids, q_start, text, k=0):
    """Put the fixed sequence for `text` at q_start, in place of k placeholder positions."""
    return ids[:q_start] + fixed_ids(tok, text) + ids[q_start + k:]
