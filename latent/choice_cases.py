"""Choice variants for the request pilot, derived by rule from the imported cases.

Each case gives up to two kinds of question:
  select    the case's own question: its acceptable alternatives, or its status
  ruledout  which alternative the rule rules out: every alternative labelled not
            feasible. Defined for assessed cases that have one, and for cases where
            no alternative is feasible.
Each question is asked with the alternatives in several orders. Development gets the
listed order, the reverse and, for three alternatives, one rotation. Training gets
two orders drawn with a fixed seed.

Every alternative also gets its place in the state: the characters where it is named.
Found by rule, or from choice_names.json where the id differs from the state's wording.
Tokenizer only (for the pointer audit), no model.
"""

import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from choice_common import MARK, render, span_ids, state_text, tokenise  # noqa: E402

SEED = 20261111
HAND = json.loads((HERE / "choice_names.json").read_text())


PUNCTUATION = ",.;:?!'\"()"


def candidates(case, identity):
    """Character ranges in the state where the alternative is named, best first, and how they were found."""
    state = state_text(case)
    hand = HAND["names"].get(case["family"], {}).get(identity)
    if hand is not None and hand in state:
        return [m.span() for m in re.finditer(re.escape(hand), state)], "by hand"
    found = []
    if len(identity) == 1:
        found += [(m.end() - 1, m.end()) for m in re.finditer(rf"(?i)\boption {re.escape(identity)}\b", state)]
    for pattern, flags in ((rf"(?<!\w){re.escape(identity)}(?!\w)", 0),
                           (rf"(?<!\w){re.escape(re.sub('[-_]', ' ', identity))}(?!\w)", re.I)):
        found += [m.span() for m in re.finditer(pattern, state, flags)]
    if not found:
        raise ValueError(f"{case['id']} {case['family']}: no place in the state for {identity!r}")
    return found, "by rule"


def locate(tok, case, identity, before, head):
    """The first place whose tokens decode to exactly the name; failing that, the first place.

    A pointer is a range of tokens. Where a token straddles the name's edge it brings a
    neighbouring punctuation mark, or the n of an escaped line break, with it.
    """
    state = state_text(case)
    spans, found = candidates(case, identity)
    for span in spans:
        if tok.decode(span_ids(before, head, state, span)).strip() == state[span[0]:span[1]]:
            return span, found, True
    got, name = tok.decode(span_ids(before, head, state, spans[0])).strip(), state[spans[0][0]:spans[0][1]]
    assert got.strip(PUNCTUATION) == name or got.strip(PUNCTUATION) == "n" + name, (case["id"], got, name)
    return spans[0], found, False


def variants(tok, case, split, rng, how):
    before = render(tok, {**case, "kind": "select"}, True).split(MARK)[0]
    head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
    alts = []
    for a in case["alternatives"]:
        span, found, clean = locate(tok, case, a["identity"], before, head)
        how[found] += 1
        how["exact tokens" if clean else "with a neighbouring mark"] += 1
        alts.append({**a, "span": list(span)})
    ids = [a["identity"] for a in alts]
    kinds = [("select", case["status"], case["value"]["acceptable_ids"] if case["status"] == "Assessed" else [])]
    infeasible = [i for i in ids if case["feasibility"][i] is False]
    if (case["status"] == "Assessed" and infeasible) or case["status"] == "NoFeasibleChoice":
        kinds.append(("ruledout", "Assessed", infeasible))
    if split == "train":
        orders = lambda: rng.sample(list(itertools.permutations(range(len(ids)))), 2)
    else:
        fixed = [tuple(range(len(ids))), tuple(reversed(range(len(ids))))] + ([(1, 2, 0)] if len(ids) == 3 else [])
        orders = lambda: fixed
    out = []
    for kind, status, acceptable in kinds:
        for n, order in enumerate(orders()):
            out.append({
                "id": f"{case['id']}-{kind}-{n}", "base_id": case["id"], "split": split, "family": case["family"],
                "kind": kind, "order": n, "listed_order": list(order) == list(range(len(ids))),
                "prompt": case["prompt"], "facts": case["facts"], "rules": case["rules"],
                "alternatives": [alts[i] for i in order], "status": status, "acceptable": acceptable,
                "source_status": case["status"], "letter_names": all(len(i) == 1 for i in ids),
            })
    return out


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    rng = random.Random(SEED)
    summary = {}
    for split in ("train", "development"):
        cases = [c for c in map(json.loads, open(ROOT / f"data/qwenonetwo/{split}.jsonl")) if c["operation"] == "Choice"]
        how, out, excluded = Counter(), [], 0
        for case in cases:
            if case["family"] in HAND["excluded"]:
                excluded += 1
                continue
            out += variants(tok, case, split, rng, how)
        lengths = []
        for v in out:  # the pointers of the finished prompt are the ones chosen above
            state = state_text(v)
            _, _, _, spans = tokenise(tok, v, "latent", 16)
            for a, span in zip(v["alternatives"], spans):
                name, got = state[a["span"][0]:a["span"][1]], tok.decode(span).strip().strip(PUNCTUATION)
                assert got in (name, "n" + name), (v["id"], got, name)
                lengths.append(len(span))
        (ROOT / f"data/latent/choice-{split}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in out))
        states = {v["base_id"] for v in out}
        summary[split] = {
            "cases": len(states), "cases_excluded": excluded, "questions": len(out),
            "kinds": dict(Counter(v["kind"] for v in out)), "status": dict(Counter(v["status"] for v in out)),
            "cases_with_letter_names": len({v["base_id"] for v in out if v["letter_names"]}),
            "alternatives_located": dict(how), "pointer_tokens_min_max": [min(lengths), max(lengths)],
        }
    (HERE / "results/choice-cases.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
