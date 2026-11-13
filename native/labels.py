"""One record per state: the operation it calls for and the items it can point at.

Taken from what the request pilots already used. The operation is the case's own. The
items are the pointers the adapter was trained on: a Noul prerequisite where the rule
sentence names it, a Choice alternative at its first exact mention. Score has none.
A tenth of the training states is held out by family.
Tokenizer only, no model.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "latent"))
from cases import RULE  # noqa: E402
from reading import OPERATIONS, native_pass, reads_as, state_text, tokens_of  # noqa: E402

SEED = 20261113
MARKER = "both prerequisites hold: "
FILES = {"train": ("noul-train", "choice-train", "score-train"),
         "development": ("noul-development-panel4", "choice-development", "score-development")}


def load(name):
    return [json.loads(x) for x in open(ROOT / f"data/latent/{name}.jsonl")]


def noul_items(case):
    """The two prerequisites, at their place in the rule sentence."""
    if not case["parsed"]:
        return []
    state = state_text(case)
    _, a, b = RULE.match(case["rules"][0]).groups()
    start = state.rindex(MARKER) + len(MARKER)
    spans = [(a, [start, start + len(a)]), (b, [start + len(a) + 2, start + len(a) + 2 + len(b)])]
    for name, (s, e) in spans:
        assert state[s:e] == name, (case["id"], name)
    return [{"name": name, "span": span} for name, span in spans]


def states(split):
    noul, choice, score = (load(name) for name in FILES[split])
    out, seen = [], set()
    for op, cases in (("Noul", noul), ("Choice", choice), ("Score", score)):
        for c in cases:
            if c["base_id"] in seen:
                continue
            seen.add(c["base_id"])
            if op == "Noul":
                items = noul_items(c)
            elif op == "Choice":
                state = state_text(c)  # the name as the state words it, which is not always the alternative's id
                items = [{"name": state[a["span"][0]:a["span"][1]], "identity": a["identity"], "span": a["span"]} for a in c["alternatives"]]
            else:
                items = []
            out.append({"id": c["base_id"], "split": split, "operation": op, "family": c["family"],
                        "prompt": c["prompt"], "facts": c["facts"], "rules": c["rules"],
                        "items": sorted(items, key=lambda i: i["span"][0])})
    return out


def hold_out(records):
    """A tenth of each operation's training states, whole families at a time."""
    rng = random.Random(SEED)
    for op in OPERATIONS:
        mine = [r for r in records if r["operation"] == op]
        families = sorted({r["family"] for r in mine})
        rng.shuffle(families)
        held, count = set(), 0
        for family in families:
            if count >= len(mine) / 10:
                break
            held.add(family)
            count += sum(r["family"] == family for r in mine)
        for r in mine:
            r["held_out"] = r["family"] in held


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    (ROOT / "data/native").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    summary = {}
    for split in ("train", "development"):
        records = states(split)
        if split == "train":
            hold_out(records)
        lengths, item_lengths, marks = [], [], 0
        for r in records:
            state = state_text(r)
            ids, offsets, first, before = native_pass(tok, state)
            r["tokens"], r["first_state_token"] = len(ids), first
            lengths.append(len(ids))
            for item in r["items"]:
                a, b = tokens_of(offsets, before, state, item["span"])
                assert first <= a <= b < len(ids), "an item lies outside the state"
                text = tok.decode(ids[a:b + 1])
                assert reads_as(text, item["name"]), (r["id"], text, item["name"])
                marks += text.strip() != item["name"]
                item["token_range"] = [a, b]
                item_lengths.append(b - a + 1)
            ranges = [tuple(i["token_range"]) for i in r["items"]]
            assert ranges == sorted(ranges) and len(set(ranges)) == len(ranges), "items overlap or are out of order"
        (ROOT / f"data/native/states-{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        by_op = defaultdict(list)
        for r in records:
            by_op[r["operation"]].append(r)
        summary[split] = {
            "states": len(records),
            "by_operation": {op: {"states": len(v), "with_items": sum(bool(r["items"]) for r in v),
                                  "items": dict(Counter(len(r["items"]) for r in v)),
                                  **({"held_out": sum(r["held_out"] for r in v)} if split == "train" else {})}
                             for op, v in by_op.items()},
            "tokens_read_mean": round(sum(lengths) / len(lengths), 1), "tokens_read_max": max(lengths),
            "tokens_read_total": sum(lengths),
            "item_tokens_min_max": [min(item_lengths), max(item_lengths)],
            "items": len(item_lengths), "items_carrying_a_neighbouring_mark": marks,
        }
    (HERE / "results/labels.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
