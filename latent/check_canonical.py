"""Pilot 3, before any training: do the fixed wordings say enough?

The text arm of pilot 1 was trained on each question's own wording. Here it reads the
canonical wording instead (evaluate.py --mode text --canonical, tag check-canonical-text),
with no further training. This breaks both results down by type of question.
Single-prerequisite questions keep their own wording in this check: their canonical
form needs a pointer, which text mode does not have.
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATES = ("The eligibility rule is met for", "At least one prerequisite holds for")


def group(c):
    if c["kind"] in ("pa", "pb"):
        return "single prerequisite (own wording in both)"
    if c["kind"] == "or":
        return "at least one prerequisite"
    if not c["proposition"].removeprefix("Negation of: ").startswith(TEMPLATES):
        return "whole rule, free-form state"
    return "whole rule, templated state" + (', prompt asks "is it FALSE that"' if "Is it FALSE" in c["prompt"] else "")


def main():
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/noul-development.jsonl"))}
    out = {}
    for name, tag in (("own wording, trained (pilot 1 text arm)", "pilot-text"),
                      ("canonical wording, same adapter, not trained on it", "check-canonical-text")):
        right, total = Counter(), Counter()
        for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl")):
            for key in ("all questions", group(cases[r["id"]])):
                total[key] += 1
                right[key] += r["pick"] == r["gold"]
        out[name] = {k: f"{right[k]}/{total[k]}" for k in sorted(total)}
    (HERE / "results/check-canonical-text-breakdown.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
