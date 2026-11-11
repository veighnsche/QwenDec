"""Single-prerequisite questions under three sets of prerequisite names, for the trained pointer arms.

original  development's own names (unseen, clauses, differing in one word)
nouns     distinct unseen noun phrases
clauses   distinct unseen clauses ending in "holds"

Reads per-case outputs only. No model, no training.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import pointed  # noqa: E402

ARMS = {"pilot 2, pointer as one vector": ("pilot2-pointer-vector", "pilot2-vector"),
        "pilot 2, pointer as replayed span": ("pilot2-pointer-span", "pilot2-span"),
        "pilot 3, canonical requests": ("pilot3-canonical", "pilot3")}


def main():
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/noul-development.jsonl")) if pointed(c) is not None}

    def differ(c):  # both prerequisites assessed, with different answers
        sibling = cases[f"{c['base_id']}-{'pb' if c['kind'] == 'pa' else 'pa'}{'-neg' if c['negated'] else ''}"]
        return c["status"] == "Assessed" and sibling["status"] == "Assessed" and sibling["p_true"] != c["p_true"]

    report = {}
    for arm, (original, short) in ARMS.items():
        report[arm] = {}
        for names, tag in (("original", original), ("nouns", f"renamed-nouns-{short}"), ("clauses", f"renamed-clauses-{short}")):
            right, total = Counter(), Counter()
            for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl")):
                c = cases.get(r["id"])
                if c is None:
                    continue
                keys = ["all", "negated" if c["negated"] else "plain"] + (["prerequisites differ"] if differ(c) else [])
                for key in keys:
                    total[key] += 1
                    right[key] += r["pick"] == r["gold"]
            report[arm][names] = {k: f"{right[k]}/{total[k]}" for k in ("all", "plain", "negated", "prerequisites differ")}
    (HERE / "results/analysis-renamed.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
