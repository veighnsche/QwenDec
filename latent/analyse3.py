"""Pilot 3: results by type of question, and what the single-prerequisite errors are.

Reads per-case outputs only. No model, no training.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from check_canonical import group  # noqa: E402
from common import gold_index, pointed  # noqa: E402

ARMS = {"text, own wording (pilot 1)": "pilot-text", "text, canonical wording, not trained on it": "check-canonical-text",
        "replayed span (pilot 2)": "pilot2-pointer-span", "canonical requests (pilot 3)": "pilot3-canonical"}


def rows(tag):
    return {r["id"]: r for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl"))}


def main():
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/noul-development.jsonl"))}
    report = {"by_type": {}, "single_prerequisite": {}}
    for name, tag in ARMS.items():
        right, total = Counter(), Counter()
        for i, r in rows(tag).items():
            c = cases[i]
            keys = [group(c).replace(" (own wording in both)", "")]
            if pointed(c) is not None:
                keys.append("single prerequisite, " + ("negated" if c["negated"] else "plain"))
            for key in keys:
                total[key] += 1
                right[key] += r["pick"] == r["gold"]
        report["by_type"][name] = {k: f"{right[k]}/{total[k]}" for k in sorted(total)}

    # Pilot 3's wrong single-prerequisite answers: which other answer do they equal?
    out = rows("pilot3-canonical")
    kinds = Counter()
    for i, r in out.items():
        c = cases[i]
        if pointed(c) is None or r["pick"] == r["gold"]:
            continue
        sibling = cases[f"{c['base_id']}-{'pb' if c['kind'] == 'pa' else 'pa'}{'-neg' if c['negated'] else ''}"]
        kinds["wrong"] += 1
        kinds["question is negated"] += c["negated"]
        kinds["prerequisite holds, answered True"] += c["status"] == "Assessed" and c["p_true"] == 0 and r["pick"] == 0
        kinds["equals the answer without the negation"] += r["pick"] == gold_index(cases[c["twin"]])
        kinds["equals the other prerequisite's answer"] += r["pick"] == gold_index(sibling)
        kinds["first prerequisite" if c["kind"] == "pa" else "second prerequisite"] += 1
    report["single_prerequisite"]["pilot 3 errors"] = dict(kinds)
    (HERE / "results/analysis-pilot3.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
