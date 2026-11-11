"""Pilot 4: results by type of question, on the renamed panel and on development as it is.

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
from common import pointed  # noqa: E402

PANEL, ORIGINAL = "noul-development-panel4", "noul-development"
ROWS = {"text arm, new panel": ("panel4-text", PANEL), "pilot 4, new panel": ("pilot4-canonical", PANEL),
        "pilot 4, development's own names": ("pilot4-canonical-original-names", ORIGINAL),
        "pilot 3, new panel": ("pilot3-canonical-panel4", PANEL),
        "pilot 3, development's own names": ("pilot3-canonical", ORIGINAL)}


def main():
    report = {}
    for name, (tag, file) in ROWS.items():
        cases = {c["id"]: c for c in map(json.loads, open(ROOT / f"data/latent/{file}.jsonl"))}
        right, total = Counter(), Counter()
        for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl")):
            c = cases[r["id"]]
            keys = ["all questions", group(c).replace(" (own wording in both)", ""), f"status {c['status']}"]
            if pointed(c) is not None:
                keys.append("single prerequisite, " + ("negated" if c["negated"] else "plain"))
            for key in keys:
                total[key] += 1
                right[key] += r["pick"] == r["gold"]
        summary = json.load(open(HERE / f"results/{tag}.json"))
        report[name] = {"assessed twin pairs": summary["both_twins_right"]["assessed twin pairs"].split(" ")[0],
                        "prerequisites differ": summary["single_assessed_prerequisites_differ"].split(" ")[0],
                        **{k: f"{right[k]}/{total[k]}" for k in sorted(total)}}
    (HERE / "results/analysis-pilot4.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
