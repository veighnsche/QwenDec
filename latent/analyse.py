"""Where the latent arms fail: first against second prerequisite.

Two measurements, no training and no model forward:

1. Assessed single-prerequisite questions, split by whether the state's two
   prerequisites have the same answer, differ, or the other one is not stated.
2. How alike the fixed encoder's vectors are for the first- and second-prerequisite
   question of one state, compared with the same kind of question across states.

The vector arithmetic runs on the RTX 3090.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(HERE))
from common import question_text  # noqa: E402
from model import require_gpu  # noqa: E402

NAMES = ["True", "False", "Insufficient", "Conflicting", "Unresolved"]
GROUPS = ("prerequisites agree", "prerequisites differ", "other prerequisite not stated")


def main():
    device = require_gpu()
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/noul-development.jsonl"))}
    plain = {}
    for c in cases.values():
        if c["kind"] in ("pa", "pb") and not c["negated"]:
            plain.setdefault(c["base_id"], {})[c["kind"]] = c
    plain = {k: v for k, v in plain.items() if len(v) == 2}

    def group(case):
        a, b = plain[case["base_id"]]["pa"], plain[case["base_id"]]["pb"]
        if a["status"] != "Assessed" or b["status"] != "Assessed":
            return GROUPS[2]
        return GROUPS[0] if a["p_true"] == b["p_true"] else GROUPS[1]

    report = {}
    for k in (4, 16):
        right, total, wrong = Counter(), Counter(), Counter()
        for row in map(json.loads, open(HERE / f"results/pilot-latent-k{k}.cases.jsonl")):
            case = cases[row["id"]]
            if case["kind"] not in ("pa", "pb") or case["status"] != "Assessed":
                continue
            g = group(case)
            total[g] += 1
            if row["pick"] == row["gold"]:
                right[g] += 1
            else:
                wrong[(g, NAMES[row["pick"]])] += 1

        file = torch.load(ROOT / f"data/latent/question-vectors-k{k}.pt")
        vectors, index = file["vectors"].to(device).float(), file["index"]
        vectors = vectors - vectors.mean(0, keepdim=True)  # drop what every question shares
        flat = torch.nn.functional.normalize(vectors.flatten(1), dim=1)
        first = flat[[index[question_text(v["pa"])] for v in plain.values()]]
        second = flat[[index[question_text(v["pb"])] for v in plain.values()]]
        same_state = (first * second).sum(1)
        across = (first @ first.T)[~torch.eye(len(first), dtype=torch.bool, device=device)]

        report[f"{k} vectors"] = {
            "single_prerequisite_assessed": {
                g: {"right": f"{right[g]}/{total[g]}",
                    "wrong_picks": {name: n for (gg, name), n in wrong.items() if gg == g}} for g in GROUPS},
            "cosine_first_vs_second_same_state": {"mean": round(float(same_state.mean()), 3),
                                                  "min": round(float(same_state.min()), 3),
                                                  "max": round(float(same_state.max()), 3)},
            "cosine_first_prerequisite_across_states_mean": round(float(across.mean()), 3),
            "states": len(plain),
        }
    (HERE / "results/analysis-prerequisites.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
