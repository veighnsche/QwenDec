"""Score pilot: the pass rule applied to the latent arm, and where each arm is wrong.

Reads per-case outputs only. No model, no training.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "baseline"))
from measure_decision4b import interval_distance  # noqa: E402

ARMS = {"starting adapter, text": "score-ref-forkv1-text", "text arm": "score-text", "latent arm": "score-latent"}


def count(s):
    return int(s.split("/")[0])


def rate(hits):
    return f"{sum(hits)}/{len(hits)}"


def main():
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/score-development.jsonl"))}
    report, right = {}, {}
    for name, tag in ARMS.items():
        rows = {r["id"]: r for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl"))}
        assert rows.keys() == cases.keys()
        right[name] = {i: r["right"] for i, r in rows.items()}
        groups, near = defaultdict(list), defaultdict(list)
        mirrors, moved = [], []
        for i, c in cases.items():
            r = rows[i]
            wording = "rule wording as in training" if c["rule_wording_in_training"] else "rule wording unlike training"
            groups[f"{c['target_type']}, {wording}"].append(r["right"])
            groups[f"{c['target_type']}, kind {c['kind']}"].append(r["right"])
            if c["target_type"] == "in between":
                level = None if isinstance(r["picked"], str) else r["picked"]
                for tolerance in (0.05, 0.1, 0.15, 0.2):
                    near[f"within {tolerance}"].append(level is not None and interval_distance(level, c["interval"]) <= tolerance + 1e-9)
                if c["kind"] == "score":
                    other = rows[c["twin"]]["picked"]
                    if level is not None and not isinstance(other, str):
                        # do the two answers on one state mirror each other, whether or not they are right?
                        mirrors.append(abs(level + other - 1) <= 0.1 + 1e-9)
                        moved.append(abs(level - other) > 1e-9)
        summary = json.load(open(HERE / f"results/{tag}.json"))
        report[name] = {
            "all questions": summary["all_questions"], "status or end of the scale": summary["status_or_end_of_the_scale"],
            "in between": summary["in_between"], "steering pairs both right": summary["steering_pairs_both_right"],
            "steering pairs by target": summary["steering_pairs_by_target"],
            "mean distance of picked level": summary["mean_distance_of_picked_level"],
            "accuracy": {k: rate(v) for k, v in sorted(groups.items())},
            "in-between questions by distance of the picked level": {k: rate(v) for k, v in near.items()},
            "in-between pairs whose two answers add up to 1, within 0.1": rate(mirrors),
            "in-between pairs whose two answers differ": rate(moved),
        }
    text, latent = right["text arm"], right["latent arm"]
    for kind in ("status", "end of the scale", "in between"):
        ids = [i for i, c in cases.items() if c["target_type"] == kind]
        report[f"text arm against latent arm, {kind}"] = {
            "both right": sum(text[i] and latent[i] for i in ids), "only text right": sum(text[i] and not latent[i] for i in ids),
            "only latent right": sum(latent[i] and not text[i] for i in ids), "both wrong": sum(not text[i] and not latent[i] for i in ids)}

    t, l = report["text arm"], report["latent arm"]
    n, between = len(cases), sum(c["target_type"] == "in between" for c in cases.values())
    need = {"1 all questions": math.ceil(count(t["all questions"]) - 0.05 * n),
            "2 steering pairs both right": math.ceil(0.9 * count(t["steering pairs both right"])),
            "3 in-between values": math.ceil(count(t["in between"]) - 0.05 * between)}
    got = {"1 all questions": count(l["all questions"]), "2 steering pairs both right": count(l["steering pairs both right"]),
           "3 in-between values": count(l["in between"])}
    conclusive = count(t["status or end of the scale"]) >= 238
    report["pass rule"] = {
        "0 conclusive": {"text arm": count(t["status or end of the scale"]), "needed": 238, "met": conclusive},
        **{k: {"latent arm": got[k], "needed": need[k], "met": got[k] >= need[k]} for k in need},
        "passed": conclusive and all(got[k] >= need[k] for k in need),
    }
    report["in-between values count as working (at least half)"] = {
        name: count(report[name]["in between"]) * 2 >= between for name in ARMS}
    (HERE / "results/analysis-score.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
