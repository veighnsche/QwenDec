"""Score variants for the request pilot, derived by rule from the imported cases.

Each case gives two questions on the same state:
  score       the case's own status, or its value
  complement  the same status, or one minus the value
No model and no tokenizer. The checks here are the ones the design asks for before
training that need no GPU: every complement mirrors its case, statuses are unchanged.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from score_common import right_levels  # noqa: E402

SEEN_IN_TRAINING = ("Completion score equals", "Quality is scored")  # the two rule templates of the training cases


def interior(interval):
    return 0 < interval[0] and interval[1] < 1


def variants(case, split):
    interval = case["value"]["score_interval"] if case["status"] == "Assessed" else None
    mirrored = [round(1 - interval[1], 6), round(1 - interval[0], 6)] if interval else None
    out = []
    for kind, value in (("score", interval), ("complement", mirrored)):
        out.append({
            "id": f"{case['id']}-{kind}", "base_id": case["id"], "split": split, "family": case["family"], "kind": kind,
            "twin": f"{case['id']}-{'complement' if kind == 'score' else 'score'}",
            "prompt": case["prompt"], "facts": case["facts"], "rules": case["rules"], "rubric": case["rubric"],
            "status": case["status"], "interval": value,
            "target_type": "status" if value is None else "in between" if interior(value) else "end of the scale",
            # a pair steers only if no single level is right for both of its questions
            "pair_has_no_shared_level": bool(interval) and not (right_levels(interval) & right_levels(mirrored)),
            "rule_wording_in_training": case["rules"][0].startswith(SEEN_IN_TRAINING),
            "rubric_adds_midpoint": "midpoint" in case["rubric"],
        })
    a, b = out
    assert a["status"] == b["status"], "a complement changed the status"
    if interval:
        assert abs((a["interval"][0] + b["interval"][1]) - 1) < 1e-6 and abs((a["interval"][1] + b["interval"][0]) - 1) < 1e-6
        assert a["target_type"] == b["target_type"]
    return out


def main():
    summary = {}
    for split in ("train", "development"):
        cases = [c for c in map(json.loads, open(ROOT / f"data/qwenonetwo/{split}.jsonl")) if c["operation"] == "Score"]
        out = [v for c in cases for v in variants(c, split)]
        (ROOT / f"data/latent/score-{split}.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in out))
        score = [v for v in out if v["kind"] == "score"]
        values = {round(sum(v["interval"]) / 2, 3) for v in out if v["interval"] and interior(v["interval"])}
        summary[split] = {
            "cases": len(cases), "questions": len(out), "kinds": dict(Counter(v["kind"] for v in out)),
            "status": dict(Counter(v["status"] for v in out)),
            "questions_by_target": dict(Counter(v["target_type"] for v in out)),
            "assessed_pairs": sum(v["status"] == "Assessed" for v in score),
            "assessed_pairs_with_no_shared_level": dict(Counter(v["target_type"] for v in score if v["pair_has_no_shared_level"])),
            "distinct_in_between_values": len(values),
            "questions_with_rule_wording_in_training": sum(v["rule_wording_in_training"] for v in out),
            "questions_whose_rubric_adds_a_midpoint": sum(v["rubric_adds_midpoint"] for v in out),
            "complements_mirror_their_case": True, "statuses_unchanged": True,  # asserted above for every case
        }
    (HERE / "results/score-cases.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
