"""Choice pilot: where each arm is wrong, and the pass rule applied to the latent arm.

Reads per-case outputs only. No model, no training.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = {"starting adapter, text": "choice-ref-forkv1-text", "text arm": "choice-text", "latent arm": "choice-latent"}
HAND = json.loads((HERE / "choice_names.json").read_text())["names"]


def count(s):
    return int(s.split("/")[0])


def naming(c):
    if c["letter_names"]:
        return "single letters"
    by_hand = any(a["identity"] in HAND.get(c["family"], {}) for a in c["alternatives"])
    return "other names, mapped by hand" if by_hand else "other names, found by rule"


def wrong_kind(c, r):
    """What a wrong answer was: another alternative, a status in place of an alternative, and so on."""
    asked_status = c["status"] != "Assessed"
    picked_status = r["pick"] >= len(c["alternatives"])
    if asked_status:
        return "an alternative picked where a status was right" if not picked_status else "the wrong status"
    return "a status picked where an alternative was right" if picked_status else "the wrong alternative"


def main():
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/choice-development.jsonl"))}
    report, right = {}, {}
    for name, tag in ARMS.items():
        rows = {r["id"]: r for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl"))}
        assert rows.keys() == cases.keys()
        right[name] = {i: r["right"] for i, r in rows.items()}
        groups, wrong, families = defaultdict(list), Counter(), Counter()
        every_order = defaultdict(list)
        for i, c in cases.items():
            ok = rows[i]["right"]
            for key in (f"{c['kind']}, {naming(c)}", f"{c['kind']}, source status {c['source_status']}",
                        f"{c['kind']}, {len(c['alternatives'])} alternatives"):
                groups[key].append(ok)
            if c["kind"] == "select" and c["status"] == "Assessed":
                every_order[(naming(c), c["base_id"])].append(ok)
            if not ok:
                wrong[f"{c['kind']}: {wrong_kind(c, rows[i])}"] += 1
                families[c["family"]] += 1
        states = defaultdict(list)
        for (group, _), oks in every_order.items():
            states[group].append(all(oks))
        report[name] = {
            "summary": {k: v for k, v in json.load(open(HERE / f"results/{tag}.json")).items()
                        if k in ("all_questions", "kind_pairs_both_right", "assessed_states_right_in_every_order",
                                 "select_questions_in_listed_order")},
            "accuracy": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(groups.items())},
            "assessed_states_right_in_every_order_by_naming": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(states.items())},
            "what_the_wrong_answers_were": dict(sorted(wrong.items())),
            "wrong_answers_by_family": dict(families.most_common()),
        }
    text, latent = right["text arm"], right["latent arm"]
    report["text arm against latent arm"] = {
        "both right": sum(text[i] and latent[i] for i in cases), "only text right": sum(text[i] and not latent[i] for i in cases),
        "only latent right": sum(latent[i] and not text[i] for i in cases), "both wrong": sum(not text[i] and not latent[i] for i in cases),
    }

    t, l = report["text arm"]["summary"], report["latent arm"]["summary"]
    n = len(cases)
    need = {"1_all_questions": math.ceil(count(t["all_questions"]) - 0.05 * n),
            "2_kind_pairs_both_right": math.ceil(0.9 * count(t["kind_pairs_both_right"])),
            "3_assessed_states_right_in_every_order": math.ceil(0.9 * count(t["assessed_states_right_in_every_order"]))}
    got = {"1_all_questions": count(l["all_questions"]), "2_kind_pairs_both_right": count(l["kind_pairs_both_right"]),
           "3_assessed_states_right_in_every_order": count(l["assessed_states_right_in_every_order"])}
    conclusive = count(t["all_questions"]) >= 0.9 * n
    report["pass rule"] = {
        "0_conclusive": {"text arm": count(t["all_questions"]), "needed": math.ceil(0.9 * n), "met": conclusive},
        **{k: {"latent arm": got[k], "needed": need[k], "met": got[k] >= need[k]} for k in need},
        "passed": conclusive and all(got[k] >= need[k] for k in need),
    }
    (HERE / "results/analysis-choice.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
