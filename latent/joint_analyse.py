"""Joint Noul and Choice run: the pass rule applied, and each measure against the separate arms.

Reads per-case outputs only. No model, no training.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from check_canonical import group  # noqa: E402
from choice_analyse import naming, wrong_kind  # noqa: E402

PANEL, ORIGINAL = "noul-development-panel4", "noul-development"
NOUL = {"text arm, new panel": ("panel4-text", PANEL), "pilot 4, new panel": ("pilot4-canonical", PANEL),
        "joint, new panel": ("joint-noul", PANEL),
        "pilot 4, development's own names": ("pilot4-canonical-original-names", ORIGINAL),
        "joint, development's own names": ("joint-noul-original-names", ORIGINAL)}
CHOICE = {"text arm": "choice-text", "choice pilot": "choice-latent", "joint": "joint-choice"}
NEEDED = {"N1 assessed twin pairs": 385, "N2 all questions": 1115, "N3 free-form questions": 151,
          "N4 prerequisites differ": 65, "N5 answer follows moved pointer": 65,
          "C1 all questions": 921, "C2 kind pairs both right": 331, "C3 assessed states right in every order": 93}


def count(s):
    return int(s.split("/")[0].split(" ")[0])


def noul(tag, file):
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / f"data/latent/{file}.jsonl"))}
    right, total = Counter(), Counter()
    for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl")):
        c = cases[r["id"]]
        for key in ("all questions", group(c).replace(" (own wording in both)", ""), f"status {c['status']}",
                    f"gold status {c['status']}, {'free-form' if 'free-form' in group(c) else 'other'} question"):
            total[key] += 1
            right[key] += r["pick"] == r["gold"]
    summary = json.load(open(HERE / f"results/{tag}.json"))
    return {"assessed twin pairs": summary["both_twins_right"]["assessed twin pairs"].split(" ")[0],
            "prerequisites differ": summary["single_assessed_prerequisites_differ"].split(" ")[0],
            **{k: f"{right[k]}/{total[k]}" for k in sorted(total)}}


def choice(tag):
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/choice-development.jsonl"))}
    rows = {r["id"]: r for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl"))}
    groups, wrong = defaultdict(list), Counter()
    for i, c in cases.items():
        ok = rows[i]["right"]
        for key in (f"{c['kind']}, {naming(c)}", f"{c['kind']}, source status {c['source_status']}"):
            groups[key].append(ok)
        if not ok:
            wrong[f"{c['kind']}: {wrong_kind(c, rows[i])}"] += 1
    summary = json.load(open(HERE / f"results/{tag}.json"))
    return {"all questions": summary["all_questions"], "kind pairs both right": summary["kind_pairs_both_right"],
            "assessed states right in every order": summary["assessed_states_right_in_every_order"],
            "select questions in listed order": summary["select_questions_in_listed_order"],
            **{k: f"{sum(v)}/{len(v)}" for k, v in sorted(groups.items())},
            "what the wrong answers were": dict(sorted(wrong.items())),
            "states with a wrong answer": len({cases[i]["base_id"] for i in cases if not rows[i]["right"]})}, \
        {i: r["right"] for i, r in rows.items()}


def main():
    report = {"noul": {name: noul(tag, file) for name, (tag, file) in NOUL.items()}, "choice": {}}
    right = {}
    for name, tag in CHOICE.items():
        report["choice"][name], right[name] = choice(tag)
    a, b = right["choice pilot"], right["joint"]
    report["choice"]["choice pilot against joint"] = {
        "both right": sum(a[i] and b[i] for i in a), "only choice pilot right": sum(a[i] and not b[i] for i in a),
        "only joint right": sum(b[i] and not a[i] for i in a), "both wrong": sum(not a[i] and not b[i] for i in a)}

    n, c = report["noul"]["joint, new panel"], report["choice"]["joint"]
    moved = json.load(open(HERE / "results/joint-noul-swap-pointer.json"))["answer_matches_other_prerequisite"]
    got = {"N1 assessed twin pairs": count(n["assessed twin pairs"]), "N2 all questions": count(n["all questions"]),
           "N3 free-form questions": count(n["whole rule, free-form state"]), "N4 prerequisites differ": count(n["prerequisites differ"]),
           "N5 answer follows moved pointer": count(moved), "C1 all questions": count(c["all questions"]),
           "C2 kind pairs both right": count(c["kind pairs both right"]),
           "C3 assessed states right in every order": count(c["assessed states right in every order"])}
    parts = {k: {"needed": NEEDED[k], "measured": got[k], "met": got[k] >= NEEDED[k]} for k in NEEDED}
    report["pass rule"] = {**parts, "noul met": all(v["met"] for k, v in parts.items() if k[0] == "N"),
                           "choice met": all(v["met"] for k, v in parts.items() if k[0] == "C"),
                           "passed": all(v["met"] for v in parts.values())}
    (HERE / "results/analysis-joint.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
