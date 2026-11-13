"""Word-sequence run: the eleven-part pass rule applied, and each measure against the earlier arms.

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
NOUL = {"text arm": ("panel4-text", PANEL), "vectors, Noul only": ("pilot4-canonical", PANEL),
        "vectors, joint adapter": ("joint-noul", PANEL), "words, three operations": ("words-noul", PANEL),
        "vectors, Noul only, development's own names": ("pilot4-canonical-original-names", ORIGINAL),
        "vectors, joint adapter, development's own names": ("joint-noul-original-names", ORIGINAL),
        "words, three operations, development's own names": ("words-noul-original-names", ORIGINAL)}
CHOICE = {"text arm": "choice-text", "vectors, Choice only": "choice-latent", "vectors, joint adapter": "joint-choice",
          "words, three operations": "words-choice"}
SCORE = {"text arm with rubric": "score-text", "words, no rubric, Score only": "score-text-norubric",
         "vectors, Score only": "score-latent", "words, three operations": "words-score"}
NEEDED = {"N1 assessed twin pairs": 385, "N2 all questions": 1115, "N3 free-form questions": 151,
          "N4 prerequisites differ": 65, "N5 answer follows moved pointer": 65,
          "C1 all questions": 921, "C2 kind pairs both right": 331, "C3 assessed states right in every order": 93,
          "S1 all questions": 291, "S2 steering pairs both right": 59, "S3 in-between values": 63}


def count(s):
    return int(s.split("/")[0].split(" ")[0])


def rate(hits):
    return f"{sum(hits)}/{len(hits)}"


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
        for key in (f"{c['kind']}, {naming(c)}", f"{c['kind']}, source status {c['source_status']}", f"kind {c['kind']}",
                    "single letters" if c["letter_names"] else "other names"):
            groups[key].append(rows[i]["right"])
        if not rows[i]["right"]:
            wrong[f"{c['kind']}: {wrong_kind(c, rows[i])}"] += 1
    summary = json.load(open(HERE / f"results/{tag}.json"))
    return {"all questions": summary["all_questions"], "kind pairs both right": summary["kind_pairs_both_right"],
            "assessed states right in every order": summary["assessed_states_right_in_every_order"],
            "select questions in listed order": summary["select_questions_in_listed_order"],
            **{k: rate(v) for k, v in sorted(groups.items())}, "what the wrong answers were": dict(sorted(wrong.items()))}


def score(tag):
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/score-development.jsonl"))}
    rows = {r["id"]: r for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl"))}
    groups = defaultdict(list)
    for i, c in cases.items():
        r = rows[i]
        wording = "rule wording as in training" if c["rule_wording_in_training"] else "rule wording unlike training"
        groups[f"{c['target_type']}, {wording}"].append(r["right"])
        groups[f"status {c['status']}"].append(r["right"])
        if c["target_type"] == "in between":
            level = None if isinstance(r["picked"], str) else r["picked"]
            lo, hi = c["interval"]
            for tolerance in (0.1, 0.2):
                groups[f"in between, within {tolerance}"].append(level is not None and max(lo - level, 0, level - hi) <= tolerance + 1e-9)
    summary = json.load(open(HERE / f"results/{tag}.json"))
    return {"all questions": summary["all_questions"], "status or end of the scale": summary["status_or_end_of_the_scale"],
            "in between": summary["in_between"], "steering pairs both right": summary["steering_pairs_both_right"],
            "steering pairs by target": summary["steering_pairs_by_target"],
            "mean distance of picked level": summary["mean_distance_of_picked_level"],
            **{k: rate(v) for k, v in sorted(groups.items())}}


def main():
    report = {"noul": {name: noul(tag, file) for name, (tag, file) in NOUL.items()},
              "choice": {name: choice(tag) for name, tag in CHOICE.items()},
              "score": {name: score(tag) for name, tag in SCORE.items()}}
    held = json.load(open(HERE / "results/words-noul-trainnames.json"))
    report["noul"]["words, three operations, training states with distinct names"] = held["accuracy"]["all questions"].split(" ")[0]

    n, c, s = (report[op]["words, three operations"] for op in ("noul", "choice", "score"))
    moved = json.load(open(HERE / "results/words-noul-swap-pointer.json"))["answer_matches_other_prerequisite"]
    got = {"N1 assessed twin pairs": count(n["assessed twin pairs"]), "N2 all questions": count(n["all questions"]),
           "N3 free-form questions": count(n["whole rule, free-form state"]), "N4 prerequisites differ": count(n["prerequisites differ"]),
           "N5 answer follows moved pointer": count(moved),
           "C1 all questions": count(c["all questions"]), "C2 kind pairs both right": count(c["kind pairs both right"]),
           "C3 assessed states right in every order": count(c["assessed states right in every order"]),
           "S1 all questions": count(s["all questions"]), "S2 steering pairs both right": count(s["steering pairs both right"]),
           "S3 in-between values": count(s["in between"])}
    parts = {k: {"needed": NEEDED[k], "measured": got[k], "met": got[k] >= NEEDED[k], "margin": got[k] - NEEDED[k]} for k in NEEDED}
    report["pass rule"] = {**parts, **{f"{name} met": all(v["met"] for k, v in parts.items() if k[0] == letter)
                                       for letter, name in (("N", "noul"), ("C", "choice"), ("S", "score"))},
                           "passed": all(v["met"] for v in parts.values())}
    report["in-between score values count as working (at least 67 of 134)"] = got["S3 in-between values"] >= 67
    (HERE / "results/analysis-words.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
