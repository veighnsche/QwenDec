"""Does the latent arm fail only on "first" against "second", or on any name?

Development states all call their prerequisites "the first/second documented
prerequisite holds". Training states use 192 distinct names. This takes 150
training states with all eight questions (half of the questions were trained on,
half were held back) so the trained arms can be measured on distinct names.

  train_names.py sample    write data/latent/noul-train-sample.jsonl
  train_names.py report    break down results/*-trainnames*.cases.jsonl

Between the two, run evaluate.py with --cases data/latent/noul-train-sample.jsonl
and the tags listed in TAGS.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLE = ROOT / "data/latent/noul-train-sample.jsonl"
SEED = 20261111  # the selection seed of prepare.py
TAGS = ("diag-trainnames-text", "diag-trainnames-latent-k16",
        "pilot2-pointer-vector-trainnames", "pilot2-pointer-span-trainnames", "pilot3-canonical-trainnames")


def trained_on():
    by_state = defaultdict(list)
    for c in map(json.loads, open(ROOT / "data/latent/noul-train.jsonl")):
        by_state[c["base_id"]].append(c)
    rng, chosen = random.Random(SEED), set()
    for base in sorted(by_state):  # same order and draws as prepare.py
        group = by_state[base]
        chosen |= {c["id"] for c in (rng.sample(group, 4) if len(group) == 8 else group)}
    return by_state, chosen


def sample():
    by_state, chosen = trained_on()
    full = sorted(b for b, g in by_state.items() if len(g) == 8)
    cases = [c for b in random.Random(7).sample(full, 150) for c in by_state[b]]
    SAMPLE.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"{len(cases)} questions, {sum(c['id'] not in chosen for c in cases)} held back from training")


def report():
    _, chosen = trained_on()
    cases = {c["id"]: c for c in map(json.loads, open(SAMPLE))}
    plain = defaultdict(dict)
    for c in cases.values():
        if c["kind"] in ("pa", "pb") and not c["negated"]:
            plain[c["base_id"]][c["kind"]] = c

    def group(c):
        a, b = plain[c["base_id"]]["pa"], plain[c["base_id"]]["pb"]
        if a["status"] != "Assessed" or b["status"] != "Assessed":
            return "other prerequisite not stated"
        return "prerequisites agree" if a["p_true"] == b["p_true"] else "prerequisites differ"

    out = {}
    for tag in TAGS:
        right, total = Counter(), Counter()
        for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl")):
            c = cases[r["id"]]
            single = c["kind"] in ("pa", "pb")
            keys = [("trained on" if c["id"] in chosen else "held back") + (", single prerequisite" if single else ", whole rule")]
            if single and c["status"] == "Assessed":
                keys.append("single assessed, " + group(c))
            for key in keys:
                total[key] += 1
                right[key] += r["pick"] == r["gold"]
        out[tag] = {k: f"{right[k]}/{total[k]}" for k in sorted(total)}
    (HERE / "results/analysis-train-names.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"sample": sample, "report": report}[sys.argv[1]]()
