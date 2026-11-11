"""Turn the imported cases into tensors the GPU loop can train on.

Runs once on the host before training; it tokenises text and runs no model. Each
case becomes one prompt in the adapter's own format, with the non-assessed statuses
offered as extra options, plus:
  gate_start  first position where the adapter is on (the state before it is read natively)
  answer_pos  the position whose option-letter logits are trained
  target      weight over the options (see below)
Prompts are right-padded into length buckets that are multiples of 64, and a seeded
order of batches is fixed for every pass.

Targets:
  Noul, assessed    the true or the false option
  Choice, assessed  every acceptable alternative; the loss credits the set as a whole
  Score, assessed   the two rubric levels around the target, weighted so the expected
                    level equals the target
  not assessed      the matching status option
Choice alternatives are listed in a fresh random order on every pass, because the
source lists the right one first far more often than chance.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "gpu_loop"))
from measure_decision4b import LETTERS, LEVELS, STATUS_OPTIONS, SYSTEM, build  # noqa: E402

STATE_END = '", "question": "'
MAX_OPTIONS = 16
CHUNK = 64


def target_weights(case, values):
    """Weight per option, and whether the loss credits the weighted set as a whole."""
    w = [0.0] * len(values)
    if case["status"] != "Assessed":
        w[values.index(case["status"])] = 1.0
        return w, False
    if case["operation"] == "Noul":
        w[values.index(bool(case["value"]["p_true"]))] = 1.0
        return w, False
    if case["operation"] == "Choice":
        for identity in case["value"]["acceptable_ids"]:
            w[values.index(identity)] = 1.0
        return w, True
    lo, hi = case["value"]["score_interval"]
    x = min(max((lo + hi) / 2, 0.0), 1.0) * (len(LEVELS) - 1)
    below = min(int(x), len(LEVELS) - 2)
    w[below], w[below + 1] = 1.0 - (x - below), x - below
    return w, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "data/qwenonetwo/train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/prepared/train-v1.pt"))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--positions-per-step", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    cases = [json.loads(x) for x in open(args.cases)]
    rng = random.Random(args.seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    letter_ids = None
    first_listed_right = Counter()
    rows = []  # one per (pass, case)
    for p in range(args.passes):
        for index, case in enumerate(cases):
            if case["operation"] == "Choice":
                case = {**case, "alternatives": rng.sample(case["alternatives"], len(case["alternatives"]))}
            task, values = build(case, "status")
            weights, credit_set = target_weights(case, values)
            if case["operation"] == "Choice" and case["status"] == "Assessed":
                first_listed_right[weights[0] > 0] += 1

            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
            if letter_ids is None:  # the adapter card's rule: last token of prompt + letter
                letter_ids = [tok.encode(prompt + x, add_special_tokens=False)[-1] for x in LETTERS[:MAX_OPTIONS]]
            elif rng.random() < 0.01:
                assert letter_ids == [tok.encode(prompt + x, add_special_tokens=False)[-1] for x in LETTERS[:MAX_OPTIONS]]
            boundary = prompt.index(STATE_END) + 1
            gate_start = sum(end <= boundary for _, end in enc["offset_mapping"])
            assert 0 < gate_start < len(enc["input_ids"]) and len(values) <= MAX_OPTIONS
            rows.append({"pass": p, "case": index, "ids": enc["input_ids"], "gate_start": gate_start,
                         "weights": weights, "credit_set": credit_set})

    # Buckets by padded length; batch size keeps positions per step under the budget.
    lengths = sorted({-(-len(r["ids"]) // CHUNK) * CHUNK for r in rows})
    buckets = []
    for length in lengths:
        members = [r for r in rows if -(-len(r["ids"]) // CHUNK) * CHUNK == length]
        n = len(members)
        tokens = torch.full((n, length), pad, dtype=torch.int32)
        target = torch.zeros(n, MAX_OPTIONS)
        valid = torch.zeros(n, MAX_OPTIONS, dtype=torch.bool)
        for i, r in enumerate(members):
            tokens[i, : len(r["ids"])] = torch.tensor(r["ids"], dtype=torch.int32)
            target[i, : len(r["weights"])] = torch.tensor(r["weights"])
            valid[i, : len(r["weights"])] = True
        buckets.append({
            "length": length,
            "batch": max(1, args.positions_per_step // length),
            "tokens": tokens,
            "gate_start": torch.tensor([r["gate_start"] for r in members], dtype=torch.int64),
            "answer_pos": torch.tensor([len(r["ids"]) - 1 for r in members], dtype=torch.int64),
            "target": target,
            "valid": valid,
            "credit_set": torch.tensor([r["credit_set"] for r in members]),
            "case": torch.tensor([r["case"] for r in members], dtype=torch.int64),
            "pass": [r["pass"] for r in members],
        })

    # Seeded order: within each pass, every row once, batches from all buckets interleaved.
    batch_max = max(b["batch"] for b in buckets)
    schedule = []
    for p in range(args.passes):
        batches = []
        for b, bucket in enumerate(buckets):
            mine = [i for i, q in enumerate(bucket["pass"]) if q == p]
            rng.shuffle(mine)
            for start in range(0, len(mine), bucket["batch"]):
                batches.append((b, mine[start:start + bucket["batch"]]))
        rng.shuffle(batches)
        schedule += batches
    steps = len(schedule)
    bucket_of_step = torch.tensor([b for b, _ in schedule], dtype=torch.int64)
    step_rows = torch.zeros(steps, batch_max, dtype=torch.int64)
    step_weight = torch.zeros(steps, batch_max)
    for t, (b, members) in enumerate(schedule):
        step_rows[t, : len(members)] = torch.tensor(members)
        step_weight[t, : len(members)] = 1.0

    real = sum(len(r["ids"]) for r in rows)
    padded = sum(buckets[b]["length"] * buckets[b]["batch"] for b, _ in schedule)
    meta = {
        "cases_file": args.cases,
        "cases_sha256": hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
        "cases": len(cases), "passes": args.passes, "seed": args.seed, "steps": steps,
        "positions_per_step_budget": args.positions_per_step,
        "real_positions": real, "executed_positions": padded,
        "padding_share": round(1 - real / padded, 3),
        "prompt_tokens": {"min": min(len(r["ids"]) for r in rows), "max": max(len(r["ids"]) for r in rows),
                          "mean": round(real / len(rows), 1)},
        "adapter_on_share": round(1 - sum(r["gate_start"] for r in rows) / real, 3),
        "choice_right_answer_listed_first": dict(first_listed_right),
        "buckets": [{"length": b["length"], "batch": b["batch"], "rows": len(b["pass"])} for b in buckets],
        "letter_token_ids": letter_ids,
    }
    for b in buckets:
        del b["pass"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "buckets": buckets, "bucket_of_step": bucket_of_step, "rows": step_rows,
                "weight": step_weight, "letters": torch.tensor(letter_ids, dtype=torch.int64),
                "case_ids": [c["id"] for c in cases]}, out)
    meta["file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
