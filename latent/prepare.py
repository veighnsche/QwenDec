"""Prepare the Noul question variants for one arm of the pilot (text or latent).

Both arms use the same examples: for every state with eight questions, four are
drawn with a fixed seed; other states keep their question and its negation. One
pass. The output has the same layout as train/prepare.py, plus, for the latent
arm, where each row's K question positions start and which vectors go there.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import VALUES, gold_index, question_text, tokenise  # noqa: E402

MAX_OPTIONS, CHUNK = 16, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("text", "latent"), required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--positions-per-step", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    cases = [json.loads(x) for x in open(ROOT / "data/latent/noul-train.jsonl")]
    by_state = defaultdict(list)
    for c in cases:
        by_state[c["base_id"]].append(c)
    rng = random.Random(args.seed)
    chosen = []
    for base in sorted(by_state):
        group = by_state[base]
        chosen += rng.sample(group, 4) if len(group) == 8 else group
    selection = hashlib.sha256("\n".join(c["id"] for c in chosen).encode()).hexdigest()

    index = None
    if args.mode == "latent":
        store = torch.load(ROOT / f"data/latent/question-vectors-k{args.k}.pt")
        index = store["index"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = []
    for c in chosen:
        ids, gate_start, q_start = tokenise(tok, c, args.mode, args.k, pad)
        rows.append({"ids": ids, "gate_start": gate_start, "q_start": q_start, "gold": gold_index(c),
                     "q_index": index[question_text(c)] if index else -1})

    buckets = []
    for length in sorted({-(-len(r["ids"]) // CHUNK) * CHUNK for r in rows}):
        members = [r for r in rows if -(-len(r["ids"]) // CHUNK) * CHUNK == length]
        n = len(members)
        tokens = torch.full((n, length), pad, dtype=torch.int32)
        target = torch.zeros(n, MAX_OPTIONS)
        valid = torch.zeros(n, MAX_OPTIONS, dtype=torch.bool)
        for i, r in enumerate(members):
            tokens[i, : len(r["ids"])] = torch.tensor(r["ids"], dtype=torch.int32)
            target[i, r["gold"]] = 1.0
            valid[i, : len(VALUES)] = True
        buckets.append({
            "length": length, "batch": max(1, args.positions_per_step // length), "tokens": tokens,
            "gate_start": torch.tensor([r["gate_start"] for r in members]),
            "answer_pos": torch.tensor([len(r["ids"]) - 1 for r in members]),
            "target": target, "valid": valid, "credit_set": torch.zeros(n, dtype=torch.bool),
            "q_start": torch.tensor([r["q_start"] for r in members]),
            "q_index": torch.tensor([r["q_index"] for r in members]),
        })

    schedule = []
    for b, bucket in enumerate(buckets):
        mine = list(range(bucket["tokens"].shape[0]))
        rng.shuffle(mine)
        schedule += [(b, mine[s:s + bucket["batch"]]) for s in range(0, len(mine), bucket["batch"])]
    rng.shuffle(schedule)
    batch_max = max(b["batch"] for b in buckets)
    step_rows = torch.zeros(len(schedule), batch_max, dtype=torch.int64)
    step_weight = torch.zeros(len(schedule), batch_max)
    for t, (b, members) in enumerate(schedule):
        step_rows[t, : len(members)] = torch.tensor(members)
        step_weight[t, : len(members)] = 1.0

    name = "text" if args.mode == "text" else f"latent-k{args.k}"
    out = ROOT / f"data/latent/prepared-{name}.pt"
    meta = {
        "arm": name, "examples": len(chosen), "selection_sha256": selection, "steps": len(schedule), "passes": 1,
        "seed": args.seed, "answers": dict(Counter(str(VALUES[r["gold"]]) for r in rows)),
        "prompt_tokens_mean": round(sum(len(r["ids"]) for r in rows) / len(rows), 1),
        "executed_positions": sum(buckets[b]["length"] * buckets[b]["batch"] for b, _ in schedule),
        "buckets": [{"length": b["length"], "batch": b["batch"], "rows": b["tokens"].shape[0]} for b in buckets],
    }
    data = {"meta": meta, "buckets": buckets, "bucket_of_step": torch.tensor([b for b, _ in schedule]),
            "rows": step_rows, "weight": step_weight, "letters": torch.arange(32, 32 + MAX_OPTIONS)}
    if args.mode == "latent":
        data["vectors_file"] = f"data/latent/question-vectors-k{args.k}.pt"
        data["k"] = args.k
    torch.save(data, out)
    meta["file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "buckets"}), [(b["length"], b["batch"], b["rows"]) for b in meta["buckets"]])


if __name__ == "__main__":
    main()
