"""Prepare the run with fixed word-sequence requests: one adapter, Noul, Choice and Score.

Nothing is derived again from the cases. Every row of the three prepared files that
the vector arms trained on is taken as it is, and its K vector positions are replaced
by the fixed word sequence of its request. State, pointers, options and targets stay.
Rows of equal padded length share a bucket and the steps get a new seeded order.
Tokenizer and tensor bookkeeping only: no model, no GPU.

After writing, the file is read back and every row is compared with its source.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from words import fixed_ids  # noqa: E402

SOURCES = [("noul", "data/latent/prepared-canonical-span-k16-order4.pt"),
           ("choice", "data/latent/prepared-choice-latent.pt"),
           ("score", "data/latent/prepared-score-latent.pt")]
MAX_OPTIONS, CHUNK = 16, 64


def load(path):
    data = torch.load(ROOT / path)
    meta = json.loads((ROOT / path).with_suffix(".json").read_text())
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == meta["file_sha256"], f"{path} is not the file its summary describes"
    index = torch.load(ROOT / data["vectors_file"])["index"]
    wording = {i: text for text, i in index.items()}  # which request each row's vectors stood for
    return data, meta, wording


def rows_of(tok, op, data, wording):
    k, out = data["k"], []
    for b, bucket in enumerate(data["buckets"]):
        for r in range(bucket["tokens"].shape[0]):
            n, q = int(bucket["answer_pos"][r]) + 1, int(bucket["q_start"][r])
            ids = bucket["tokens"][r, :n].tolist()
            text = wording[int(bucket["q_index"][r])]
            request = fixed_ids(tok, text)
            out.append({"ids": ids[:q] + request + ids[q + k:], "gate_start": int(bucket["gate_start"][r]),
                        "target": bucket["target"][r], "valid": bucket["valid"][r], "credit_set": bool(bucket["credit_set"][r]),
                        "request_start": q, "request_len": len(request), "request": text,
                        "operation": op, "source_bucket": b, "source_row": r})
    return out


def verify(out, sources, tok):
    """Read the written file back: every source row is in it once, unchanged outside its request."""
    data = torch.load(out)
    requests = data["requests"]
    seen = Counter()
    for bucket in data["buckets"]:
        for i in range(bucket["tokens"].shape[0]):
            op, b, r = int(bucket["operation"][i]), int(bucket["source_bucket"][i]), int(bucket["source_row"][i])
            source_data, _, wording = sources[op]
            source, k = source_data["buckets"][b], source_data["k"]
            n, q = int(source["answer_pos"][r]) + 1, int(source["q_start"][r])
            old = source["tokens"][r, :n].tolist()
            new = bucket["tokens"][i, : int(bucket["answer_pos"][i]) + 1].tolist()
            start, length = int(bucket["request_start"][i]), int(bucket["request_len"][i])
            text = requests[int(bucket["request"][i])]
            assert start == q and text == wording[int(source["q_index"][r])]
            assert new[:start] == old[:q], "the state changed"
            assert new[start:start + length] == fixed_ids(tok, text), "the request is not its fixed sequence"
            assert new[start + length:] == old[q + k:], "pointers or options changed"
            assert int(bucket["gate_start"][i]) == int(source["gate_start"][r]) <= start
            for key in ("target", "valid", "credit_set"):
                assert torch.equal(bucket[key][i], source[key][r]), key
            assert (bucket["tokens"][i, len(new):] == data["pad"]).all()
            seen[(op, b, r)] += 1
    expected = sum(b["tokens"].shape[0] for d, _, _ in sources for b in d["buckets"])
    assert len(seen) == expected and set(seen.values()) == {1}, "a source row is missing or repeated"
    scheduled = Counter()
    for t in range(data["rows"].shape[0]):
        b = int(data["bucket_of_step"][t])
        for j in range(data["buckets"][b]["batch"]):
            if float(data["weight"][t, j]):
                scheduled[(b, int(data["rows"][t, j]))] += 1
    assert len(scheduled) == expected and set(scheduled.values()) == {1}, "a row is not trained exactly once"
    return expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-seed", type=int, default=7)
    ap.add_argument("--positions-per-step", type=int, default=4096)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    sources = [load(path) for _, path in SOURCES]
    assert all(torch.equal(d["letters"], sources[0][0]["letters"]) for d, _, _ in sources)
    names = [name for name, _ in SOURCES]
    rows = [row for op, (data, _, wording) in enumerate(sources) for row in rows_of(tok, op, data, wording)]
    requests = sorted({r["request"] for r in rows})

    buckets = []
    for length in sorted({-(-len(r["ids"]) // CHUNK) * CHUNK for r in rows}):
        members = [r for r in rows if -(-len(r["ids"]) // CHUNK) * CHUNK == length]
        n = len(members)
        tokens = torch.full((n, length), pad, dtype=torch.int32)
        for i, r in enumerate(members):
            tokens[i, : len(r["ids"])] = torch.tensor(r["ids"], dtype=torch.int32)
        buckets.append({
            "length": length, "batch": max(1, args.positions_per_step // length), "tokens": tokens,
            "gate_start": torch.tensor([r["gate_start"] for r in members]),
            "answer_pos": torch.tensor([len(r["ids"]) - 1 for r in members]),
            "target": torch.stack([r["target"] for r in members]), "valid": torch.stack([r["valid"] for r in members]),
            "credit_set": torch.tensor([r["credit_set"] for r in members]),
            # bookkeeping, not used by the training step
            "request_start": torch.tensor([r["request_start"] for r in members]),
            "request_len": torch.tensor([r["request_len"] for r in members]),
            "request": torch.tensor([requests.index(r["request"]) for r in members]),
            "operation": torch.tensor([r["operation"] for r in members]),
            "source_bucket": torch.tensor([r["source_bucket"] for r in members]),
            "source_row": torch.tensor([r["source_row"] for r in members]),
        })

    rng = random.Random(args.order_seed)
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

    holds = [sorted({names[int(buckets[b]["operation"][r])] for r in members}) for b, members in schedule]
    # Steps for the FP32 check: for each bucket, the first step holding the most operations.
    check_steps = []
    for b in range(len(buckets)):
        mine = [t for t, (sb, _) in enumerate(schedule) if sb == b]
        check_steps.append(max(mine, key=lambda t: (len(holds[t]), -t)))

    out = ROOT / "data/latent/prepared-words.pt"
    meta = {
        "arm": "words", "examples": len(rows), "steps": len(schedule), "passes": 1, "order_seed": args.order_seed,
        "sources": [{"operation": name, "file": path, "file_sha256": m["file_sha256"], "examples": m["examples"],
                     "selection_sha256": m["selection_sha256"]} for (name, path), (_, m, _) in zip(SOURCES, sources)],
        "requests": {text: {"tokens": len(fixed_ids(tok, text)), "rows": sum(r["request"] == text for r in rows)} for text in requests},
        "steps_holding": dict(Counter(" and ".join(h) for h in holds)),
        "check_steps": check_steps,
        "prompt_tokens_mean": round(sum(len(r["ids"]) for r in rows) / len(rows), 1),
        "executed_positions": sum(buckets[b]["length"] * buckets[b]["batch"] for b, _ in schedule),
        "buckets": [{"length": b["length"], "batch": b["batch"], "rows": b["tokens"].shape[0],
                     **{name: int((b["operation"] == op).sum()) for op, name in enumerate(names)}} for b in buckets],
    }
    data = {"meta": meta, "buckets": buckets, "bucket_of_step": torch.tensor([b for b, _ in schedule]),
            "rows": step_rows, "weight": step_weight, "letters": sources[0][0]["letters"], "requests": requests, "pad": pad}
    torch.save(data, out)
    meta["rows_verified_against_sources"] = verify(out, sources, tok)
    meta["file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
