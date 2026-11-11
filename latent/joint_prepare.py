"""Prepare the joint Noul and Choice run: one adapter, one projection, eight requests.

Nothing is tokenised or encoded again. The rows are taken as they are from the two
prepared files the separate arms trained on, the two request vector files are put
end to end, rows of equal padded length share a bucket, and the steps get a new
seeded order. Tensor bookkeeping only: no model, no GPU.

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
SOURCES = [("noul", "data/latent/prepared-canonical-span-k16-order4.pt"),
           ("choice", "data/latent/prepared-choice-latent.pt")]
VECTORS = "data/latent/question-vectors-k16-joint.pt"
ROW_KEYS = ("tokens", "gate_start", "answer_pos", "target", "valid", "credit_set", "q_start", "q_index")


def load(path):
    data = torch.load(ROOT / path)
    meta = json.loads((ROOT / path).with_suffix(".json").read_text())
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == meta["file_sha256"], f"{path} is not the file its summary describes"
    return data, meta


def joint_vectors(sources):
    """The request vectors of both files, end to end. Returns where each file's requests start."""
    stores = [torch.load(ROOT / data["vectors_file"]) for data, _ in sources]
    assert len({s["k"] for s in stores}) == 1 and len({s["vectors"].dtype for s in stores}) == 1
    offsets, index, at = [], {}, 0
    for s in stores:
        offsets.append(at)
        assert not set(s["index"]) & set(index), "two requests share a wording"
        index.update({text: i + at for text, i in s["index"].items()})
        at += s["vectors"].shape[0]
    torch.save({"vectors": torch.cat([s["vectors"] for s in stores]), "index": index, "k": stores[0]["k"]}, ROOT / VECTORS)
    return offsets, stores


def verify(out, sources, offsets, stores):
    """Read the written file back: every source row is in it exactly once, unchanged."""
    data = torch.load(out)
    joint = torch.load(ROOT / VECTORS)["vectors"]
    seen = Counter()
    for bucket in data["buckets"]:
        for i in range(bucket["tokens"].shape[0]):
            op, b, r = int(bucket["operation"][i]), int(bucket["source_bucket"][i]), int(bucket["source_row"][i])
            source = sources[op][0]["buckets"][b]
            assert source["length"] == bucket["length"]
            for key in ROW_KEYS:
                if key == "q_index":  # the same request vectors, at their new place in the joint file
                    assert torch.equal(joint[int(bucket[key][i])], stores[op]["vectors"][int(source[key][r])])
                else:
                    assert torch.equal(bucket[key][i], source[key][r]), (key, op, b, r)
            seen[(op, b, r)] += 1
    expected = sum(b["tokens"].shape[0] for data_, _ in sources for b in data_["buckets"])
    assert len(seen) == expected and set(seen.values()) == {1}, "a source row is missing or repeated"
    for op, store in enumerate(stores):
        n = store["vectors"].shape[0]
        assert torch.equal(joint[offsets[op]:offsets[op] + n], store["vectors"])
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
    ap.add_argument("--order-seed", type=int, default=5)
    args = ap.parse_args()

    sources = [load(path) for _, path in SOURCES]
    assert len({d["k"] for d, _ in sources}) == 1 and all(torch.equal(d["letters"], sources[0][0]["letters"]) for d, _ in sources)
    offsets, stores = joint_vectors(sources)

    lengths = sorted({b["length"] for d, _ in sources for b in d["buckets"]})
    buckets = []
    for length in lengths:
        parts = [(op, i, b) for op, (d, _) in enumerate(sources) for i, b in enumerate(d["buckets"]) if b["length"] == length]
        assert len({b["batch"] for _, _, b in parts}) == 1
        bucket = {"length": length, "batch": parts[0][2]["batch"]}
        for key in ROW_KEYS:
            if key == "q_index":
                bucket[key] = torch.cat([b[key] + offsets[op] for op, _, b in parts])
            else:
                bucket[key] = torch.cat([b[key] for _, _, b in parts])
        rows = [b["tokens"].shape[0] for _, _, b in parts]
        bucket["operation"] = torch.cat([torch.full((n,), op) for (op, _, _), n in zip(parts, rows)])
        bucket["source_bucket"] = torch.cat([torch.full((n,), i) for (_, i, _), n in zip(parts, rows)])
        bucket["source_row"] = torch.cat([torch.arange(n) for n in rows])
        buckets.append(bucket)

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

    names = [name for name, _ in SOURCES]
    holds = [sorted({names[int(buckets[b]["operation"][r])] for r in members}) for b, members in schedule]
    # Steps for the FP32 check: for each bucket its first step, one that holds both operations where there is one.
    check_steps = []
    for b in range(len(buckets)):
        mine = [t for t, (sb, _) in enumerate(schedule) if sb == b]
        both = [t for t in mine if len(holds[t]) == 2]
        check_steps.append((both or mine)[0])

    out = ROOT / "data/latent/prepared-joint.pt"
    meta = {
        "arm": "joint", "examples": sum(m["examples"] for _, m in sources), "steps": len(schedule), "passes": 1,
        "order_seed": args.order_seed,
        "sources": [{"operation": name, "file": path, "file_sha256": m["file_sha256"], "examples": m["examples"],
                     "selection_sha256": m["selection_sha256"], "requests": int(s["vectors"].shape[0]), "first_request": o}
                    for (name, path), (_, m), s, o in zip(SOURCES, sources, stores, offsets)],
        "steps_holding": dict(Counter(" and ".join(h) for h in holds)),
        "check_steps": check_steps,
        "executed_positions": sum(buckets[b]["length"] * buckets[b]["batch"] for b, _ in schedule),
        "buckets": [{"length": b["length"], "batch": b["batch"], "rows": b["tokens"].shape[0],
                     **{name: int((b["operation"] == op).sum()) for op, name in enumerate(names)}} for b in buckets],
    }
    data = {"meta": meta, "buckets": buckets, "bucket_of_step": torch.tensor([b for b, _ in schedule]),
            "rows": step_rows, "weight": step_weight, "letters": sources[0][0]["letters"],
            "vectors_file": VECTORS, "k": sources[0][0]["k"]}
    torch.save(data, out)
    meta["rows_verified_against_sources"] = verify(out, sources, offsets, stores)
    meta["file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    meta["vectors_sha256"] = hashlib.sha256((ROOT / VECTORS).read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
