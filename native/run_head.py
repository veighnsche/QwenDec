"""Train the reader on the frozen native model's states, then let it produce requests.

Everything that runs a model runs on the RTX 3090:
  1. the frozen base model (no adapter) reads every state once; its final-layer states
     are kept on the GPU
  2. the head is trained on the training states that are not held out, in the on-device
     loop: the GPU runs `--launch-steps` steps per launch by itself
  3. the trained head produces an operation and item ranges for the development states
     and for the held-out training states

`--check` runs the loop against the same steps driven from Python instead, bit for bit.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(HERE))
from head import SLOTS, Head, HeadTrainer, read_out  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402
from reading import OPERATIONS, native_pass, state_text  # noqa: E402

SEED = 20261113


@torch.no_grad()
def read_states(model, token_lists, device):
    """Final-layer states of the frozen model for every state, as one flat tensor on the GPU."""
    total = sum(len(t) for t in token_lists)
    flat = torch.zeros(total, model.config.hidden_size, dtype=torch.bfloat16, device=device)
    offsets, at = [], 0
    for t in token_lists:
        offsets.append(at)
        at += len(t)
    order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
    for start in range(0, len(order), 32):
        batch = order[start:start + 32]
        width = max(len(token_lists[i]) for i in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long, device=device)
        for row, i in enumerate(batch):
            ids[row, : len(token_lists[i])] = torch.tensor(token_lists[i])
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state  # right padding is causal-safe
        for row, i in enumerate(batch):
            n = len(token_lists[i])
            flat[offsets[i]:offsets[i] + n] = hidden[row, :n].to(torch.bfloat16)
    return flat, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/native-head"))
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--launch-steps", type=int, default=100)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = require_gpu(blocking_sync=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    records = [json.loads(x) for split in ("train", "development") for x in open(ROOT / f"data/native/states-{split}.jsonl")]
    token_lists = []
    for r in records:
        ids, _, first, _ = native_pass(tok, state_text(r))
        assert len(ids) == r["tokens"] and first == r["first_state_token"], "the labels were made for other tokens"
        token_lists.append(ids)

    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    assert all(p.is_cuda for p in model.parameters()), "a model tensor is on the CPU"
    features, offsets = read_states(model, token_lists, device)
    hidden = model.config.hidden_size
    del model
    torch.cuda.empty_cache()

    length = max(r["tokens"] for r in records)
    places = torch.full((len(records), 2 * SLOTS), length, dtype=torch.long)  # the common length means "empty"
    for i, r in enumerate(records):
        for slot, item in enumerate(r["items"]):
            places[i, 2 * slot], places[i, 2 * slot + 1] = item["token_range"]
    train = [i for i, r in enumerate(records) if r["split"] == "train" and not r["held_out"]]
    schedule = []
    for p in range(args.passes):
        order = train[:]
        random.Random(SEED + p).shuffle(order)
        schedule += [order[s:s + args.batch] for s in range(0, len(order), args.batch)]
    rows = torch.zeros(len(schedule), args.batch, dtype=torch.long)
    weight = torch.zeros(len(schedule), args.batch)
    for t, members in enumerate(schedule):
        rows[t, : len(members)] = torch.tensor(members)
        weight[t, : len(members)] = 1.0
    data = {"features": features, "offset": torch.tensor(offsets, device=device),
            "tokens": torch.tensor([r["tokens"] for r in records], device=device),
            "first": torch.tensor([r["first_state_token"] for r in records], device=device),
            "operation": torch.tensor([OPERATIONS.index(r["operation"]) for r in records], device=device),
            "places": places.to(device), "rows": rows.to(device), "weight": weight.to(device), "length": length}

    head = Head(hidden).to(device)
    trainer = HeadTrainer(head, data, device, lr=args.lr, weight_decay=args.weight_decay, clip=args.clip)
    print(json.dumps({"states": len(records), "trained_on": len(train), "steps": trainer.steps, "common_length": length,
                      "head_parameters": sum(p.numel() for p in head.parameters()),
                      "native_states_gib": round(features.numel() * 2 / 2**30, 2)}), flush=True)

    if args.check:
        same = lambda a, b: all(torch.equal(torch.nan_to_num(a[k]), torch.nan_to_num(b[k])) for k in a)
        init = trainer.snapshot()
        trainer.warm_up()
        trainer.run_eager(6)
        torch.cuda.synchronize()
        eager = trainer.snapshot()
        trainer.restore(init)
        graph = trainer.record()
        trainer.restore(init)
        trainer.stop.fill_(6)
        graph.replay()
        torch.cuda.synchronize()
        print(f"loop against eager over 6 steps: bit-identical {same(eager, trainer.snapshot())}; "
              f"losses {[round(x, 3) for x in trainer.loss_log[:6].tolist()]}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels_sha = {split: hashlib.sha256((ROOT / f"data/native/states-{split}.jsonl").read_bytes()).hexdigest() for split in ("train", "development")}
    manifest = {"base_model": "Qwen3.5-4B", "native_model_trained": False, "reads": "final-layer states, adapter off",
                "labels_sha256": labels_sha, "trained_on": len(train), "held_out": sum(r.get("held_out", False) for r in records),
                "passes": args.passes, "batch": args.batch, "steps": trainer.steps, "lr": args.lr, "weight_decay": args.weight_decay,
                "clip": args.clip, "betas": [0.9, 0.999], "seed": SEED, "deterministic": True, "torch": torch.__version__,
                "head_parameters": sum(p.numel() for p in head.parameters())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    trainer.warm_up()
    graph = trainer.record()
    done = 0
    with open(out / "log.jsonl", "w") as log:
        while done < trainer.steps and int(trainer.bad) < trainer.bad_limit:
            cpu, wall = time.process_time(), time.perf_counter()
            trainer.stop.fill_(min(done + args.launch_steps, trainer.steps))
            graph.replay()
            torch.cuda.synchronize()  # the host sleeps here while the GPU runs the steps
            cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
            now = int(trainer.t)
            line = {"step": now, "mean_loss": round(float(trainer.loss_log[done:now].nanmean()), 4),
                    "mean_grad_norm": round(float(trainer.gnorm_log[done:now].nanmean()), 3), "skipped_total": int(trainer.bad),
                    "s_per_step": round(wall / max(1, now - done), 4), "host_cpu_percent": round(100 * cpu / wall, 2)}
            log.write(json.dumps(line) + "\n")
            log.flush()
            print(json.dumps(line), flush=True)
            done = now
    finished = done >= trainer.steps
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, out / "head.pt")
    (out / "result.json").write_text(json.dumps({
        "steps_executed": done, "steps_planned": trainer.steps, "skipped_steps": int(trainer.bad), "execution_complete": finished,
        "learning_accepted": None,
        "note": "learning_accepted stays empty until the produced requests are judged against the agreed rule"}, indent=1) + "\n")
    if not finished:
        raise SystemExit("training did not finish; nothing is produced")

    # What the trained head produces, for development and for the held-out training states.
    head.eval()
    produced = []
    wanted = [i for i, r in enumerate(records) if r["split"] == "development" or r["held_out"]]
    with torch.no_grad():
        for start in range(0, len(wanted), args.batch):
            members = wanted[start:start + args.batch]
            index = torch.tensor(members, device=device)
            states, last, pointable = trainer.batch(index)
            operation, place_logits = head(states, last, pointable)
            for row, i in enumerate(members):
                r = records[i]
                ranges = read_out(place_logits[row].float().cpu(), length)
                produced.append({"id": r["id"], "split": r["split"], "operation": OPERATIONS[int(operation[row].argmax())],
                                 "ranges": [list(x) for x in ranges],
                                 "texts": [tok.decode(token_lists[i][a:b + 1]) for a, b in ranges]})
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/produced.json").write_text(json.dumps(produced, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"produced_for": len(produced)}))


if __name__ == "__main__":
    main()
