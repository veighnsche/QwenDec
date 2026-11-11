"""Mechanical checks of the fork-mode training step on real prepared data.

Runs a few dozen real steps on the RTX 3090 and throws the weights away. It proves
the machinery, not learning:
  1. data audit   every prepared row decodes back to its prompt, gate and target
  2. loss check   the batched, padded, gated loss equals an independent per-case one
  3. loop check   the on-device loop is bit-identical to the same steps run eagerly
  4. guard check  a poisoned step is skipped on the GPU and counted
  5. throughput   steps per second, host CPU and GPU utilisation over a longer launch
  6. resume       (second invocation, fresh process) a saved checkpoint continues to
                  the bit-identical final state
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_decision4b import LEVELS, STATUS_OPTIONS  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402
from proof import Meter, smi  # noqa: E402
from step import Trainer  # noqa: E402


def audit_data(data, tok, cases):
    """Decode every prepared row and check it against the source case."""
    checked = 0
    for bucket in data["buckets"]:
        for i in range(bucket["tokens"].shape[0]):
            case = cases[int(bucket["case"][i])]
            n = int(bucket["answer_pos"][i]) + 1
            ids = bucket["tokens"][i, :n].tolist()
            start = int(bucket["gate_start"][i])
            # The switch token is usually `.",`: the state's last full stop shares a token with its closing quote.
            assert tok.decode(ids[start:]).find('", "question": "') in (0, 1), "gate is not at the end of the state"
            assert '"question"' not in tok.decode(ids[:start]), "gate is after the question started"
            text = tok.decode(ids)
            task = json.loads(text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
            keys = [o["key"] for o in task["options"]]
            w = bucket["target"][i, : len(keys)].tolist()
            assert bucket["valid"][i].sum() == len(keys) and abs(sum(w) - (sum(x > 0 for x in w) if bucket["credit_set"][i] else 1)) < 1e-5
            chosen = {k for k, x in zip(keys, w) if x > 0}
            if case["status"] != "Assessed":
                assert chosen == {STATUS_OPTIONS[case["status"]][0]}
            elif case["operation"] == "Noul":
                assert chosen == {"true" if case["value"]["p_true"] else "false"}
            elif case["operation"] == "Choice":
                assert chosen == set(case["value"]["acceptable_ids"])
            else:
                expected = sum(x * float(k) for k, x in zip(keys, w) if x > 0)
                assert abs(expected - sum(case["value"]["score_interval"]) / 2) < 1e-5
            assert case["prompt"] in task["state"]
            checked += 1
    return checked


@torch.no_grad()
def reference_loss(model, trainer, t):
    """Independent loss for step t: stock forward, one unpadded case at a time."""
    b = int(trainer.bucket_of_step[t])
    d = trainer.buckets[b]
    total = weight_sum = 0.0
    for j in range(d["batch"]):
        weight = float(trainer.weight[t, j])
        if weight == 0:
            continue
        row = int(trainer.rows[t, j])
        n = int(d["answer_pos"][row]) + 1
        ids = d["tokens"][row, :n].long().unsqueeze(0)
        trainer.gate.mask = (torch.arange(n, device=ids.device) >= d["gate_start"][row]).view(1, n, 1)
        logits = model(input_ids=ids).logits[0, -1].float()
        k = int(d["valid"][row].sum())
        logp = logits[trainer.letters[:k]].log_softmax(-1)
        w = d["target"][row, :k]
        if bool(d["credit_set"][row]):
            loss = -logp[w > 0].logsumexp(-1)
        else:
            loss = -(w * logp).sum()
        total += weight * float(loss)
        weight_sum += weight
    return total / weight_sum


def same_state(a, b):
    return all(torch.equal(a[k], b[k], ) or (torch.isnan(a[k]) == torch.isnan(b[k])).all() and
               torch.equal(torch.nan_to_num(a[k]), torch.nan_to_num(b[k])) for k in a)


def launch(trainer, graph, until):
    trainer.stop.fill_(until)
    graph.replay()
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/prepared/train-v1.pt"))
    ap.add_argument("--work", required=True, help="folder for throwaway checkpoints")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--long", type=int, default=60)
    ap.add_argument("--resume", action="store_true", help="fresh-process resume check only")
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--autocast", action="store_true", help="run the whole forward in BF16 autocast")
    ap.add_argument("--fp32-loss-check", action="store_true",
                    help="loss check only, with the model in FP32 so rounding cannot hide a logic error")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoTokenizer

    device = require_gpu(blocking_sync=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device, torch.float32 if args.fp32_loss_check else torch.bfloat16)
    peft = PeftModel.from_pretrained(model, str(ROOT / "decision-4b"), is_trainable=True)
    data = torch.load(args.data)
    data["meta"]["file_sha256"] = json.loads(Path(args.data).with_suffix(".json").read_text())["file_sha256"]
    trainer = Trainer(model, peft, data, device, recompute=not args.no_recompute, autocast=args.autocast)
    assert all(p.is_cuda for p in peft.parameters()), "a model tensor is on the CPU"
    init = trainer.snapshot()
    n = args.steps

    if args.resume:
        trainer.warm_up()
        graph = trainer.record()
        trainer.load(work / "after-half.pt")
        print(f"resumed at step {int(trainer.t)} in a fresh process")
        launch(trainer, graph, n)
        expected = torch.load(work / "after-all.pt", map_location=device)["state"]
        print(f"6. resume: final state bit-identical to the uninterrupted run: {same_state(trainer.snapshot(), expected)}")
        return

    if args.fp32_loss_check:
        trainer.recompute = False
        pairs = []
        for t in range(8):
            b = int(trainer.bucket_of_step[t])
            batch = trainer.buckets[b]["batch"]
            with torch.no_grad():
                ours = float(trainer.loss_for(b, trainer.rows[t, :batch], trainer.weight[t, :batch]))
            pairs.append((reference_loss(peft, trainer, t), ours))
        worst = max(abs(a - b) / max(abs(a), 1e-9) for a, b in pairs)
        print(f"2b. FP32 loss check, steps 0-7 (independent per-case, batched padded gated): "
              f"{[(round(a, 5), round(b, 5)) for a, b in pairs]}; worst relative gap {worst:.1e}")
        return

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    cases = [json.loads(x) for x in open(data["meta"]["cases_file"])]
    print(f"1. data audit: {audit_data(data, tok, cases)} prepared rows decode back to their case, gate and target")

    gaps = []
    for t in range(4):
        b = int(trainer.bucket_of_step[t])
        batch = trainer.buckets[b]["batch"]
        ours = {}
        for mode in (False, True):
            trainer.autocast = mode
            with torch.no_grad():
                ours[mode] = float(trainer.loss_for(b, trainer.rows[t, :batch], trainer.weight[t, :batch]))
        trainer.autocast = args.autocast
        gaps.append((round(reference_loss(peft, trainer, t), 4), round(ours[False], 4), round(ours[True], 4)))
    print(f"2. loss check for steps 0-3, (independent per-case, batched stock numerics, batched autocast): {gaps}")

    trainer.warm_up()
    t0 = time.perf_counter()
    trainer.run_eager(n)
    torch.cuda.synchronize()
    eager_s = (time.perf_counter() - t0) / n
    eager = trainer.snapshot()
    trainer.restore(init)
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    graph = trainer.record()
    print(f"   recording the loop took {time.perf_counter() - t0:.1f} s for {len(trainer.buckets)} buckets")
    trainer.restore(init)
    launch(trainer, graph, n // 2)
    trainer.save(work / "after-half.pt")
    launch(trainer, graph, n)
    trainer.save(work / "after-all.pt")
    looped = trainer.snapshot()
    used = sorted({int(x) for x in trainer.bucket_of_step[:n]})
    print(f"3. loop check over {n} steps touching buckets {used}: bit-identical to eager: {same_state(eager, looped)}; "
          f"losses {[round(x, 3) for x in looped['loss_log'][:n].tolist()]}")

    trainer.restore(init)
    b = int(trainer.bucket_of_step[2])
    poisoned = trainer.rows[2, : trainer.buckets[b]["batch"]]
    keep = trainer.buckets[b]["target"].index_select(0, poisoned).clone()
    trainer.buckets[b]["target"].index_fill_(0, poisoned, float("nan"))
    launch(trainer, graph, 5)
    log = trainer.loss_log[:5]
    finite = all(bool(torch.isfinite(p).all()) for p in trainer.params + trainer.m + trainer.v)
    print(f"4. guard check: steps run {int(trainer.t)}, skipped {int(trainer.bad)}, step-2 loss is NaN {bool(torch.isnan(log[2]))}, "
          f"other losses finite {bool(torch.isfinite(log[[0, 1, 3, 4]]).all())}, weights and optimizer state finite {finite}")
    trainer.buckets[b]["target"].index_copy_(0, poisoned, keep)

    trainer.restore(init)
    torch.cuda.synchronize()
    with Meter() as meter:
        trainer.stop.fill_(args.long)
        t0 = time.perf_counter()
        graph.replay()
        launch_ms = 1000 * (time.perf_counter() - t0)
        torch.cuda.synchronize()
    executed = sum(trainer.buckets[int(b)]["length"] * trainer.buckets[int(b)]["batch"] for b in trainer.bucket_of_step[: args.long])
    per_step = meter.wall / args.long
    print("5. throughput: " + json.dumps({
        "steps": args.long, "launches": 1, "launch_ms": round(launch_ms, 1),
        "s_per_step": round(per_step, 3), "eager_s_per_step": round(eager_s, 3),
        "positions_per_s": round(executed / meter.wall),
        "cpu_percent_of_one_core": round(100 * meter.cpu / meter.wall, 2),
        "gpu_util_percent": round(meter.gpu_util, 1),
        "gpu_memory_used_mib": int(smi("memory.used")),
        "skipped_steps": int(trainer.bad),
        "estimated_hours_for_all_steps": round(per_step * trainer.steps / 3600, 2),
    }))


if __name__ == "__main__":
    main()
