"""Run fork-mode training on the RTX 3090 with the host out of the loop.

The GPU runs `--launch-steps` steps per launch by itself. Between launches the host
wakes, reads the on-GPU logs, writes a checkpoint and one log line, and launches
again. Settings are written to a manifest before the first step; a run folder can
only be continued with the same settings and data.

Finishing a run means the steps executed. Whether the adapter learned anything
useful is a separate question, answered by evaluation.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import load_text_model, require_gpu  # noqa: E402
from step import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/prepared/train-v1.pt"))
    ap.add_argument("--out", required=True, help="run folder: manifest, checkpoint, log, final adapter")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--launch-steps", type=int, default=50)
    ap.add_argument("--stop-after", type=int, default=None, help="stop early at this step (for tests)")
    ap.add_argument("--autocast", action="store_true")
    ap.add_argument("--adapter", default=str(ROOT / "decision-4b"), help="adapter to start from")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel

    device = require_gpu(blocking_sync=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    data = torch.load(args.data)
    data["meta"]["file_sha256"] = json.loads(Path(args.data).with_suffix(".json").read_text())["file_sha256"]
    if "vectors_file" in data:  # latent question arm
        data["vectors"] = torch.load(ROOT / data["vectors_file"])["vectors"]
    adapter = Path(args.adapter)
    manifest = {
        "start_adapter": str(adapter), "question_vectors": data.get("vectors_file"),
        "data": args.data, "data_sha256": data["meta"]["file_sha256"], "steps": data["meta"]["steps"],
        "passes": data["meta"]["passes"], "base_model": "Qwen3.5-4B",
        "start_adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "lr": args.lr, "warmup": args.warmup, "weight_decay": args.weight_decay, "clip": args.clip,
        "betas": [0.9, 0.999], "autocast": args.autocast, "recompute": True, "deterministic": True,
        "torch": torch.__version__,
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit(f"{manifest_path} was written with different settings; use a new run folder")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    peft = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    assert all(p.is_cuda for p in peft.parameters()), "a model tensor is on the CPU"
    trainer = Trainer(model, peft, data, device, lr=args.lr, warmup=args.warmup,
                      weight_decay=args.weight_decay, clip=args.clip, autocast=args.autocast)

    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        trainer.load(checkpoint)
        print(f"continuing from step {int(trainer.t)}", flush=True)
    trainer.warm_up()
    graph = trainer.record()

    target = min(args.stop_after or trainer.steps, trainer.steps)
    done = int(trainer.t)
    with open(out / "log.jsonl", "a") as log:
        while done < target and int(trainer.bad) < trainer.bad_limit:
            cpu, wall = time.process_time(), time.perf_counter()
            trainer.stop.fill_(min(done + args.launch_steps, target))
            graph.replay()
            torch.cuda.synchronize()  # the host sleeps here while the GPU runs the steps
            cpu, wall = time.process_time() - cpu, time.perf_counter() - wall

            # Everything below happens between launches.
            now = int(trainer.t)
            losses = trainer.loss_log[done:now]
            line = {"step": now, "mean_loss": round(float(losses.nanmean()), 4),
                    "mean_grad_norm": round(float(trainer.gnorm_log[done:now].nanmean()), 3),
                    "skipped_total": int(trainer.bad), "s_per_step": round(wall / max(1, now - done), 3),
                    "host_cpu_percent": round(100 * cpu / wall, 2)}
            log.write(json.dumps(line) + "\n")
            log.flush()
            print(json.dumps(line), flush=True)
            trainer.save(out / "checkpoint.tmp")
            os.replace(out / "checkpoint.tmp", checkpoint)
            done = now

    finished = done >= trainer.steps
    if finished:
        peft.save_pretrained(str(out / "adapter"))
        if trainer.vectors is not None:
            torch.save({"projection": trainer.projection.detach().cpu()}, out / "projection.pt")
    result = {"steps_executed": done, "steps_planned": trainer.steps, "skipped_steps": int(trainer.bad),
              "execution_complete": finished, "learning_accepted": None,
              "note": "learning_accepted stays empty until evaluation is run and judged against the agreed rule"}
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
