"""Mechanical checks of the latent-question training step. Throwaway weights, no learning claim.

  --fp32   loss of the batched step against an independent per-case stock forward, model in
           FP32 and a deliberately non-identity projection, so a wiring error cannot hide
  default  the on-device loop against the same steps run eagerly, bit for bit
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(ROOT / "train"))
from model import load_text_model, require_gpu  # noqa: E402
from step import Trainer  # noqa: E402


@torch.no_grad()
def reference_loss(peft, model, trainer, t):
    b = int(trainer.bucket_of_step[t])
    d = trainer.buckets[b]
    total = count = 0.0
    for j in range(d["batch"]):
        if float(trainer.weight[t, j]) == 0:
            continue
        row = int(trainer.rows[t, j])
        n = int(d["answer_pos"][row]) + 1
        ids = d["tokens"][row, :n].long().unsqueeze(0)
        trainer.gate.mask = (torch.arange(n, device=ids.device) >= d["gate_start"][row]).view(1, n, 1)
        embeds = model.model.embed_tokens(ids)
        start, k = int(d["q_start"][row]), trainer.vectors.shape[1]
        q = trainer.vectors[int(d["q_index"][row])].float() @ trainer.projection.T
        embeds[0, start:start + k] = q.to(embeds.dtype)
        if "ptr_ids" in d:  # pointer as one vector, computed here case by case
            n_ptr = int(d["ptr_mask"][row].sum())
            embeds[0, start + k] = model.model.embed_tokens(d["ptr_ids"][row, :n_ptr].long()).float().mean(0) if n_ptr else 0.0
        logits = peft(inputs_embeds=embeds).logits[0, -1].float()
        logp = logits[trainer.letters[: int(d["valid"][row].sum())]].log_softmax(-1)
        total += float(-(d["target"][row, : logp.shape[0]] * logp).sum())
        count += 1
    return total / count


def same(a, b):
    return all(torch.equal(torch.nan_to_num(a[k]), torch.nan_to_num(b[k])) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/latent/prepared-latent-k4.pt"))
    ap.add_argument("--adapter", default=str(ROOT / "runs/fork-v1/adapter"))
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    from peft import PeftModel

    device = require_gpu(blocking_sync=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device, torch.float32 if args.fp32 else torch.bfloat16)
    peft = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    data = torch.load(args.data)
    data["meta"]["file_sha256"] = json.loads(Path(args.data).with_suffix(".json").read_text())["file_sha256"]
    data["vectors"] = torch.load(ROOT / data["vectors_file"])["vectors"]
    trainer = Trainer(model, peft, data, device, recompute=not args.fp32)
    print(f"trainable tensors {len(trainer.params)}, of which projection {tuple(trainer.projection.shape)}")

    if args.fp32:
        gen = torch.Generator(device=device).manual_seed(0)
        with torch.no_grad():
            trainer.projection.add_(0.05 * torch.randn(trainer.projection.shape, device=device, generator=gen))
        pairs = []
        for t in range(6):
            b = int(trainer.bucket_of_step[t])
            batch = trainer.buckets[b]["batch"]
            with torch.no_grad():
                ours = float(trainer.loss_for(b, trainer.rows[t, :batch], trainer.weight[t, :batch]))
            pairs.append((reference_loss(peft, model, trainer, t), ours))
        worst = max(abs(a - b) / max(abs(a), 1e-9) for a, b in pairs)
        print(f"FP32 loss check, steps 0-5 (independent per-case, batched step): "
              f"{[(round(a, 5), round(b, 5)) for a, b in pairs]}; worst relative gap {worst:.1e}")
        return

    init = trainer.snapshot()
    trainer.warm_up()
    trainer.run_eager(args.steps)
    torch.cuda.synchronize()
    eager = trainer.snapshot()
    moved = float((eager[f"p{len(trainer.params) - 1}"] - init[f"p{len(trainer.params) - 1}"]).abs().max())
    trainer.restore(init)
    torch.cuda.empty_cache()
    graph = trainer.record()
    trainer.restore(init)
    trainer.stop.fill_(args.steps)
    graph.replay()
    torch.cuda.synchronize()
    print(f"loop against eager over {args.steps} steps: bit-identical {same(eager, trainer.snapshot())}; "
          f"projection received updates {moved > 0} (largest change {moved:.2e}); "
          f"losses {[round(x, 3) for x in trainer.loss_log[: args.steps].tolist()]}")


if __name__ == "__main__":
    main()
