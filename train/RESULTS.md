# Fork-mode training step: what was built and checked

10 November 2026. Qwen3.5-4B with the `decision-4b` LoRA, RTX 3090, PyTorch 2.15.0+cu132. No training run has been done. Every check below used a few dozen throwaway steps; no trained weights were kept.

## What a step does

1. Reads its own batch from a schedule that lives on the GPU.
2. Reads the state with the adapter off and the question and options with it on.
3. Takes the loss on the option-letter logits at the answer position.
4. Does one backward pass, clips, applies AdamW, and logs, all on the GPU.
5. Skips the update on the GPU if the loss or gradient norm is not finite, and counts it.

The loop runs on the GPU. The host launches it, sleeps, and wakes between launches to write a checkpoint and a log line.

## Prepared data (`data/prepared/train-v1.pt`)

| Item | Value |
| --- | --- |
| Cases | 3,830, each seen once per pass |
| Passes | 2 |
| Steps | 1,034 |
| Prompt length | 271 to 856 tokens, mean 476 |
| Length buckets | 10, from 320 to 896 positions |
| Padding | 7% of executed positions |
| Adapter on | 66.5% of positions |
| Right Choice alternative listed first, after shuffling | 561 of 1,126 |

## Checks

| Check | Result |
| --- | --- |
| Every prepared row decodes back to its case, switch point and target | 7,660 of 7,660 |
| Batched, padded, gated loss against an independent per-case loss, model in FP32 | worst relative gap 1.5e-5 over 8 steps |
| On-device loop against the same steps run eagerly, 12 steps across 6 buckets | bit-identical losses, weights and optimizer state |
| Poisoned step | skipped on the GPU, counted once, all state stayed finite |
| Checkpoint at step 6, continued in a fresh process to step 12 | bit-identical to the uninterrupted run |
| Run script: launch, continue, refuse changed settings, save adapter | all worked; saved adapter loads and evaluates |

## Throughput on the real data

| Numerics | Time per step | Positions per second | Host CPU (share of one core) | GPU utilisation | GPU memory | Two passes |
| --- | --- | --- | --- | --- | --- | --- |
| Stock (BF16 backbone, FP32 LoRA branch) | 4.94 s | 768 | 0.04% | 99.7% | 17.2 GB | about 1.4 h |
| Whole forward in BF16 autocast | 4.35 s | 867 | 0.04% | 99.8% | 20.7 GB | about 1.25 h |

Stock numerics is the default, so training and evaluation use the same arithmetic.

## Things to know

- **BF16 rounding is visible in the loss.** With the model in BF16, the same batch gives losses that differ by up to about 0.02 depending on batch shape. In FP32 the difference vanishes, so it is rounding, not a logic error.
- **The switch token is `.",`.** The state's final full stop shares a token with its closing quote, so that one character is read with the adapter on. The baseline measurements used the same switch point.
- **At these batch sizes eager PyTorch is no slower.** The step is GPU-bound either way; what the loop removes is the CPU core that eager execution pins at 100%.
- **No retention data yet.** The prepared data holds only the imported cases, none of the public sources the adapter was trained on.

## Commands

```bash
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python train/prepare.py
.venv/bin/python train/verify.py --work /path/to/throwaway
.venv/bin/python train/run.py --out runs/<name>
.venv/bin/python baseline/measure_fork.py --adapter runs/<name>/adapter --tag <name>
```
