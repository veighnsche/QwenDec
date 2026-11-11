# Training with no CPU in the loop: proof on the RTX 3090

10 November 2026. Qwen3.5-4B with the `decision-4b` LoRA made trainable (496 tensors, 16.2M parameters), PyTorch 2.15.0+cu132, Transformers 5.22.2.

This is a mechanical proof of throughput and host CPU. The tokens and targets are random, so nothing here is evidence of learning. No weights were saved.

## The method

1. **Everything a step needs is a GPU tensor.** Dataset, sampling order, learning-rate table, step counter, optimizer state and metric logs.
2. **One step is a pure tensor program.** Sample batch, forward, combined loss, one backward, clip, AdamW, log, advance the counter. No Python branch depends on a tensor value and nothing is read back.
3. **The step is recorded inside an on-device while loop.** PyTorch records the step as the body of a CUDA graph while-node whose condition is `step < stop`, computed on the GPU.
4. **The host launches once and sleeps.** It writes `stop`, calls `replay()` once, and blocks without spinning until the GPU has run every step.

## Measurements

One sequence of 512 positions per step, 16 steps, deterministic kernels.

| How the steps run | Time per step | Host CPU (share of one core) | GPU utilisation |
| --- | --- | --- | --- |
| Ordinary eager PyTorch | 733 ms | 99.5% | 56% |
| One recorded step, replayed by the host each step | 394 ms | 6.3% | 94% |
| Recorded on-device loop, one launch for all steps | 397 ms | 0.4% | 96% |

A 300-step run from a single 24 ms launch used 0.06% of one core with the GPU at 99.7%.

With deterministic kernels, all three modes produce bit-identical losses at every step and bit-identical final weights. Determinism costs about 8% in step time.

## Memory and sequence length

Recomputing each layer's activations during the backward pass (`--checkpoint`) is extra GPU work recorded in the same loop, so it adds no host work.

| Positions | Recompute | Time per step | GPU memory in use |
| --- | --- | --- | --- |
| 512 | off | 397 ms | 16.9 GB |
| 512 | on | 537 ms | 9.7 GB |
| 1,024 | off | out of memory | over 24 GB |
| 1,024 | on | 1,121 ms | 10.3 GB |
| 2,048 | on | 2,558 ms | 11.3 GB |

## Limits

- **Shapes are static.** Each (batch, length) pair needs its own recorded loop, and lengths must be multiples of 64. Padding and masks for real variable-length data are not built.
- **No random numbers inside the loop.** PyTorch rejects generator use in a while-node. The sampling order is precomputed; dropout would need precomputed masks.
- **The host still acts between launches.** Saving checkpoints, reading the metric logs and evaluation happen every K steps, outside the loop.
- **Do not queue replays ahead.** Queuing 12 per-step replays made the driver block and spin for 3.2 s of a 4.3 s run. The on-device loop avoids this.
- **The while-node capture calls are beta.** They are not in the public documentation; the usage came from `torch/_higher_order_ops/cudagraph_conditional_nodes.py`. Pin the PyTorch version.
- **This is plain full-sequence LoRA training.** The decision forks and adapter on/off switching within a step are not expressed yet.

## Reproduce

```bash
cd gpu_loop
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
../.venv/bin/python check_forward.py ../Qwen3.5-4B 512
../.venv/bin/python proof.py --model ../Qwen3.5-4B --adapter ../decision-4b --length 512 --steps 16 --deterministic --blocking-sync
../.venv/bin/python proof.py --model ../Qwen3.5-4B --adapter ../decision-4b --length 2048 --steps 12 --deterministic --blocking-sync --checkpoint --skip eager,graph
```
