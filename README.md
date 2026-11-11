# QwenDec

**Attempt #2** · [Failed attempt #1: QwenOneTwo](https://github.com/veighnsche/QwenOneTwo)

Work in progress toward a language model that makes its own internal decisions without leaving its hidden state.

The target is one inference job in which [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) thinks and answers natively, while a decision LoRA ([evalengine/decision-4b](https://huggingface.co/evalengine/decision-4b)) is switched on only to evaluate three kinds of internal request on the live model state:

- **Noul**: the probability that a proposition is true
- **Choice**: which of several alternatives to pick
- **Score**: a value in [0, 1] against a rubric

Requests and results are meant to cross as vectors, never as generated text. The full requirements are in [docs/requirements.md](docs/requirements.md).

**Most of that target is not built.** What exists is listed below, with what each piece does and does not show.

## What exists

| Piece | Folder | What it shows |
| --- | --- | --- |
| Training with no CPU in the loop | [gpu_loop/](gpu_loop/) | A full training step runs as an on-device loop: one launch, the host sleeps |
| Baseline of the unchanged adapter | [baseline/](baseline/) | How decision-4b does on the project's cases, with the state read adapter-on and adapter-off |
| Fork-mode training | [train/](train/) | The adapter trained to decide from a natively read state; one run, `fork-v1` |
| Latent question pilot | [latent/](latent/) | A Noul question delivered as 4 or 16 vectors instead of text: did not pass (83% and 87% right, against 99% for text). The vectors carry the kind of question but do not bind a named item to the state. A second pilot added a pointer to the item's name in the state: it did not pass its rule either. With the pointed tokens replayed, binding works (72/72 where pilot 1 was at chance) but the arm is 11 questions short overall, nearly all on free-form questions, a wording training never contains; with the pointer pooled into one vector, binding fails on development's near-identical names ([results](latent/RESULTS2.md)). A third pilot with six fixed requests fixed the free-form questions (153/184) but did not pass: it drops the negation on single-prerequisite questions with development's names ([results](latent/RESULTS3.md)). A check with development's prerequisites renamed shows the pointer arms handle unseen names (464/464); what they miss is development's one pair of names that differ in a single word. A fourth pilot, a fresh run of the same design judged on distinct names, passed its rule: 1,162/1,182 against 1,174 for text, and it also got the near-identical pair right. Two runs of the design differ widely on cases unlike training, and six fixed requests are a narrower claim than vectors that carry a question ([results](latent/RESULTS4.md)). A Choice pilot in the same format (two fixed requests as vectors, one replayed pointer per alternative, no properties restated) passed its rule: 955/1,001 against 971 for text, with the answer holding when the alternatives change order (109/112 states, text 103). It is weaker than text at recognising that no alternative should be picked ([results](latent/RESULTS-CHOICE.md)) |

Not built: the native side (deciding when to call, forming a request, using the result), the allowance predictor, the runtime and its accounting, and Score over vectors. Choice over vectors exists only as a pilot.

## Training with no CPU in the loop

Ordinary PyTorch pins one CPU core at 100% while it launches every GPU kernel. Here the whole step is a tensor program recorded once as a CUDA graph while-node:

- The dataset, the batch order, the learning-rate table, the step counter, the optimizer state and the logs are GPU tensors.
- One step samples its batch, runs forward, loss, one backward, clipping and AdamW, and logs, with nothing read back to the host.
- The loop condition and a non-finite guard are evaluated on the GPU.
- The host launches once per 50 steps and blocks without spinning.

On an RTX 3090 with Qwen3.5-4B and the rank-8 LoRA (16.2M trainable parameters), a real 1,034-step run used 0.04% of one CPU core at 99.7% GPU utilisation. With deterministic kernels the loop is bit-identical to the same steps in ordinary PyTorch, and a checkpoint resumes to the bit-identical final state in a fresh process. Details: [gpu_loop/RESULTS.md](gpu_loop/RESULTS.md) and [train/RESULTS.md](train/RESULTS.md).

This relies on while-node and if-node capture in `torch.cuda.CUDAGraph`, which is beta and undocumented in PyTorch 2.15.

## Fork mode

The state is read with the adapter off and only the question and options with it on. In one causal forward pass that equals forking a natively encoded prefix, including the recurrent state of the 24 Gated DeltaNet layers. It is implemented as a per-position multiplier on every LoRA branch.

Development cases, state read adapter-off:

| Measure | Unchanged decision-4b | After `fork-v1` |
| --- | --- | --- |
| Noul: true or false | 120/130 | 120/130 |
| Choice: acceptable alternative picked | 114/114 | 114/114 |
| Noul: status and value both right | 194/243 | 228/243 |
| Choice: status and value both right | 165/232 | 225/232 |
| Score: status and value both right | 111/199 | 130/199 |
| "Unresolved" recognised | 2/81 | 69/81 |
| "Conflicting" recognised | 47/85 | 80/85 |
| Score within 0.05, interior targets | 15/67 | 3/67 |

`fork-v1` was accepted for Noul, Choice and status detection. Score is open: interior values got worse, and the training cases hold only 19 distinct Score values.

The latent request format of pilot 4 (fixed request kinds as vectors plus a replayed pointer) was accepted for Noul as the basis for the next stage. Reliability across runs is open, and Score requests are not built. A Choice request pilot in the same format passed its rule on one run and was accepted for Choice on the same terms ([results](latent/RESULTS-CHOICE.md)). Noul and Choice requests have not been trained in one adapter. A run that does so is designed and built but not yet run ([design](latent/JOINT.md)).

The ten Noul cases every model misses are mislabelled in the source: the stated facts contradict the label.

## Limits of the evidence

- **The cases are synthetic and templated.** They come from 288 families in six domains. Gains on the development cases may not carry to other wording.
- **Labels were computed by rule and reviewed by a model, not by people.**
- **General skill is unmeasured.** Training used none of the adapter's original public sources, so nothing here tests whether its broader ability eroded.
- **Small panels.** Most figures are uncertain by 5 to 9 points either way.
- **BF16 rounding.** The same batch gives losses that differ by up to about 0.02 depending on batch shape.
- **A locked evaluation set exists and has never been read.** It is not in this repository.

## Layout

```
gpu_loop/     lean forward for Qwen3.5, the on-device loop proof
train/        prepare.py, step.py, verify.py, run.py for fork-mode training
baseline/     measurements of the unchanged adapter, per-case outputs
latent/       question variants, fixed encoder, pilot design and pass rule
data/         the cases (JSONL); prepared tensors are regenerated, not stored
data_import/  how the cases were brought in
runs/         manifests, logs and results of training runs (no weights)
docs/         requirements
```

## Reproducing

Needs an NVIDIA GPU with 24 GB, Python 3.12, and the two model folders next to the code:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers peft safetensors psutil
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir Qwen3.5-4B
hf download evalengine/decision-4b --revision aadc55517ae8bfb1264fd5932ec452b939361e47 --local-dir decision-4b
```

The scripts select one specific GPU by UUID (`RTX3090_UUID` in `gpu_loop/model.py`) and refuse to run anywhere else, CPU included. Change that constant for your card.

```bash
export CUDA_VISIBLE_DEVICES=<your GPU UUID> CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python gpu_loop/proof.py --model Qwen3.5-4B --adapter decision-4b --deterministic --blocking-sync
.venv/bin/python baseline/measure_decision4b.py
.venv/bin/python baseline/measure_fork.py
.venv/bin/python train/prepare.py
.venv/bin/python train/verify.py --work /tmp/qwendec-verify
.venv/bin/python train/run.py --out runs/my-run
```

Model weights, the trained adapter and checkpoints are not in this repository.

## License

Apache-2.0, see [LICENSE](LICENSE). Qwen3.5-4B and decision-4b are separate works under their own licenses and are not redistributed here.
