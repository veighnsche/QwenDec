#!/usr/bin/env bash
# Pilot 4: pilot 3's design, a new step order, judged on the renamed panel. One pass, no retries.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
run=runs/pilot4-canonical
$py train/run.py --data data/latent/prepared-canonical-span-k16-order4.pt --adapter runs/fork-v1/adapter --out $run
common="--adapter $run/adapter --mode latent --pointer span --canonical --vectors data/latent/question-vectors-k16-canonical.pt --projection $run/projection.pt"
panel="--cases data/latent/noul-development-panel4.jsonl"
$py latent/evaluate.py $common $panel --tag pilot4-canonical
$py latent/evaluate.py $common $panel --ablate swap-pointer --tag pilot4-canonical-swap-pointer
$py latent/evaluate.py $common $panel --ablate zero --tag pilot4-canonical-zero
$py latent/evaluate.py $common --tag pilot4-canonical-original-names
$py latent/evaluate.py $common --cases data/latent/noul-train-sample.jsonl --tag pilot4-canonical-trainnames
echo "PILOT 4 FINISHED $(date '+%H:%M:%S')"
