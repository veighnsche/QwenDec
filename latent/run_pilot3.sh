#!/usr/bin/env bash
# Pilot 3: train the canonical-request arm, then measure it. One pass, no retries.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
run=runs/pilot3-canonical
$py train/run.py --data data/latent/prepared-canonical-span-k16.pt --adapter runs/fork-v1/adapter --out $run
common="--adapter $run/adapter --mode latent --pointer span --canonical --vectors data/latent/question-vectors-k16-canonical.pt --projection $run/projection.pt"
$py latent/evaluate.py $common --tag pilot3-canonical
$py latent/evaluate.py $common --ablate swap-pointer --tag pilot3-canonical-swap-pointer
$py latent/evaluate.py $common --ablate zero --tag pilot3-canonical-zero
$py latent/evaluate.py $common --cases data/latent/noul-train-sample.jsonl --tag pilot3-canonical-trainnames
echo "PILOT 3 FINISHED $(date '+%H:%M:%S')"
