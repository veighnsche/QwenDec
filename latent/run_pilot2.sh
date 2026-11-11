#!/usr/bin/env bash
# Pilot 2: train both pointer arms, then measure each. One pass each, no retries.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
for arm in vector span; do
    run=runs/pilot2-pointer-$arm
    $py train/run.py --data data/latent/prepared-pointer-$arm-k16.pt --adapter runs/fork-v1/adapter --out $run
    common="--adapter $run/adapter --mode latent --pointer $arm --vectors data/latent/question-vectors-k16-pointer.pt --projection $run/projection.pt"
    $py latent/evaluate.py $common --tag pilot2-pointer-$arm
    $py latent/evaluate.py $common --ablate swap-pointer --tag pilot2-pointer-$arm-swap-pointer
    $py latent/evaluate.py $common --ablate zero --tag pilot2-pointer-$arm-zero
    $py latent/evaluate.py $common --cases data/latent/noul-train-sample.jsonl --tag pilot2-pointer-$arm-trainnames
    echo "ARM $arm FINISHED $(date '+%H:%M:%S')"
done
echo "PILOT 2 FINISHED $(date '+%H:%M:%S')"
