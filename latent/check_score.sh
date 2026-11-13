#!/usr/bin/env bash
# Score request pilot: what has to happen on the GPU before training. Throwaway weights, no learning claim.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
$py latent/encode.py --k 16 --score
$py latent/score_vectors_check.py
for arm in text latent; do
    $py latent/check.py --data data/latent/prepared-score-$arm.pt --fp32
    $py latent/check.py --data data/latent/prepared-score-$arm.pt
done
