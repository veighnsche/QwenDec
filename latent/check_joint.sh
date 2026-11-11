#!/usr/bin/env bash
# Joint run: the two checks that need the GPU, to be run before training. Throwaway weights, no learning claim.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
data=data/latent/prepared-joint.pt
# One step per length bucket, holding both operations where the bucket has both.
steps=$($py -c "import json; print(*json.load(open('data/latent/prepared-joint.json'))['check_steps'])")
$py latent/check.py --data $data --fp32 --at $steps
$py latent/check.py --data $data
