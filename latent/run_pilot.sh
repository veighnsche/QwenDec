#!/usr/bin/env bash
# Train the text arm, then the latent arm, then measure both. One pass each.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
k="${1:-4}"
py=.venv/bin/python
start=runs/fork-v1/adapter
if [ ! -f runs/pilot-text/result.json ] || ! grep -q '"execution_complete": true' runs/pilot-text/result.json; then
    $py train/run.py --data data/latent/prepared-text.pt --adapter $start --out runs/pilot-text
fi
$py train/run.py --data data/latent/prepared-latent-k$k.pt --adapter $start --out runs/pilot-latent-k$k
[ -f latent/results/pilot-text.json ] || $py latent/evaluate.py --adapter runs/pilot-text/adapter --mode text --tag pilot-text
latent="--adapter runs/pilot-latent-k$k/adapter --mode latent --vectors data/latent/question-vectors-k$k.pt --projection runs/pilot-latent-k$k/projection.pt"
$py latent/evaluate.py $latent --tag pilot-latent-k$k
$py latent/evaluate.py $latent --ablate zero --tag pilot-latent-k$k-zero
$py latent/evaluate.py $latent --ablate swap --tag pilot-latent-k$k-swap
echo "PILOT FINISHED $(date '+%H:%M:%S')"
