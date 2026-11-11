#!/usr/bin/env bash
# Choice request pilot: train the text arm and the latent arm, then measure each. One pass each, no retries.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
for arm in text latent; do
    run=runs/choice-$arm
    $py train/run.py --data data/latent/prepared-choice-$arm.pt --adapter runs/fork-v1/adapter --out $run
    extra=""; [ $arm = latent ] && extra="--projection $run/projection.pt"
    $py latent/choice_evaluate.py --adapter $run/adapter --mode $arm $extra --tag choice-$arm
    echo "ARM $arm FINISHED $(date '+%H:%M:%S')"
done
echo "CHOICE PILOT FINISHED $(date '+%H:%M:%S')"
