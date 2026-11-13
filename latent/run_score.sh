#!/usr/bin/env bash
# Score request pilot: the no-training references, then the text arm and the latent arm. One pass each, no retries.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
$py latent/score_refs.py
$py latent/score_evaluate.py --adapter runs/fork-v1/adapter --mode text --tag score-ref-forkv1-text
echo "REFERENCES FINISHED $(date '+%H:%M:%S')"
for arm in text latent; do
    run=runs/score-$arm
    $py train/run.py --data data/latent/prepared-score-$arm.pt --adapter runs/fork-v1/adapter --out $run
    extra=""; [ $arm = latent ] && extra="--projection $run/projection.pt"
    $py latent/score_evaluate.py --adapter $run/adapter --mode $arm $extra --tag score-$arm
    echo "ARM $arm FINISHED $(date '+%H:%M:%S')"
done
echo "SCORE PILOT FINISHED $(date '+%H:%M:%S')"
