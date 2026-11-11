#!/usr/bin/env bash
# Joint Noul and Choice run: one adapter, one projection. One pass, no retries. Then the five measurements.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
run=runs/joint
$py train/run.py --data data/latent/prepared-joint.pt --adapter runs/fork-v1/adapter --out $run
noul="--adapter $run/adapter --mode latent --pointer span --canonical --vectors data/latent/question-vectors-k16-canonical.pt --projection $run/projection.pt"
panel="--cases data/latent/noul-development-panel4.jsonl"
$py latent/evaluate.py $noul $panel --tag joint-noul
$py latent/evaluate.py $noul $panel --ablate swap-pointer --tag joint-noul-swap-pointer
$py latent/evaluate.py $noul --tag joint-noul-original-names
$py latent/evaluate.py $noul --cases data/latent/noul-train-sample.jsonl --tag joint-noul-trainnames
$py latent/choice_evaluate.py --adapter $run/adapter --mode latent --projection $run/projection.pt --tag joint-choice
echo "JOINT RUN FINISHED $(date '+%H:%M:%S')"
