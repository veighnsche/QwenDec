#!/usr/bin/env bash
# Score pilot follow-up: a text arm that reads the fixed wording without the rubric. One pass, no retries.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
data=data/latent/prepared-score-text-norubric.pt
if [ "${1:-}" = check ]; then
    $py latent/check.py --data $data --fp32
    $py latent/check.py --data $data
    exit
fi
run=runs/score-text-norubric
$py latent/score_evaluate.py --adapter runs/fork-v1/adapter --mode text --no-rubric --tag score-ref-forkv1-text-norubric
$py train/run.py --data $data --adapter runs/fork-v1/adapter --out $run
$py latent/score_evaluate.py --adapter $run/adapter --mode text --no-rubric --tag score-text-norubric
echo "NO-RUBRIC ARM FINISHED $(date '+%H:%M:%S')"
