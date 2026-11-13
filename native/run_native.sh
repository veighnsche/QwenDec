#!/usr/bin/env bash
# Native side, step 1: train the reader on the frozen model's states, let it produce requests, and run them
# through the accepted adapter. One run, no retries.  `run_native.sh check` runs the loop check instead.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
if [ "${1:-}" = check ]; then
    $py native/run_head.py --check
    exit
fi
$py native/run_head.py
$py native/judge.py panels
adapter="--adapter runs/words/adapter --mode words"
$py latent/evaluate.py $adapter --pointer span --canonical --cases data/native/noul-development-panel4-produced.jsonl --tag native-noul
$py latent/choice_evaluate.py $adapter --cases data/native/choice-development-produced.jsonl --tag native-choice
$py native/judge.py final
echo "NATIVE STEP 1 FINISHED $(date '+%H:%M:%S')"
