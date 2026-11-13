#!/usr/bin/env bash
# One adapter, three operations, requests as fixed word sequences. One pass, no retries. Then the six measurements.
# `run_words.sh check` runs the two GPU checks instead.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
data=data/latent/prepared-words.pt
if [ "${1:-}" = check ]; then
    # One step per prompt length, the one holding the most operations.
    steps=$($py -c "import json; print(*json.load(open('data/latent/prepared-words.json'))['check_steps'])")
    $py latent/check.py --data $data --fp32 --at $steps
    $py latent/check.py --data $data
    exit
fi
run=runs/words
$py train/run.py --data $data --adapter runs/fork-v1/adapter --out $run
noul="--adapter $run/adapter --mode words --pointer span --canonical"
panel="--cases data/latent/noul-development-panel4.jsonl"
$py latent/evaluate.py $noul $panel --tag words-noul
$py latent/evaluate.py $noul $panel --ablate swap-pointer --tag words-noul-swap-pointer
$py latent/evaluate.py $noul --tag words-noul-original-names
$py latent/evaluate.py $noul --cases data/latent/noul-train-sample.jsonl --tag words-noul-trainnames
$py latent/choice_evaluate.py --adapter $run/adapter --mode words --tag words-choice
$py latent/score_evaluate.py --adapter $run/adapter --mode words --tag words-score
echo "WORDS RUN FINISHED $(date '+%H:%M:%S')"
