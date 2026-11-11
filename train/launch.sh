#!/usr/bin/env bash
# Run one named training run on the RTX 3090, then measure the result on the
# development cases. Safe to run again with the same name: training continues from
# its checkpoint. Does not retry by itself and does not judge the result.
set -euo pipefail

name="$1"
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run="runs/$name"
mkdir -p "$run"
exec >>"$run/launch.log" 2>&1

echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') starting $name"

# Do not compete with another job for the card.
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 1000 ]; then
    echo "the RTX 3090 already has ${used} MiB in use; not starting"
    exit 1
fi

.venv/bin/python train/run.py --out "$run"

if grep -q '"execution_complete": true' "$run/result.json"; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') training steps finished; measuring on the development cases"
    .venv/bin/python baseline/measure_fork.py --adapter "$run/adapter" --tag "$name"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') done"
else
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') training stopped before the last step; see $run/result.json"
    exit 1
fi
