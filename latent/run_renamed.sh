#!/usr/bin/env bash
# Measure the already trained pointer arms on development states with renamed prerequisites. No training.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=.venv/bin/python
for version in nouns clauses; do
    cases="--cases data/latent/noul-development-renamed-$version.jsonl --mode latent"
    for arm in vector span; do
        run=runs/pilot2-pointer-$arm
        $py latent/evaluate.py $cases --adapter $run/adapter --pointer $arm --projection $run/projection.pt \
            --vectors data/latent/question-vectors-k16-pointer.pt --tag renamed-$version-pilot2-$arm
    done
    run=runs/pilot3-canonical
    $py latent/evaluate.py $cases --adapter $run/adapter --pointer span --canonical --projection $run/projection.pt \
        --vectors data/latent/question-vectors-k16-canonical.pt --tag renamed-$version-pilot3
done
echo "RENAMED FINISHED $(date '+%H:%M:%S')"
