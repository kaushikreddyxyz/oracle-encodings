#!/usr/bin/env bash
# Phase-A driver: build all 16 concept datasets at full scale, with §10 gates per concept.
# Resumable at concept grain (run_concept.py skips a concept whose gates already passed; --force to rebuild).
#
#   bash run_all.sh                 # full scale: 4000 positives/value, all 16 concepts
#   POS=100 bash run_all.sh         # smoke scale
#   CONCURRENCY=96 bash run_all.sh  # bump OpenRouter concurrency
#
# One concept at full scale:
#   .venv/bin/python3 code/run_concept.py --concept month --pos-per-value 4000 --heldout-per-value 40
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python3"

POS="${POS:-4000}"
HELDOUT="${HELDOUT:-40}"
CONCURRENCY="${CONCURRENCY:-64}"

# thread-cap hygiene (252-core host BLAS thrash; see handoff §5.1)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

CONCEPTS=(season compass weekday moon_phase month color_hue \
          costliness size europe america africa indoors outdoors lovingness harmfulness duration)

for c in "${CONCEPTS[@]}"; do
  echo "=== $(date -u +%H:%M:%S) building $c (pos/value=$POS) ==="
  # fault-tolerant: one concept failing must not abort the remaining 15 (resume re-tries it)
  "$PY" "$HERE/run_concept.py" --concept "$c" --pos-per-value "$POS" \
        --heldout-per-value "$HELDOUT" --concurrency "$CONCURRENCY" \
        || echo "!!! $(date -u +%H:%M:%S) FAILED $c (continuing; resume will retry)"
done
echo "=== Phase A complete: $ROOT/data/*.jsonl ==="
