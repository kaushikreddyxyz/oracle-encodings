#!/usr/bin/env bash
# Parallel Phase-A driver: run PAR concepts concurrently (rolling pool via xargs -P),
# each with per-process OpenRouter concurrency CONC. Total concurrency ~ PAR*CONC
# (keep near the proven-safe ~96-128; the client backs off on 429). Resumable:
# run_concept.py skips a concept whose gates already passed.
#
#   PAR=5 CONC=24 bash run_all_parallel.sh     # ~120 total concurrency
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POS="${POS:-4000}" HELDOUT="${HELDOUT:-40}" CONC="${CONC:-24}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
PAR="${PAR:-5}"
# big concepts first so they start early (they take longest); mixed with mediums to fill the pool
CONCEPTS="month color_hue weekday moon_phase season compass costliness size europe america africa indoors outdoors lovingness harmfulness duration"
echo "$(date -u +%H:%M:%S) Phase A PARALLEL start: PAR=$PAR CONC=$CONC POS=$POS" | tee -a "$HERE/../logs/phaseA_parallel.log"
printf '%s\n' $CONCEPTS | xargs -P "$PAR" -I{} bash "$HERE/run_one_concept.sh" {}
echo "$(date -u +%H:%M:%S) Phase A PARALLEL complete" | tee -a "$HERE/../logs/phaseA_parallel.log"
