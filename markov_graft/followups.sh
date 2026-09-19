#!/usr/bin/env bash
# Follow-up cells run after the main grid (same pod, MPS env inherited). Tagged so analyze.py keeps them out of the grid figures.
#  1. optimisation-vs-capacity at the size-m cliff: 4x longer training, no graft, k=5 and k=6
#  2. graft dropout p=0.5: can reliance be turned off where the model can compute the feature (k=3) vs where it cannot (k=5)?
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs results
run() { python train.py "$@" > "logs/followup_$(echo "$*" | tr ' /' '__').log" 2>&1; }
run --k 5 --size m --arm none --steps 24000 --tag steps24k &
run --k 6 --size m --arm none --steps 24000 --tag steps24k &
run --k 3 --size m --arm oracle_3 --graft-dropout 0.5 --tag gd50 &
run --k 5 --size m --arm oracle_3 --graft-dropout 0.5 --tag gd50 &
wait
run --k 3 --size m --arm leak_1 --graft-dropout 0.5 --tag gd50 &
run --k 5 --size m --arm leak_1 --graft-dropout 0.5 --tag gd50 &
run --k 5 --size m --arm oracle_3 --steps 24000 --tag steps24k &
wait
echo followups-done
