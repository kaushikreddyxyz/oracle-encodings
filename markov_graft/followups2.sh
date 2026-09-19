#!/usr/bin/env bash
# Second follow-up batch: plateau length vs table size (does 4.0M bits escape with 8x the budget?), leak with a longer budget,
# and seed-1 replication of the plateau escape at 501k bits.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs results
run() { python train.py "$@" > "logs/followup_$(echo "$*" | tr ' /' '__').log" 2>&1; }
run --k 6 --size m --arm none --steps 48000 --tag steps48k &
run --k 5 --size m --arm leak_1 --steps 24000 --tag steps24k &
run --k 6 --size m --arm oracle_3 --steps 24000 --tag steps24k &
run --k 5 --size m --arm none --steps 24000 --tag steps24k --seed 1 &
wait
echo followups2-done
