#!/usr/bin/env bash
# Seed-1 replication of the graft-dropout follow-ups (runs once no sweep is active on this pod).
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while ps -eo args | grep -q "[s]weep.py --sizes"; do sleep 20; done
mkdir -p logs results
run() { python train.py "$@" > "logs/followup_$(echo "$*" | tr ' /' '__').log" 2>&1; }
run --k 3 --size m --arm oracle_3 --graft-dropout 0.5 --tag gd50 --seed 1 &
run --k 5 --size m --arm oracle_3 --graft-dropout 0.5 --tag gd50 --seed 1 &
run --k 5 --size m --arm oracle_3 --steps 24000 --tag steps24k --seed 1 &
run --k 5 --size m --arm leak_1 --graft-dropout 0.5 --tag gd50 --seed 1 &
wait
echo followups3-done
