#!/usr/bin/env bash
# Wait until no sweep is running on this pod, then run the given sweep.py argument sets one after another.
# usage: tail_watcher.sh "--sizes l --seed-list 1 --tables 7:5" "--sizes l --seed-list 1 --tables 7:3"
cd "$(dirname "$0")"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while ps -eo args | grep -q "[s]weep.py --sizes"; do sleep 30; done
for args in "$@"; do
  echo "$(date -u +%H:%M) tail slice: $args"
  python sweep.py $args --parallel 4 --steps 6000 >> logs/sweep.log 2>&1
done
echo "$(date -u +%H:%M) tail slices done"
