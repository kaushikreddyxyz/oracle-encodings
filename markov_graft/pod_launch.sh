#!/usr/bin/env bash
# (Re)launch the sweep on a pod: kills any running sweep/train/follow-up watcher, then starts
#   sweep.py "$@"   under MPS, and (if FOLLOWUPS=1) arms followups.sh to run when the sweep exits.
# Lives in a file so the kill patterns never appear on the ssh command line (pkill would hit the caller).
cd "$(dirname "$0")"
for pat in "[f]ollowups.sh" "[s]weep.py --sizes" "[m]arkov_graft/train.py"; do
  ps -eo pid,args | grep "$pat" | awk '{print $1}' | xargs -r kill
done
sleep 3
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# killed MPS clients can leave the server refusing new ones ("device busy or unavailable"): restart it every launch
echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d; sleep 2
mkdir -p logs results
nohup python sweep.py "$@" >> logs/sweep.log 2>&1 < /dev/null &
if [ "${FOLLOWUPS:-0}" = "1" ]; then
  nohup bash -c 'sleep 60; while pgrep -f "[s]weep.py --sizes" >/dev/null; do sleep 30; done; bash followups.sh' > logs/followups.log 2>&1 < /dev/null &
fi
sleep 2
echo "launched: sweep.py $*"; ps -eo pid,args | grep -c "[m]arkov_graft/train.py"
