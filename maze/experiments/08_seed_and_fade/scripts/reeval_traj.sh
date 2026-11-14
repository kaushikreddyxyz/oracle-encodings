#!/bin/bash
# POD-SIDE: re-run the trajectory eval for every DONE run that lacks one (e.g. after
# the thread-oversubscription stall of 2026-09-22), round-robin over the GPUs.
#   bash maze/experiments/08_seed_and_fade/scripts/reeval_traj.sh
cd /workspace
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SCRIPTS=maze/experiments/08_seed_and_fade/scripts; SET=/workspace/distsets/g6.npz
TRAJ="1400,2000,3000,4000,5000,6000,7000"
NGPU=$(nvidia-smi -L | wc -l); i=0
for RUN in /workspace/runs/fade/s*/*/; do
  [ -f "$RUN/DONE" ] || continue
  [ -f "$RUN/eval_fade_traj/metrics.json" ] && continue
  ls "$RUN/kept/model.iter_1400.pt" >/dev/null 2>&1 || { echo "no kept ckpts: $RUN"; continue; }
  seed=$(basename $(dirname $RUN) | tr -d s)
  if [[ "$(basename $RUN)" == fade-baseline* ]]; then
    W="--w_seed $seed"; C="--conditions off --ret_layers \"\""
  else
    W=""; C="--conditions off,on,delta"
  fi
  echo "[$(date +%H:%M:%S)] reeval gpu$((i % NGPU)): $(basename $RUN)"
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) eval python -u $SCRIPTS/eval_fade.py --run_dir "$RUN" --set_npz $SET $W --iters="$TRAJ" --n_base 500 $C \
    --out_dir "$RUN/eval_fade_traj" > "$RUN/reeval_traj.log" 2>&1 &
  i=$((i + 1))
  [ $((i % NGPU)) = 0 ] && wait
done
wait
echo "[$(date +%H:%M:%S)] REEVAL_DONE"
