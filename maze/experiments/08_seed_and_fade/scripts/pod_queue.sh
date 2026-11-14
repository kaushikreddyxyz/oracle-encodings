#!/bin/bash
# POD-SIDE: run every job of a jobs file, one job per GPU at a time.
#   bash maze/experiments/08_seed_and_fade/scripts/pod_queue.sh /workspace/jobs.txt
# jobs file: one line per run = the train_fade.py flags, e.g.
#   --seed 0 --inject fixed --inject_layer 3 --fade bernoulli
# A line starting with "base " is a base model; the rest of the line is
#   <seed> <layers to score it at, comma separated>
# (scored with the same-seed fixed graft's directions as the negative control;
# those are a pure function of the seed, so no grafted run is needed for them).
# Per job: train -> trajectory eval (500 base mazes) -> full eval of the final
# model -> drop all checkpoints except the kept ones -> DONE marker.
# Datasets are built ONCE per seed (jobs wait for theirs), so every arm of a seed
# reads the same file. Every job line must carry --seed.
cd /workspace
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
JOBS="$1"
SCRIPTS=maze/experiments/08_seed_and_fade/scripts
RUNS=/workspace/runs/fade; LOGS=/workspace/runs/logs; ST=/workspace/runs/STATUS
GRID=${GRID:-6}
SET=/workspace/distsets/g$GRID.npz
EDIT_OFFSETS=${EDIT_OFFSETS:-"0 2 4"}   # final eval also at site+2, site+4 (capped at 11)
TRAJ="1400,2000,3000,4000,5000,6000,7000"
KEEP="$TRAJ 1500 4000"   # trajectory ckpts survive so evals can be re-run post hoc
mkdir -p $RUNS $LOGS /workspace/data
NGPU=$(nvidia-smi -L | wc -l)
say() { echo "[$(date +%H:%M:%S)] $*" >> $ST; }

say "queue start: $(grep -cv '^\s*$' $JOBS) jobs, $NGPU GPUs"
# datasets: built in the background, one seed after another, in order of first use
(
  for s in $(grep -v '^\s*$' $JOBS | sed -E 's/^base ([0-9]+).*/\1/; s/.*--seed ([0-9]+).*/\1/' | awk '!seen[$0]++'); do
    [ -f /workspace/runs/pretok-s$s.ok ] && continue
    [ -f /workspace/runs/pretok-s$s.building ] && continue   # built elsewhere (e.g. tmux ds0); jobs wait for the .ok
    say "dataset seed $s: building"
    nice -n 10 python -u $SCRIPTS/train_fade.py --seed $s --grid_n $GRID --pretok_only True > $LOGS/pretok-s$s.log 2>&1 \
      && touch /workspace/runs/pretok-s$s.ok && say "dataset seed $s: ready" || { say "DATASET FAILED seed $s"; touch /workspace/runs/pretok-s$s.failed; }
  done
) &
wait_dataset() {   # $1 = seed
  until [ -f /workspace/runs/pretok-s$1.ok ]; do [ -f /workspace/runs/pretok-s$1.failed ] && return 1; sleep 5; done
}

run_job() {   # $1 = gpu, $2 = job index, $3.. = job line
  local gpu=$1 idx=$2; shift 2
  local line="$*" log=$LOGS/job$idx evfail=0
  export CUDA_VISIBLE_DEVICES=$gpu
  if [ "${line%% *}" = "base" ]; then
    set -- $line; local seed=$2 layers=$3
    wait_dataset $seed || { say "job$idx skipped: no dataset"; return; }
    say "job$idx gpu$gpu train start: base seed $seed"
    python -u $SCRIPTS/train_fade.py --inject none --seed $seed --grid_n $GRID --base_path $RUNS/s$seed > $log.train.log 2>&1
    local RUN=$(grep -m1 '^RUN_DIR=' $log.train.log | cut -d= -f2-)
    [ -n "$RUN" ] && [ -f "$RUN/model.final.pt" ] || { say "TRAIN FAILED job$idx"; return; }
    say "job$idx train done"
    for L in $(echo $layers | tr ',' ' '); do
      python -u $SCRIPTS/eval_fade.py --run_dir "$RUN" --set_npz $SET --w_seed $seed --w_layer $L --iters=-1 \
        --conditions off,delta,delta_perm,ablate,ablate_rand --out_dir "$RUN/eval_fade_L$L" > $log.eval_L$L.log 2>&1 || { say "EVAL FAILED job$idx L$L"; evfail=1; }
    done
    # base reference curve: autopilot solve rate at the trajectory checkpoints
    python -u $SCRIPTS/eval_fade.py --run_dir "$RUN" --set_npz $SET --w_seed $seed --iters="$TRAJ" --n_base 500 \
      --conditions off --ret_layers "" --out_dir "$RUN/eval_fade_traj" > $log.eval_traj.log 2>&1 || { say "EVAL FAILED job$idx traj"; evfail=1; }
  else
    local seed=$(echo "$line" | grep -o -- '--seed [0-9]*' | awk '{print $2}')
    wait_dataset $seed || { say "job$idx skipped: no dataset"; return; }
    say "job$idx gpu$gpu train start: $line"
    python -u $SCRIPTS/train_fade.py $line --grid_n $GRID --base_path $RUNS/s$seed > $log.train.log 2>&1
    local RUN=$(grep -m1 '^RUN_DIR=' $log.train.log | cut -d= -f2-)
    [ -n "$RUN" ] && [ -f "$RUN/model.final.pt" ] || { say "TRAIN FAILED job$idx"; return; }
    say "job$idx train done: $(basename $RUN)"
    python -u $SCRIPTS/eval_fade.py --run_dir "$RUN" --set_npz $SET --iters="$TRAJ" --n_base 500 \
      --conditions off,on,delta --out_dir "$RUN/eval_fade_traj" > $log.eval_traj.log 2>&1 || { say "EVAL FAILED job$idx traj"; evfail=1; }
    local site=$(python3 -c "import json;print(json.load(open('$RUN/fade_config.json'))['layer'])")
    for off in $EDIT_OFFSETS; do
      local L=$((site + off)); [ $L -gt 11 ] && continue
      local od="$RUN/eval_fade" rl="all"; [ $off != 0 ] && { od="$RUN/eval_fade_edit_L$L"; rl=""; }
      python -u $SCRIPTS/eval_fade.py --run_dir "$RUN" --set_npz $SET --iters=-1 --edit_layer $L --ret_layers "$rl" \
        --out_dir "$od" > $log.eval_L$L.log 2>&1 || { say "EVAL FAILED job$idx final L$L"; evfail=1; }
    done
  fi
  mkdir -p "$RUN/kept"
  for k in $(echo $KEEP | tr ',' ' '); do mv "$RUN/checkpoints/"*".iter_$k.pt" "$RUN/kept/" 2>/dev/null; done
  if [ "$evfail" = 0 ]; then rm -rf "$RUN/checkpoints"; else say "job$idx: eval failed, checkpoints kept"; fi
  rm -rf "$RUN/wandb"
  touch "$RUN/DONE"; say "job$idx done: $(basename $RUN)"
}

# one worker per GPU; workers claim job indices through a lock directory
# (claims/ persists: `rm -rf /workspace/runs/claims` before re-running a partial queue)
mapfile -t LINES < <(grep -v '^\s*$' $JOBS)
mkdir -p /workspace/runs/claims
for gpu in $(seq 0 $((NGPU - 1))); do
  (
    for idx in "${!LINES[@]}"; do
      mkdir /workspace/runs/claims/$idx 2>/dev/null || continue
      run_job $gpu $idx ${LINES[$idx]}
    done
  ) &
done
wait
say "QUEUE_DONE"
