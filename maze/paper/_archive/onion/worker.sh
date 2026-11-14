#!/bin/bash
# POD-SIDE worker for the onion-graft pilot: one (grid, seed) dataset, a list of jobs, P training slots.
#   bash /workspace/maze/paper/onion/worker.sh 6 0 3 "onion-checker:fixed:0 onion-checker:trainable:0 ..."
# job = <inject_code>:<fixed|trainable>:<dummy 0|1>. Same trainer flags as the curves campaign (worker1.sh).
# Evals are serialised with a lock (they share the eval-set cache and saturate the GPU); training runs P at a time.
cd /workspace
export PYTHONUNBUFFERED=1
g=$1; s=$2; P=$3; JOBS="$4"
ST=/workspace/runs/STATUS; mkdir -p /workspace/runs /workspace/data /workspace/evalsets /workspace/dist_out /workspace/local_out
export TRAIN_FLAGS="--dataset_base_path /workspace/data --wandb_project maze-onion --padding_mode right --precision bf16 --save_zanj False --zanj_final True --ckpt_every 200 --ckpt_early 0,25,50,100,150,200,300,400,600,800 --loss_csv True"
echo "[$(date +%H:%M:%S)] worker start: g$g s$s slots=$P jobs=$JOBS" >> $ST
python -u maze/core/tests/test_onion.py > /workspace/runs/test_onion.log 2>&1 || { echo "[$(date +%H:%M:%S)] TEST_ONION FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
python -u maze/core/train.py --inject none --grid_n $g --seed $s --base_path /workspace/runs/pretok $TRAIN_FLAGS --pretok_only > /workspace/runs/pretok.log 2>&1 \
  && echo "[$(date +%H:%M:%S)] dataset ready" >> $ST || { echo "[$(date +%H:%M:%S)] DATASET FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
if echo "$JOBS" | grep -q ":1"; then
  python -u maze/core/train.py --inject fixed --dummy_data --grid_n $g --seed $s --base_path /workspace/runs/pretok $TRAIN_FLAGS --pretok_only > /workspace/runs/pretok-dummy.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] dummy dataset ready" >> $ST || echo "[$(date +%H:%M:%S)] DUMMY DATASET FAILED" >> $ST
fi
one() {
  job=$1; g=$2; s=$3; ST=/workspace/runs/STATUS
  code=${job%%:*}; rest=${job#*:}; m=${rest%%:*}; dummy=${rest#*:}
  name="$code-$m$( [ "$dummy" = 1 ] && echo -dummy )"; base=/workspace/runs/g$g-s$s-$code$( [ "$dummy" = 1 ] && echo -dummy )
  echo "[$(date +%H:%M:%S)] train start $name" >> $ST
  python -u maze/core/train.py --inject $m --inject_code $code $( [ "$dummy" = 1 ] && echo --dummy_data ) --grid_n $g --seed $s --base_path $base $TRAIN_FLAGS > /workspace/runs/train-$name.log 2>&1
  rc=$?
  RUN=$(ls -d $base/fast-inj-$m-*/ 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ ! -f "$RUN/model.final.pt" ]; then echo "[$(date +%H:%M:%S)] TRAIN FAILED $name rc=$rc" >> $ST; return; fi
  echo "[$(date +%H:%M:%S)] train done $name" >> $ST
  (
    flock 9
    python -u maze/paper/curves/eval_ckpts.py --run_dir "$RUN" --grid_n $g > /workspace/runs/eval-$name.log 2>&1
    [ -f "$RUN/eval/metrics.json" ] && echo "[$(date +%H:%M:%S)] eval_ckpts done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_CKPTS FAILED $name" >> $ST
    if [ "$dummy" != 1 ]; then   # the standard set and the one-swap set put maze A in the prompt: undefined for blanked prompts
      python -u maze/paper/curves/eval_distance.py run --run_dirs "${RUN%/}" --out_dir /workspace/dist_out > /workspace/runs/dist-$name.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] eval_distance done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_DISTANCE FAILED $name" >> $ST
      python -u maze/paper/curves/eval_local.py run --run_dir "${RUN%/}" --out_dir /workspace/local_out > /workspace/runs/local-$name.log 2>&1 \
        && echo "[$(date +%H:%M:%S)] eval_local done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_LOCAL FAILED $name" >> $ST
    fi
  ) 9> /workspace/eval.lock
  [ -f "$RUN/eval/metrics.json" ] && { rm -rf "$RUN/checkpoints/"model.iter_*.pt "$RUN/wandb"; touch "$RUN/DONE"; }
}
export -f one
echo $JOBS | tr ' ' '\n' | xargs -P $P -I{} bash -c "one {} $g $s"
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
