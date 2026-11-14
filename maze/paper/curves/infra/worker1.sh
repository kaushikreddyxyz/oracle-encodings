#!/bin/bash
# POD-SIDE single-job worker: one (grid, seed) dataset, a subset of conditions.
#   bash /workspace/maze/paper/curves/infra/worker1.sh 10:4:fixed,trainable
cd /workspace
export PYTHONUNBUFFERED=1
JOB="$1"; g=${JOB%%:*}; rest=${JOB#*:}; s=${rest%%:*}; conds=${rest#*:}; [ "$conds" = "$rest" ] && conds="none,fixed,trainable"
ST=/workspace/runs/STATUS; mkdir -p /workspace/runs /workspace/data /workspace/evalsets
echo "[$(date +%H:%M:%S)] worker1 start: g$g s$s conds=$conds" >> $ST
TRAIN_FLAGS="--dataset_base_path /workspace/data --wandb_project maze-curves --padding_mode right --precision bf16 --save_zanj False --zanj_final True --ckpt_every 200 --ckpt_early 0,25,50,100,150,200,300,400,600,800 --loss_csv True"
python -u maze/core/train.py --inject none --grid_n $g --seed $s --base_path /workspace/runs/g$g-s$s --pretok_only $TRAIN_FLAGS > /workspace/runs/pretok-g$g-s$s.log 2>&1 \
  && echo "[$(date +%H:%M:%S)] dataset ready g$g s$s" >> $ST || { echo "[$(date +%H:%M:%S)] DATASET FAILED g$g s$s" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
for m in $(echo $conds | tr ',' ' '); do
  echo "[$(date +%H:%M:%S)] train start g$g s$s $m" >> $ST
  python -u maze/core/train.py --inject $m --grid_n $g --seed $s --base_path /workspace/runs/g$g-s$s $TRAIN_FLAGS > /workspace/runs/train-g$g-s$s-$m.log 2>&1
  rc=$?
  RUN=$(ls -d /workspace/runs/g$g-s$s/*/ | grep -E "$( [ $m = none ] && echo 'fast-baseline' || echo "fast-inj-$m-" )" | head -1)
  if [ $rc -ne 0 ] || [ ! -f "$RUN/model.final.pt" ]; then echo "[$(date +%H:%M:%S)] TRAIN FAILED g$g s$s $m rc=$rc" >> $ST; continue; fi
  echo "[$(date +%H:%M:%S)] train done g$g s$s $m" >> $ST
  python -u maze/paper/curves/eval_ckpts.py --run_dir "$RUN" --grid_n $g > /workspace/runs/eval-g$g-s$s-$m.log 2>&1
  if [ $? -ne 0 ] || [ ! -f "$RUN/eval/metrics.json" ]; then echo "[$(date +%H:%M:%S)] EVAL FAILED g$g s$s $m" >> $ST; continue; fi
  rm -rf "$RUN/checkpoints" "$RUN/wandb"
  echo "[$(date +%H:%M:%S)] eval done g$g s$s $m" >> $ST
  touch "$RUN/DONE"
done
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
