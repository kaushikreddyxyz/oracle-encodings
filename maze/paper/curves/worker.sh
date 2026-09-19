#!/bin/bash
# POD-SIDE worker: for each (grid:seed) triple, build the dataset, then train +
# score + prune baseline / fixed / trainable in sequence. Datasets for later
# triples are generated in the background while earlier triples train.
#   bash /workspace/maze/paper/curves/worker.sh 5:0 10:3
# Progress markers: /workspace/runs/STATUS (one line per event).
cd /workspace
export PYTHONUNBUFFERED=1
TRIPLES="$*"
ST=/workspace/runs/STATUS; mkdir -p /workspace/runs /workspace/data /workspace/evalsets
echo "[$(date +%H:%M:%S)] worker start: $TRIPLES" >> $ST
TRAIN_FLAGS="--dataset_base_path /workspace/data --wandb_project maze-curves --padding_mode right --precision bf16 --save_zanj False --zanj_final True --ckpt_every 200 --ckpt_early 0,25,50,100,150,200,300,400,600,800 --loss_csv True"

# background dataset builder (sequential, in triple order)
( for tr in $TRIPLES; do g=${tr%%:*}; s=${tr##*:}
    python -u maze/core/train.py --inject none --grid_n $g --seed $s --base_path /workspace/runs/g$g-s$s --pretok_only $TRAIN_FLAGS > /workspace/runs/pretok-g$g-s$s.log 2>&1 \
      && echo "[$(date +%H:%M:%S)] dataset ready g$g s$s" >> $ST || echo "[$(date +%H:%M:%S)] DATASET FAILED g$g s$s" >> $ST
    touch /workspace/runs/pretok-g$g-s$s.done
  done ) &

for tr in $TRIPLES; do g=${tr%%:*}; s=${tr##*:}
  until [ -f /workspace/runs/pretok-g$g-s$s.done ]; do sleep 20; done
  grep -q "DATASET FAILED g$g s$s" $ST && { echo "[$(date +%H:%M:%S)] skipping g$g s$s (dataset failed)" >> $ST; continue; }
  for m in none fixed trainable; do
    run_glob="/workspace/runs/g$g-s$s/fast-*$( [ $m = none ] && echo baseline || echo inj-$m )*"
    echo "[$(date +%H:%M:%S)] train start g$g s$s $m" >> $ST
    python -u maze/core/train.py --inject $m --grid_n $g --seed $s --base_path /workspace/runs/g$g-s$s $TRAIN_FLAGS > /workspace/runs/train-g$g-s$s-$m.log 2>&1
    rc=$?
    RUN=$(ls -d /workspace/runs/g$g-s$s/*/ | grep -E "$( [ $m = none ] && echo 'fast-baseline' || echo "fast-inj-$m-" )" | head -1)
    if [ $rc -ne 0 ] || [ ! -f "$RUN/model.final.pt" ]; then echo "[$(date +%H:%M:%S)] TRAIN FAILED g$g s$s $m rc=$rc" >> $ST; continue; fi
    echo "[$(date +%H:%M:%S)] train done g$g s$s $m" >> $ST
    python -u maze/paper/curves/eval_ckpts.py --run_dir "$RUN" --grid_n $g > /workspace/runs/eval-g$g-s$s-$m.log 2>&1
    if [ $? -ne 0 ] || [ ! -f "$RUN/eval/metrics.json" ]; then echo "[$(date +%H:%M:%S)] EVAL FAILED g$g s$s $m" >> $ST; continue; fi
    rm -rf "$RUN/checkpoints"            # intermediate checkpoints scored -> prune
    rm -rf "$RUN/wandb" 2>/dev/null
    echo "[$(date +%H:%M:%S)] eval done g$g s$s $m" >> $ST
    touch "$RUN/DONE"
  done
  rm -f /workspace/data/*g$g*seed*$s* 2>/dev/null   # (datasets are per-(g,s); keep it simple, delete nothing else)
done
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
