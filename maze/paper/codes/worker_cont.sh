#!/bin/bash
# POD-SIDE continuation for the edge-vs-ego scale-up: warm-start a finished 4-epoch run (weights only; AdamW moments restart,
# lr is constant so there is no schedule to resume) and train 4 more passes over the SAME 2M mazes => an 8-epoch model.
#   bash worker_cont.sh <grid> <seed> <edge|ego>      (source run must be at /workspace/runs/g<g>-s<s>-<code>-e4-oe/fast-inj-fixed-*/)
# Output: /workspace/runs/g<g>-s<s>-<code>-e8-oe/<run>-cont1/ with the same eval files as worker.sh.
cd /workspace; export PYTHONUNBUFFERED=1
g=$1; s=$2; code=$3; tcode=$( [ "$code" = edge ] && echo edge-oe || echo $code ); name="$code-fixed-e8-oe"
SRC=$(ls -d /workspace/runs/g$g-s$s-$code-e4-oe/fast-inj-fixed-*/ | head -1); SRC=${SRC%/}; OUT=/workspace/runs/g$g-s$s-$code-e8-oe
ST=/workspace/runs/STATUS; mkdir -p /workspace/runs /workspace/data /workspace/dist_out /workspace/local_out
TRAIN_FLAGS="--dataset_base_path /workspace/data --wandb_project maze-codes --padding_mode right --precision bf16 --save_zanj False --zanj_final True --ckpt_every 800 --loss_csv True"
echo "[$(date +%H:%M:%S)] cont worker start: g$g s$s $code from $SRC" >> $ST
[ -f "$SRC/model.final.pt" ] && [ -f "$SRC/injector.final.pt" ] || { echo "[$(date +%H:%M:%S)] CONT SOURCE MISSING" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
python -u maze/core/tests/test_graft_start.py > /workspace/runs/test_graft_start.log 2>&1 || { echo "[$(date +%H:%M:%S)] test FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
python -u maze/core/train.py --inject none --grid_n $g --seed $s --base_path /workspace/runs/pretok $TRAIN_FLAGS --pretok_only > /workspace/runs/pretok.log 2>&1 || { echo "[$(date +%H:%M:%S)] DATASET FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
echo "[$(date +%H:%M:%S)] dataset ready; cont train start $name" >> $ST
LOG=/workspace/runs/train-$name.log
python -u maze/core/train.py --inject fixed --inject_code $tcode --inject_start origin_end --grid_n $g --seed $s --base_path $OUT --epochs 4 --init_from "$SRC" --tag_suffix=-cont1 $TRAIN_FLAGS > $LOG 2>&1 &
pid=$!
# watchdog: wandb shutdown sometimes hangs train.py after "done" (final checkpoints already written) -> kill it and carry on
while kill -0 $pid 2>/dev/null; do
  RUN=$(ls -d $OUT/fast-inj-fixed-*/ 2>/dev/null | head -1); RUN=${RUN%/}
  if [ -n "$RUN" ] && [ -f "$RUN/model.final.pt" ] && grep -q "INFO done" $LOG; then sleep 90; kill -0 $pid 2>/dev/null && { kill $pid; sleep 5; pkill -9 -f wandb-service; echo "[$(date +%H:%M:%S)] watchdog: killed hung trainer after done" >> $ST; }; break; fi
  sleep 30
done
wait $pid 2>/dev/null
RUN=$(ls -d $OUT/fast-inj-fixed-*/ 2>/dev/null | head -1); RUN=${RUN%/}
[ -f "$RUN/model.final.pt" ] || { echo "[$(date +%H:%M:%S)] CONT TRAIN FAILED $name" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
echo "[$(date +%H:%M:%S)] cont train done $name" >> $ST
python -u maze/paper/curves/eval_ckpts.py --run_dir "$RUN" --grid_n $g > /workspace/runs/eval-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_ckpts done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_CKPTS FAILED $name" >> $ST
python -u maze/paper/codes/eval_ablate.py --run_dir "$RUN" --grid_n $g > /workspace/runs/ablate-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_ablate done $name" >> $ST || echo "[$(date +%H:%M:%S)] ABLATE FAILED $name" >> $ST
python -u maze/paper/codes/probe_nonlinear.py --run_dir "$RUN" --grid_n $g > /workspace/runs/pnl-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] probe_nonlinear done $name" >> $ST || echo "[$(date +%H:%M:%S)] PROBE_NONLINEAR FAILED $name" >> $ST
python -u maze/paper/curves/eval_distance.py run --run_dirs "$RUN" --out_dir /workspace/dist_out > /workspace/runs/dist-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_distance done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_DISTANCE FAILED $name" >> $ST
python -u maze/paper/curves/eval_local.py run --run_dir "$RUN" --out_dir /workspace/local_out > /workspace/runs/local-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_local done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_LOCAL FAILED $name" >> $ST
[ -f "$RUN/eval/metrics.json" ] && { rm -rf "$RUN/checkpoints/"model.iter_*.pt "$RUN/wandb"; touch "$RUN/DONE"; }
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
