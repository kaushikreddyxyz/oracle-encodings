#!/bin/bash
# POD-SIDE worker for the graft-code campaign (vertex / degree / ego / edge / onion-*), one job per pod.
#   bash /workspace/maze/paper/codes/worker.sh <grid> <seed> <code>:<fixed|trainable>:<dummy 0|1>:<epochs>[:<ae|oe>]
# 5th field = injection start: ae = <ADJLIST_END> (default, legacy), oe = <ORIGIN_END> (graft codes; edge -> edge-oe)
# Same trainer flags as the curves campaign; checkpoints every 800 steps so the 1-epoch point (7812) is scored too.
# After training: eval_ckpts (true + phantom world), probe_walls (wall probes per layer); with an intact prompt also
# eval_distance (standard set) and, for static codes, eval_local (one-swap set; dynamic codes are skipped there).
cd /workspace; export PYTHONUNBUFFERED=1
g=$1; s=$2; job=$3
code=${job%%:*}; rest=${job#*:}; m=${rest%%:*}; rest=${rest#*:}; dummy=${rest%%:*}; rest=${rest#*:}; E=${rest%%:*}; st=${rest#*:}; [ "$st" = "$rest" ] && st=ae
START=$( [ "$st" = oe ] && echo origin_end || echo adjlist_end ); tcode=$( [ "$code" = edge ] && [ "$st" = oe ] && echo edge-oe || echo $code )
[ "$code" = base ] && { tcode=edge; m=none; }   # base:none:... = no graft (the trainer ignores the code when --inject none)
name="$code-$m$( [ "$dummy" = 1 ] && echo -dummy )-e$E-$st"; base=/workspace/runs/g$g-s$s-$code$( [ "$dummy" = 1 ] && echo -dummy )-e$E-$st
ST=/workspace/runs/STATUS; mkdir -p /workspace/runs /workspace/data /workspace/evalsets /workspace/dist_out /workspace/local_out
TRAIN_FLAGS="--dataset_base_path /workspace/data --wandb_project maze-codes --padding_mode right --precision bf16 --save_zanj False --zanj_final True --ckpt_every 800 --ckpt_early 0,100,200,400 --loss_csv True"
echo "[$(date +%H:%M:%S)] worker start: g$g s$s $job" >> $ST
for t in test_onion test_graft_start test_${code%1h}; do   # only this job's code test (the evaluator-equivalence test needs local checkpoints; run on the Mac)
  [ -f maze/core/tests/$t.py ] || continue
  python -u maze/core/tests/$t.py > /workspace/runs/$t.log 2>&1 || { echo "[$(date +%H:%M:%S)] $t FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
done
echo "[$(date +%H:%M:%S)] tests passed" >> $ST
python -u maze/core/train.py --inject none --grid_n $g --seed $s --base_path /workspace/runs/pretok $TRAIN_FLAGS --pretok_only > /workspace/runs/pretok.log 2>&1 || { echo "[$(date +%H:%M:%S)] DATASET FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
[ "$dummy" = 1 ] && { python -u maze/core/train.py --inject fixed --dummy_data --grid_n $g --seed $s --base_path /workspace/runs/pretok $TRAIN_FLAGS --pretok_only > /workspace/runs/pretok-dummy.log 2>&1 || { echo "[$(date +%H:%M:%S)] DUMMY DATASET FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }; }
echo "[$(date +%H:%M:%S)] dataset ready; train start $name" >> $ST
LOG=/workspace/runs/train-$name.log
python -u maze/core/train.py --inject $m --inject_code $tcode --inject_start $START $( [ "$dummy" = 1 ] && echo --dummy_data ) --grid_n $g --seed $s --base_path $base --epochs $E $TRAIN_FLAGS > $LOG 2>&1 &
pid=$!
# watchdog: wandb shutdown sometimes hangs train.py after "done" (final checkpoints already written) -> kill it and carry on
while kill -0 $pid 2>/dev/null; do
  RUN=$(ls -d $base/fast-inj-$m-*/ $base/fast-baseline*/ 2>/dev/null | head -1); RUN=${RUN%/}
  if [ -n "$RUN" ] && [ -f "$RUN/model.final.pt" ] && grep -q "INFO done" $LOG; then sleep 90; kill -0 $pid 2>/dev/null && { kill $pid; sleep 5; pkill -9 -f wandb-service; echo "[$(date +%H:%M:%S)] watchdog: killed hung trainer after done" >> $ST; }; break; fi
  sleep 30
done
wait $pid 2>/dev/null; rc=$?
RUN=$(ls -d $base/fast-inj-$m-*/ $base/fast-baseline*/ 2>/dev/null | head -1); RUN=${RUN%/}
if [ ! -f "$RUN/model.final.pt" ]; then echo "[$(date +%H:%M:%S)] TRAIN FAILED $name rc=$rc" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; fi
echo "[$(date +%H:%M:%S)] train done $name" >> $ST
python -u maze/paper/curves/eval_ckpts.py --run_dir "$RUN" --grid_n $g > /workspace/runs/eval-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_ckpts done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_CKPTS FAILED $name" >> $ST
python -u maze/paper/codes/probe_walls.py --run_dir "$RUN" --grid_n $g > /workspace/runs/probe-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] probe_walls done $name" >> $ST || echo "[$(date +%H:%M:%S)] PROBE FAILED $name" >> $ST
python -u maze/paper/codes/eval_ablate.py --run_dir "$RUN" --grid_n $g > /workspace/runs/ablate-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_ablate done $name" >> $ST || echo "[$(date +%H:%M:%S)] ABLATE FAILED $name" >> $ST
python -u maze/paper/codes/probe_nonlinear.py --run_dir "$RUN" --grid_n $g > /workspace/runs/pnl-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] probe_nonlinear done $name" >> $ST || echo "[$(date +%H:%M:%S)] PROBE_NONLINEAR FAILED $name" >> $ST
if [ "$dummy" != 1 ] && [ "$m" != none ]; then
  python -u maze/paper/curves/eval_distance.py run --run_dirs "$RUN" --out_dir /workspace/dist_out > /workspace/runs/dist-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_distance done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_DISTANCE FAILED $name" >> $ST
  if true; then   # eval_local drives any code (GraftDriver), ego included
    python -u maze/paper/curves/eval_local.py run --run_dir "$RUN" --out_dir /workspace/local_out > /workspace/runs/local-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_local done $name" >> $ST || echo "[$(date +%H:%M:%S)] EVAL_LOCAL FAILED $name" >> $ST
  fi
fi
[ -f "$RUN/eval/metrics.json" ] && { rm -rf "$RUN/checkpoints/"model.iter_*.pt "$RUN/wandb"; touch "$RUN/DONE"; }
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
