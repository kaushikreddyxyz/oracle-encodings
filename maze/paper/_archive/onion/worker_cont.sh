#!/bin/bash
# POD-SIDE continuation: warm-start the pod's finished run and train EPOCHS more passes over a FRESH dataset (seed DS),
# then run the same three evals. Usage: bash worker_cont.sh <grid> <seed> <ds_seed> <epochs> <job>   (job as in worker.sh)
cd /workspace; export PYTHONUNBUFFERED=1
g=$1; s=$2; ds=$3; E=$4; job=$5
code=${job%%:*}; rest=${job#*:}; m=${rest%%:*}; dummy=${rest#*:}
name="$code-$m$( [ "$dummy" = 1 ] && echo -dummy )"; base=/workspace/runs/g$g-s$s-$code$( [ "$dummy" = 1 ] && echo -dummy )
SRC=$(ls -d $base/fast-inj-$m-*/ | head -1); SRC=${SRC%/}; OUT=$base-cont1   # own dir => distinct seed_tag in every eval output
ST=/workspace/runs/STATUS
TRAIN_FLAGS="--dataset_base_path /workspace/data --wandb_project maze-onion --padding_mode right --precision bf16 --save_zanj False --zanj_final True --ckpt_every 400 --loss_csv True"
echo "[$(date +%H:%M:%S)] cont start $name from $SRC: $E epochs on dataset seed $ds" >> $ST
python -u maze/core/train.py --inject none --grid_n $g --seed $ds --base_path /workspace/runs/pretok $TRAIN_FLAGS --pretok_only > /workspace/runs/pretok-ds$ds.log 2>&1 || { echo "[$(date +%H:%M:%S)] CONT DATASET FAILED" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; }
[ "$dummy" = 1 ] && python -u maze/core/train.py --inject fixed --dummy_data --grid_n $g --seed $ds --base_path /workspace/runs/pretok $TRAIN_FLAGS --pretok_only > /workspace/runs/pretok-dummy-ds$ds.log 2>&1
# --seed $s keeps the run's own init seed (irrelevant after warm start) and the config name; the data seed is what changed
python -u maze/core/train.py --inject $m --inject_code $code $( [ "$dummy" = 1 ] && echo --dummy_data ) --grid_n $g --seed $ds --base_path $OUT --epochs $E --init_from $SRC --tag_suffix=-cont1 $TRAIN_FLAGS > /workspace/runs/train-$name-cont1.log 2>&1
rc=$?; RUN=$(ls -d $OUT/fast-inj-$m-*/ 2>/dev/null | head -1); RUN=${RUN%/}
if [ $rc -ne 0 ] || [ ! -f "$RUN/model.final.pt" ]; then echo "[$(date +%H:%M:%S)] CONT TRAIN FAILED $name rc=$rc" >> $ST; echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST; exit 1; fi
echo "[$(date +%H:%M:%S)] cont train done $name" >> $ST
python -u maze/paper/curves/eval_ckpts.py --run_dir "$RUN" --grid_n $g > /workspace/runs/eval-$name-cont1.log 2>&1 && echo "[$(date +%H:%M:%S)] cont eval_ckpts done $name" >> $ST || echo "[$(date +%H:%M:%S)] CONT EVAL_CKPTS FAILED $name" >> $ST
if [ "$dummy" != 1 ]; then
  python -u maze/paper/curves/eval_distance.py run --run_dirs "$RUN" --out_dir /workspace/dist_out > /workspace/runs/dist-$name-cont1.log 2>&1 && echo "[$(date +%H:%M:%S)] cont eval_distance done $name" >> $ST || echo "[$(date +%H:%M:%S)] CONT EVAL_DISTANCE FAILED $name" >> $ST
  python -u maze/paper/curves/eval_local.py run --run_dir "$RUN" --out_dir /workspace/local_out > /workspace/runs/local-$name-cont1.log 2>&1 && echo "[$(date +%H:%M:%S)] cont eval_local done $name" >> $ST || echo "[$(date +%H:%M:%S)] CONT EVAL_LOCAL FAILED $name" >> $ST
fi
[ -f "$RUN/eval/metrics.json" ] && { rm -rf "$RUN/checkpoints/"model.iter_*.pt "$RUN/wandb"; touch "$RUN/DONE"; }
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
