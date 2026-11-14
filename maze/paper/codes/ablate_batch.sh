#!/bin/bash
# POD-SIDE: probes + ablations for already-trained runs shipped under /workspace/runs/<tag>/<run>/ (model.final.pt, injector.final.pt, config_holder.json)
cd /workspace; export PYTHONUNBUFFERED=1; ST=/workspace/runs/STATUS; mkdir -p /workspace/evalsets
for RUN in /workspace/runs/g6-s0-*/fast-*/; do RUN=${RUN%/}; name=$(basename $(dirname $RUN))
  python -u maze/paper/codes/probe_walls.py --run_dir "$RUN" --grid_n 6 > /workspace/runs/probe-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] probe_walls done $name" >> $ST || echo "[$(date +%H:%M:%S)] PROBE FAILED $name" >> $ST
  python -u maze/paper/codes/eval_ablate.py --run_dir "$RUN" --grid_n 6 > /workspace/runs/ablate-$name.log 2>&1 && echo "[$(date +%H:%M:%S)] eval_ablate done $name" >> $ST || echo "[$(date +%H:%M:%S)] ABLATE FAILED $name" >> $ST
done
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
