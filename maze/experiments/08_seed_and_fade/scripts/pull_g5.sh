#!/bin/bash
# LOCAL: pull evals, configs, loss curves, final models (not the kept checkpoints) from the pod.
#   bash maze/experiments/08_seed_and_fade/scripts/pull.sh [alias]
cd "$(dirname "$0")/.."
POD=${1:-runpod-fade-8x}
mkdir -p results_g5
rsync -az --ignore-existing --include='*/' --include='eval_fade*/**' --include='fade_config.json' --include='config_holder.json' \
  --include='train_loss.csv' --include='injector.final.pt' --include='model.final.pt' --include='DONE' --include='reeval_traj.log' \
  --exclude='*' "$POD":/workspace/runs/fade/ results_g5/ 2>&1 | grep -v chown
rsync -az "$POD":/workspace/runs/STATUS "$POD":/workspace/runs/queue.log "$POD":/workspace/runs/reeval.log results_g5/ 2>&1 | grep -v chown
rsync -az --ignore-existing "$POD":/workspace/runs/logs/ results_g5/logs/ 2>&1 | grep -v chown
echo "runs with final eval: $(ls results_g5/s*/*/eval_fade/metrics.json 2>/dev/null | wc -l); traj: $(ls results_g5/s*/*/eval_fade_traj/metrics.json 2>/dev/null | wc -l)"
