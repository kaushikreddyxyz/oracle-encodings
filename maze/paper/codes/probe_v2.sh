#!/bin/bash
# POD-SIDE: re-run probe_nonlinear.py (now with AUROC / balanced accuracy / per-class recall) on checkpoints pulled from HF.
#   bash probe_v2.sh <hf repo> <run folder>...     (run folder = g<g>-s<s>-<code>-e4-oe; downloads runs/<folder>/*/ from the repo)
cd /workspace; export PYTHONUNBUFFERED=1; REPO=$1; shift; ST=/workspace/runs/STATUS; mkdir -p /workspace/runs /workspace/probe_v2
echo "[$(date +%H:%M:%S)] probe_v2 start: $*" >> $ST
for tag in "$@"; do
  g=$(echo $tag | sed -E 's/^g([0-9]+)-.*/\1/')
  python - "$REPO" "$tag" <<'PY' > /workspace/runs/dl-$tag.log 2>&1 || { echo "[$(date +%H:%M:%S)] DOWNLOAD FAILED $tag" >> $ST; continue; }
import sys; from huggingface_hub import snapshot_download
repo, tag = sys.argv[1:]; snapshot_download(repo, allow_patterns=[f"runs/{tag}/*/model.final.pt", f"runs/{tag}/*/injector.final.pt", f"runs/{tag}/*/config_holder.json"], local_dir="/workspace/hf"); print("ok")
PY
  RUN=$(ls -d /workspace/hf/runs/$tag/*/ | head -1); RUN=${RUN%/}
  python -u maze/paper/codes/probe_nonlinear.py --run_dir "$RUN" --grid_n $g --out /workspace/probe_v2/$tag.json > /workspace/runs/pnl-$tag.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] probe_v2 done $tag" >> $ST || echo "[$(date +%H:%M:%S)] PROBE_V2 FAILED $tag" >> $ST
  rm -rf "/workspace/hf/runs/$tag"
done
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
