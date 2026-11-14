#!/bin/bash
# POD-SIDE: minimal-swap paired probes. bash pp_worker.sh <n_mazes> <tag>...   (tags g<g>-s<s>-<code>-e4-oe, any grids)
cd /workspace; export PYTHONUNBUFFERED=1; NM=$1; shift; ST=/workspace/runs/STATUS; mkdir -p /workspace/runs /workspace/pairs /workspace/pp_out /workspace/hf
REPO=kaushikreddyxyz/maze-graft-codes
dl() { python - "$REPO" "$1" <<'PY' > /workspace/runs/dl-$1.log 2>&1
import sys; from huggingface_hub import snapshot_download
r, t = sys.argv[1:]; snapshot_download(r, allow_patterns=[f"runs/{t}/*/model.final.pt", f"runs/{t}/*/injector.final.pt", f"runs/{t}/*/config_holder.json"], local_dir="/workspace/hf"); print("ok")
PY
}
echo "[$(date +%H:%M:%S)] pp start n=$NM: $*" >> $ST
for tag in "$@"; do
  g=$(echo $tag | sed -E 's/^g([0-9]+)-.*/\1/')
  dl $tag || { echo "[$(date +%H:%M:%S)] DOWNLOAD FAILED $tag" >> $ST; continue; }
  RUN=$(ls -d /workspace/hf/runs/$tag/*/ | head -1); RUN=${RUN%/}
  P=/workspace/pairs/g$g-n$NM.pt
  if [ ! -f $P ]; then
    python -u maze/paper/codes/probe_pairs.py build --grid_n $g --out $P --cfg_from "$RUN" --n_mazes $NM > /workspace/runs/build-g$g.log 2>&1 || { echo "[$(date +%H:%M:%S)] BUILD FAILED g$g" >> $ST; continue; }
    echo "[$(date +%H:%M:%S)] built g$g $(grep -o 'content-hash [0-9a-f]*' /workspace/runs/build-g$g.log)" >> $ST
    python -u maze/paper/codes/probe_pairs.py verify --pairs $P > /workspace/runs/verify-g$g.log 2>&1 && echo "[$(date +%H:%M:%S)] verified g$g $(tail -1 /workspace/runs/verify-g$g.log)" >> $ST || { echo "[$(date +%H:%M:%S)] VERIFY FAILED g$g $(tail -2 /workspace/runs/verify-g$g.log | head -1)" >> $ST; rm -f $P; continue; }
  fi
  python -u maze/paper/codes/probe_pairs.py run --run_dir "$RUN" --pairs $P --out /workspace/pp_out/$tag.json > /workspace/runs/pp-$tag.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] pp done $tag" >> $ST || echo "[$(date +%H:%M:%S)] PP FAILED $tag $(grep -m1 -E 'Error|assert' /workspace/runs/pp-$tag.log)" >> $ST
done
echo "[$(date +%H:%M:%S)] WORKER_DONE" >> $ST
