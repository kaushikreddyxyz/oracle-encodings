#!/bin/bash
# Mac-side control for the graft-code campaign (same layout as curves/curves_pod.sh; the code now ships from the repo's maze/core).
#   bash codes_pod.sh setup  <alias>
#   bash codes_pod.sh run    <alias> <grid> <seed> "<code>:<fixed|trainable>:<dummy>:<epochs>"
#   bash codes_pod.sh status <alias>
#   bash codes_pod.sh pull   <alias>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"; CUR="$REPO/maze/paper/curves"
ACT=$1; POD=$2; shift 2
X="--no-same-owner --no-same-permissions"
case "$ACT" in
setup)
  COPYFILE_DISABLE=1 tar -C "$REPO" --exclude .git --exclude examples --exclude notebooks --exclude tests --exclude __pycache__ -czf - maze-transformer \
    | ssh "$POD" "mkdir -p /workspace && tar -C /workspace $X -xzf -"
  COPYFILE_DISABLE=1 tar -C "$REPO" --exclude __pycache__ --exclude _ref -czf - maze/__init__.py maze/_pathsetup.py maze/core maze/paper/codes/worker.sh maze/paper/codes/worker_cont.sh maze/paper/codes/diag_collapse.py maze/paper/codes/probe_v2.sh maze/paper/codes/probe_pairs.py maze/paper/codes/pp_worker.sh maze/paper/codes/probe_walls.py maze/paper/codes/eval_ablate.py maze/paper/codes/probe_nonlinear.py maze/paper/codes/ablate_batch.sh maze/core/tests \
    maze/paper/curves/eval_ckpts.py maze/paper/curves/eval_distance.py maze/paper/curves/eval_local.py maze/paper/curves/maze_distance.py maze/paper/curves/sae_lib.py \
    | ssh "$POD" "tar -C /workspace $X -xzf -"
  G=${1:-6}   # grid size(s) whose standard / one-swap eval sets to ship, e.g. "6" or "6 8 10"
  for gg in $G; do ssh "$POD" "mkdir -p /workspace/distsets/g$gg-s4321-n3000 /workspace/localsets"
    scp -q "$CUR/results/distance/sets/g$gg.npz" "$POD:/workspace/distsets/g$gg-s4321-n3000/distset_K12_w0.05.npz"
    scp -q "$CUR/results/local/sets/g$gg.npz" "$POD:/workspace/localsets/g$gg.npz"; done
  grep -A2 "api.wandb.ai" ~/.netrc | awk '/password/{print $2}' | ssh "$POD" 'read -r WKEY; printf "machine api.wandb.ai\n  login user\n  password %s\n" "$WKEY" > ~/.netrc; chmod 600 ~/.netrc'
  ssh "$POD" 'set -e; cd /workspace/maze-transformer
    pip install -q --break-system-packages -e . 2>&1 | grep -v -i "warning\|notice" | tail -2
    pip install -q --break-system-packages fire wandb zanj 2>&1 | grep -v -i "warning\|notice" | tail -1
    cd /workspace && python -c "import maze._pathsetup, maze.core.train, maze.core.onion, torch; print(\"OK torch\", torch.__version__, torch.cuda.get_device_name(0))"
    python maze/core/tests/test_onion.py | tail -1
    command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
    echo "nproc $(nproc)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "pod setup complete"'
  ;;
run)   # bash codes_pod.sh run <alias> <grid> <seed> "<job>" ["<job2>" ...]   (jobs run sequentially in one tmux)
  g=$1; s=$2; shift 2; CMD=""; for j in "$@"; do CMD="$CMD bash /workspace/maze/paper/codes/worker.sh $g $s '$j';"; done
  ssh "$POD" "mkdir -p /workspace/runs && tmux new-session -d -s worker \"($CMD) 2>&1 | tee -a /workspace/runs/worker.log\" && tmux ls"
  ;;
status)
  ssh -o ConnectTimeout=20 "$POD" 'tail -n 8 /workspace/runs/STATUS 2>/dev/null; for f in /workspace/runs/train-*.log; do [ -f "$f" ] || continue; echo "$(basename $f): $(grep -oE "iter [0-9]+/[0-9]+ loss=[0-9.]+" $f | tail -1) $(grep -m1 -E "Traceback|Error" $f)"; done; for f in /workspace/runs/eval-*.log; do [ -f "$f" ] || continue; echo "$(basename $f): $(grep -c "^  iter" $f) rows $(grep -m1 -E "Traceback|Error" $f)"; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
  ;;
pull)
  mkdir -p "$HERE/results/runs" "$HERE/results/dist_out" "$HERE/results/local_out" "$HERE/results/logs"
  rsync -az --include='*/' --include='eval/***' --include='train_loss.csv' --include='config_holder.json' --include='model.final.pt' --include='injector.final.pt' --include='injector.iter_*.pt' --include='model.final.zanj' --include='DONE' --exclude='*' "$POD":/workspace/runs/ "$HERE/results/runs/" 2>&1 | grep -v "^$" | tail -2
  rsync -az --include='*/' --include='eval/***' --include='train_loss.csv' --include='config_holder.json' --include='model.final.pt' --include='injector.final.pt' --include='injector.iter_*.pt' --include='DONE' --exclude='*' "$POD":/workspace/runs/ "$HERE/results/runs/" >/dev/null 2>&1
  rsync -az "$POD":/workspace/dist_out/ "$HERE/results/dist_out/"; rsync -az "$POD":/workspace/local_out/ "$HERE/results/local_out/"
  rsync -az --include='*.log' --include='STATUS' --exclude='*' "$POD":/workspace/runs/ "$HERE/results/logs/"
  find "$HERE/results/runs" -name metrics.json | wc -l; ls "$HERE/results/dist_out" "$HERE/results/local_out"
  ;;
*) echo "usage: $0 setup|run|status|pull <alias> [args]"; exit 1;;
esac
