#!/bin/bash
# Mac-side control for the training-curve campaign.
#   bash curves_pod.sh setup  <alias>            code + creds + pip
#   bash curves_pod.sh run    <alias> <triples>  start worker in tmux, e.g. "5:0 10:3"
#   bash curves_pod.sh status <alias>
#   bash curves_pod.sh pull   <alias>            eval results + loss csv + finals -> results/<alias>/
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
MAZETREE="${MAZETREE:-/private/tmp/claude-501/-Users-kaushikreddy-Projects-oracle-encoding-project-oracle-encodings/6aebd3ee-0117-47b1-94b4-a672631b1e32/scratchpad/mazetree}"
ACT=$1; POD=$2; shift 2
case "$ACT" in
setup)
  COPYFILE_DISABLE=1 tar -C "$REPO" --exclude .git --exclude examples --exclude notebooks --exclude tests --exclude __pycache__ -czf - maze-transformer \
    | ssh "$POD" 'mkdir -p /workspace && tar -C /workspace --no-same-owner --no-same-permissions -xzf -'
  COPYFILE_DISABLE=1 tar -C "$MAZETREE" --exclude __pycache__ -czf - maze | ssh "$POD" 'tar -C /workspace --no-same-owner --no-same-permissions -xzf -'
  COPYFILE_DISABLE=1 tar -C "$REPO/maze/paper" --exclude __pycache__ --exclude results -czf - curves | ssh "$POD" 'mkdir -p /workspace/maze/paper && tar -C /workspace/maze/paper --no-same-owner --no-same-permissions -xzf -'
  grep -A2 "api.wandb.ai" ~/.netrc | awk '/password/{print $2}' | ssh "$POD" 'read -r WKEY; printf "machine api.wandb.ai\n  login user\n  password %s\n" "$WKEY" > ~/.netrc; chmod 600 ~/.netrc'
  ssh "$POD" 'set -e; cd /workspace/maze-transformer
    pip install -q --break-system-packages -e . 2>&1 | grep -v -i "warning\|notice" | tail -2
    pip install -q --break-system-packages fire wandb zanj 2>&1 | grep -v -i "warning\|notice" | tail -1
    cd /workspace && python -c "import maze._pathsetup, maze.core.train, maze.core.model, torch; print(\"OK torch\", torch.__version__, torch.cuda.get_device_name(0))"
    command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
    echo "nproc $(nproc)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "pod setup complete"'
  ;;
run)
  ssh "$POD" "mkdir -p /workspace/runs && tmux new-session -d -s worker \"bash /workspace/maze/paper/curves/worker.sh $* 2>&1 | tee /workspace/runs/worker.log\" && tmux ls"
  ;;
run1)  # single job g:s:cond1,cond2
  ssh "$POD" "mkdir -p /workspace/runs && tmux new-session -d -s worker \"bash /workspace/maze/paper/curves/worker1.sh $1 2>&1 | tee /workspace/runs/worker.log\" && tmux ls"
  ;;
status)
  ssh -o ConnectTimeout=20 "$POD" 'tail -n 4 /workspace/runs/STATUS 2>/dev/null; for f in /workspace/runs/train-*.log; do [ -f "$f" ] || continue; echo "$(basename $f): $(grep -oE "iter [0-9]+/[0-9]+" $f | tail -1) $(grep -m1 -E "Traceback|Error" $f)"; done | tail -n 3; for f in /workspace/runs/eval-*.log; do [ -f "$f" ] || continue; echo "$(basename $f): $(grep -c "^  iter" $f) ckpts scored $(grep -m1 -E "Traceback|Error" $f)"; done | tail -n 2; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
  ;;
pull)
  mkdir -p "$HERE/results/$POD"
  rsync -az --include='*/' --include='eval/***' --include='train_loss.csv' --include='config_holder.json' --include='model.final.pt' --include='injector.final.pt' --include='model.final.zanj' --include='DONE' --exclude='*' "$POD":/workspace/runs/ "$HERE/results/$POD/" 2>&1 | grep -v "^$" | tail -2
  rsync -az "$POD":/workspace/runs/STATUS "$HERE/results/$POD/STATUS" 2>/dev/null
  find "$HERE/results/$POD" -name metrics.json | wc -l
  ;;
*) echo "usage: $0 setup|run|status|pull <alias> [args]"; exit 1;;
esac
