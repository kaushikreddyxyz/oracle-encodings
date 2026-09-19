#!/bin/bash
# Mac-side: bash distance_deploy_one.sh <shard> <ssh-alias>   (code + pip + that shard's final models, then launch in tmux)
set -uo pipefail
SH=$1; POD=$2
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
MAZETREE="${MAZETREE:?}"; SHARDS="${SHARDS:?}"
X="--no-same-owner --no-same-permissions"
n=0; until ssh -o ConnectTimeout=6 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$POD" 'echo ready' 2>/dev/null | grep -q ready; do sleep 5; n=$((n+1)); [ $n -gt 60 ] && { echo "$SH: no sshd"; exit 1; }; done
COPYFILE_DISABLE=1 tar -C "$REPO" --exclude .git --exclude examples --exclude notebooks --exclude tests --exclude __pycache__ -czf - maze-transformer | ssh "$POD" "mkdir -p /workspace && tar -C /workspace $X -xzf -"
COPYFILE_DISABLE=1 tar -C "$MAZETREE" --exclude __pycache__ -czf - maze | ssh "$POD" "tar -C /workspace $X -xzf -"
COPYFILE_DISABLE=1 tar -C "$REPO/maze/paper" --exclude __pycache__ -czf - curves/eval_ckpts.py curves/eval_distance.py curves/maze_distance.py | ssh "$POD" "mkdir -p /workspace/maze/paper && tar -C /workspace/maze/paper $X -xzf -"
ssh "$POD" 'cd /workspace/maze-transformer && nohup sh -c "pip install -q --break-system-packages -e . && pip install -q --break-system-packages fire zanj && touch /workspace/PIP_OK" > /workspace/pip.log 2>&1 &'
COPYFILE_DISABLE=1 tar -C "$REPO/maze/paper/curves/results" -cf - -T "$SHARDS/$SH.files" | ssh "$POD" "mkdir -p /workspace/runs && tar -C /workspace/runs $X -xf -"
n=0; until ssh -o ConnectTimeout=10 "$POD" 'test -f /workspace/PIP_OK' 2>/dev/null; do sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "$SH: pip timeout"; ssh "$POD" 'tail -3 /workspace/pip.log'; exit 1; }; done
DIRS=$(cat "$SHARDS/$SH.dirs")
ssh "$POD" "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null); cd /workspace && tmux new-session -d -s dist \"PYTHONUNBUFFERED=1 python -u maze/paper/curves/eval_distance.py run --run_dirs $DIRS 2>&1 | grep -v -i warn | tee /workspace/dist.log\"; tmux ls"
echo "$SH: launched on $POD"
