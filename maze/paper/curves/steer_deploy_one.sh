#!/bin/bash
# Mac-side: bash steer_deploy_one.sh <shard> <ssh-alias> [conds] [extra run args]   (code + pip + base models of the shard + the standard eval sets;
# with conds, launches prep+run for every model of the shard in tmux)
set -uo pipefail
SH=$1; POD=$2; CONDS=${3:-}; EXTRA=${4:-}
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
MAZETREE="${MAZETREE:?}"; SHARDS="${SHARDS:?}"
X="--no-same-owner --no-same-permissions"
n=0; until ssh -n -o ConnectTimeout=6 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$POD" 'echo ready' 2>/dev/null | grep -q ready; do sleep 5; n=$((n+1)); [ $n -gt 60 ] && { echo "$SH: no sshd"; exit 1; }; done
COPYFILE_DISABLE=1 tar -C "$REPO" --exclude .git --exclude examples --exclude notebooks --exclude tests --exclude __pycache__ -czf - maze-transformer | ssh "$POD" "mkdir -p /workspace && tar -C /workspace $X -xzf -"
COPYFILE_DISABLE=1 tar -C "$MAZETREE" --exclude __pycache__ -czf - maze | ssh "$POD" "tar -C /workspace $X -xzf -"
COPYFILE_DISABLE=1 tar -C "$REPO/maze/paper" -czf - curves/eval_ckpts.py curves/eval_distance.py curves/maze_distance.py curves/eval_steering.py | ssh "$POD" "mkdir -p /workspace/maze/paper && tar -C /workspace/maze/paper $X -xzf -"
ssh -n "$POD" 'cd /workspace/maze-transformer && nohup sh -c "pip install -q --break-system-packages -e . && pip install -q --break-system-packages fire zanj && touch /workspace/PIP_OK" > /workspace/pip.log 2>&1 &'
COPYFILE_DISABLE=1 tar -C "$REPO/maze/paper/curves/results" -cf - -T "$SHARDS/$SH.files" | ssh "$POD" "mkdir -p /workspace/runs && tar -C /workspace/runs $X -xf -"
for g in 5 6 7 8 9 10; do ssh -n "$POD" "mkdir -p /workspace/distsets/g$g-s4321-n3000"; scp -q "$HERE/results/distance/sets/g$g.npz" "$POD:/workspace/distsets/g$g-s4321-n3000/distset_K12_w0.05.npz"; done
n=0; until ssh -n -o ConnectTimeout=10 "$POD" 'test -f /workspace/PIP_OK' 2>/dev/null; do sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "$SH: pip timeout"; exit 1; }; done
ssh -n "$POD" "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)"
if [ -n "$CONDS" ]; then
  { echo '#!/bin/bash'; echo 'cd /workspace; export PYTHONUNBUFFERED=1'
    while read -r d; do echo "python -u maze/paper/curves/eval_steering.py run --run_dir $d --conds '$CONDS' $EXTRA 2>&1 | grep --line-buffered -v -i warn"; done < "$SHARDS/$SH.dirs"
    echo 'echo SHARD_DONE'; } | ssh "$POD" 'cat > /workspace/steer_job.sh'
  ssh -n "$POD" 'tmux new-session -d -s steer "bash /workspace/steer_job.sh 2>&1 | tee /workspace/steer.log"; tmux ls'
fi
echo "$SH: ready on $POD"
