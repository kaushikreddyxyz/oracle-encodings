#!/bin/bash
# LOCAL: wire up a freshly created pod for this experiment and launch a jobs file on it.
#   bash maze/experiments/08_seed_and_fade/scripts/setup_pod.sh <alias> <ip> <port> <pod_id> <jobs file> <grid> [guard_seconds]
set -euo pipefail
NAME=$1; IP=$2; PORT=$3; PODID=$4; JOBS=$5; GRID=$6; GUARD=${7:-9000}
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"; EXP="$ROOT/maze/experiments/08_seed_and_fade"
grep -q "^Host runpod-$NAME\$" ~/.ssh/config || ~/.claude/skills/runpod-spinup/register-ssh-alias.sh "$NAME" "$IP" "$PORT" >/dev/null
H=runpod-$NAME
until ssh -o ConnectTimeout=5 $H 'echo ready' 2>/dev/null | grep -q ready; do sleep 5; done
echo "[$NAME] ssh up: $(ssh $H 'nvidia-smi -L | wc -l') GPUs"
ssh $H 'mkdir -p /workspace/maze/experiments /workspace/distsets /workspace/runs /workspace/data'
rsync -az --exclude .git --exclude examples --exclude notebooks --exclude tests "$ROOT/maze-transformer" $H:/workspace/ 2>&1 | grep -v chown || true
rsync -az --exclude __pycache__ "$ROOT/maze/__init__.py" "$ROOT/maze/_pathsetup.py" "$ROOT/maze/core" $H:/workspace/maze/ 2>&1 | grep -v chown || true
rsync -az --exclude __pycache__ --exclude results --exclude 'fig_*' "$EXP" $H:/workspace/maze/experiments/ 2>&1 | grep -v chown || true
rsync -az "$ROOT/maze/paper/curves/results/distance/sets/g$GRID.npz" $H:/workspace/distsets/ 2>&1 | grep -v chown || true
rsync -az "$EXP/$JOBS" $H:/workspace/jobs.txt 2>&1 | grep -v chown || true
grep -A2 "api.wandb.ai" ~/.netrc | awk '/password/{print $2}' | ssh $H 'read -r K; printf "machine api.wandb.ai\n  login user\n  password %s\n" "$K" > ~/.netrc; chmod 600 ~/.netrc'
# pod-side auto-delete guard: needs the RunPod API key (the 2026-09-22 guard silently failed without it)
RPKEY=$(python3 - <<'P'
import os,re,pathlib
for f in [pathlib.Path.home()/".runpod/config.toml", pathlib.Path.home()/".claude.json"]:
    if f.exists():
        m=re.search(r'RUNPOD_API_KEY"?\s*[:=]\s*"?([A-Za-z0-9_\-]{20,})', f.read_text()) or re.search(r'apikey\s*=\s*"([^"]+)"', f.read_text(), re.I)
        if m: print(m.group(1)); break
P
)
if [ -n "$RPKEY" ]; then
  printf '%s\n' "$RPKEY" | ssh $H "read -r K; runpodctl config --apiKey \"\$K\" >/dev/null 2>&1; printf '#!/bin/bash\nsleep $GUARD\nrunpodctl remove pod $PODID\n' > /workspace/guard.sh; nohup bash /workspace/guard.sh > /workspace/guard.log 2>&1 < /dev/null & echo guard armed"
else
  echo "[$NAME] WARNING: no RunPod API key found locally; no pod-side guard"
fi
ssh $H 'cd /workspace/maze-transformer && pip install -q --break-system-packages -e . 2>&1 | tail -1; python -c "import maze_transformer, maze_dataset, torch; print(\"deps ok torch\", torch.__version__)"; cd /workspace && python maze/experiments/08_seed_and_fade/scripts/test_fade_lib.py | tail -1'
ssh $H "cd /workspace && rm -rf /workspace/runs/claims && GRID=$GRID tmux new -d -s queue 'bash maze/experiments/08_seed_and_fade/scripts/pod_queue.sh /workspace/jobs.txt > /workspace/runs/queue.log 2>&1'"
echo "[$NAME] queue launched (grid $GRID, $(grep -cv '^\s*$' "$EXP/$JOBS") jobs)"
