#!/bin/bash
# Mac-side: bash pv2_launch.sh <pod-name> <ip> <port>  -> alias (unless SKIP_REGISTER=1), wait, setup, ship HF token, launch probe_v2.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; NAME=$1; IP=$2; PORT=$3
TAGS=$(python3 -c "import json; print(' '.join(json.load(open('$HERE/fleet_pv2.json'))['pods']['$NAME']['tags']))")
[ "${SKIP_REGISTER:-0}" = 1 ] || ~/.claude/skills/runpod-spinup/register-ssh-alias.sh "$NAME" "$IP" "$PORT" >/dev/null
bash "$HERE/wait_ssh.sh" "runpod-$NAME" >/dev/null || { echo "$NAME: no ssh"; exit 1; }
bash "$HERE/codes_pod.sh" setup "runpod-$NAME" 6 > "/tmp/setup-$NAME.log" 2>&1 || { echo "$NAME: setup failed: $(tail -n 2 /tmp/setup-$NAME.log)"; exit 1; }
ssh "runpod-$NAME" 'mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token' < ~/.cache/huggingface/token
# shellcheck disable=SC2086
echo "$NAME ($TAGS): $(ssh "runpod-$NAME" "mkdir -p /workspace/runs && tmux new-session -d -s worker \"bash /workspace/maze/paper/codes/probe_v2.sh kaushikreddyxyz/maze-graft-codes $TAGS 2>&1 | tee -a /workspace/runs/worker.log\" && tmux ls" | tail -1)"
