#!/bin/bash
# Mac-side: bash fleet_launch.sh <pod-name> <ip> <port>   -> register alias, wait for sshd, setup (eval sets for the pod's grid), launch its jobs
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; NAME=$1; IP=$2; PORT=$3
read -r G S JOBS <<< "$(python3 -c "
import json; p=json.load(open('$HERE/fleet_ev.json'))['pods']['$NAME']; print(p['grid'], p['seed'], ' '.join(p['jobs']))")"
[ "${SKIP_REGISTER:-0}" = 1 ] || ~/.claude/skills/runpod-spinup/register-ssh-alias.sh "$NAME" "$IP" "$PORT" >/dev/null
bash "$HERE/wait_ssh.sh" "runpod-$NAME" >/dev/null || { echo "$NAME: no ssh"; exit 1; }
bash "$HERE/codes_pod.sh" setup "runpod-$NAME" "$G" > "/tmp/setup-$NAME.log" 2>&1 || { echo "$NAME: setup failed: $(tail -n 2 /tmp/setup-$NAME.log)"; exit 1; }
# shellcheck disable=SC2086
echo "$NAME (g$G s$S: $JOBS): $(bash "$HERE/codes_pod.sh" run "runpod-$NAME" "$G" "$S" $JOBS | tail -1)"
