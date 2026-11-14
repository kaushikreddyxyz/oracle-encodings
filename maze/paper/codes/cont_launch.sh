#!/bin/bash
# Mac-side: bash cont_launch.sh <cont-pod-name>  -> wait for sshd, setup (eval sets for the grid), upload the 4-epoch source
# checkpoint (model.final.pt + injector.final.pt + config) into the pod's /workspace/runs mirror, launch worker_cont.sh in tmux.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; NAME=$1
read -r G S CODE <<< "$(python3 -c "
import json; p=json.load(open('$HERE/fleet_cont.json'))['pods']['$NAME']; print(p['grid'], p['seed'], p['code'])")"
POD=runpod-$NAME
bash "$HERE/wait_ssh.sh" "$POD" >/dev/null || { echo "$NAME: no ssh"; exit 1; }
bash "$HERE/codes_pod.sh" setup "$POD" "$G" > "/tmp/setup-$NAME.log" 2>&1 || { echo "$NAME: setup failed: $(tail -n 2 /tmp/setup-$NAME.log)"; exit 1; }
SRC=$(ls -d "$HERE/results/runs/g$G-s$S-$CODE-e4-oe"/fast-inj-fixed-*/ | head -1); SRC=${SRC%/}; REL=${SRC#$HERE/results/runs/}
ssh "$POD" "mkdir -p /workspace/runs/$REL"
scp -q "$SRC/model.final.pt" "$SRC/injector.final.pt" "$SRC/config_holder.json" "$POD:/workspace/runs/$REL/" || { echo "$NAME: upload failed"; exit 1; }
ssh "$POD" "tmux new-session -d -s worker \"bash /workspace/maze/paper/codes/worker_cont.sh $G $S $CODE 2>&1 | tee -a /workspace/runs/worker.log\" && tmux ls" | tail -1 | sed "s/^/$NAME (g$G s$S $CODE): /"
