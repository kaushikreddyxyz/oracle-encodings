#!/bin/bash
# Launch an injected training run + its eval follower on a satellite pod.
#   launch_satellite.sh <ssh-alias> <ip:port> <fixed|trainable>
# Run from the oracle-encodings repo root on the Mac. Copies the pretokenized
# dataset directly from pod runpod-maze (agent forwarding) so the probe set
# and training data are bit-identical across pods.
set -euo pipefail
ALIAS=$1; ADDR=$2; MODE=$3
IP=${ADDR%%:*}; PORT=${ADDR##*:}

POD="$ALIAS" bash maze/infra/pod_setup.sh

ssh "$ALIAS" 'mkdir -p /workspace/runs/baseline'
ssh -A runpod-maze "scp -P $PORT -o StrictHostKeyChecking=no /workspace/runs/baseline/pretok-*.pt root@$IP:/workspace/runs/baseline/"

RUN_NAME="fast-inj-$MODE-complete-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000"
ssh "$ALIAS" "
tmux new-session -d -s train \"cd /workspace && python -u maze/legacy/train_fast.py --inject $MODE --base_path /workspace/runs/fast 2>&1 | tee /workspace/runs/fast_inj.log\"
tmux new-session -d -s ev \"cd /workspace && python -u maze/experiments/01_complete_injection/scripts/eval_follower.py --run_dir /workspace/runs/fast/$RUN_NAME 2>&1 | tee /workspace/runs/ev.log\"
tmux ls"
