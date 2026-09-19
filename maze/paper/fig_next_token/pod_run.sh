#!/bin/bash
# Orchestrate the next-token-probability experiment on a RunPod pod.
#   POD=runpod-nt bash pod_run.sh setup     # code + seed-0 checkpoints + wandb creds + pip
#   POD=runpod-nt bash pod_run.sh train     # seeds 1,2 x {none,fixed,trainable}, 2 at a time (tmux)
#   POD=runpod-nt bash pod_run.sh status    # tail training logs
#   POD=runpod-nt bash pod_run.sh eval      # next_token_probs.py over the 9 models (tmux)
#   POD=runpod-nt bash pod_run.sh pull      # results -> maze/paper/fig_next_token/results/
# Run from the oracle-encodings repo root. Nothing runs on the laptop.
set -euo pipefail
POD="${POD:-runpod-nt}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STASH_TREE="${STASH_TREE:-/private/tmp/claude-501/-Users-kaushikreddy-Projects-oracle-encoding-project-oracle-encodings/6aebd3ee-0117-47b1-94b4-a672631b1e32/scratchpad/stashtree/maze_legacy}"
SEEDS="${SEEDS:-1 2}"
RUN_BASE="fast-baseline_jirpyish-d256-l12_dfs-g6-n2000000"
RUN_FIX="fast-inj-fixed-complete-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000"
RUN_TRN="fast-inj-trainable-complete-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000"

case "${1:-}" in
setup)
  COPYFILE_DISABLE=1 tar -C . --exclude .git --exclude examples --exclude notebooks --exclude tests --exclude __pycache__ -czf - maze-transformer \
    | ssh "$POD" 'mkdir -p /workspace && tar -C /workspace --no-same-owner --no-same-permissions -xzf -'
  COPYFILE_DISABLE=1 tar -C "$(dirname "$STASH_TREE")" --exclude __pycache__ --exclude figures --exclude hunch_results \
    --exclude variant_results --exclude counterfactual_results --exclude artifacts \
    --exclude 'final_models/dummy-*' --exclude 'final_models/teacher-*' --exclude 'final_models/window4-*' \
    -czf - maze_legacy | ssh "$POD" 'mkdir -p /workspace && tar -C /workspace --no-same-owner --no-same-permissions -xzf - && rm -rf /workspace/maze_injection && mv /workspace/maze_legacy /workspace/maze_injection'
  scp -q "$HERE/next_token_probs.py" "$POD":/workspace/maze_injection/
  grep -A2 "api.wandb.ai" ~/.netrc | awk '/password/{print $2}' | ssh "$POD" '
    read -r WKEY; printf "machine api.wandb.ai\n  login user\n  password %s\n" "$WKEY" > ~/.netrc; chmod 600 ~/.netrc'
  ssh "$POD" '
    set -e; cd /workspace/maze-transformer
    pip install -q --break-system-packages -e . 2>&1 | tail -2
    pip install -q --break-system-packages fire wandb zanj 2>&1 | tail -1
    python -c "import maze_transformer, maze_dataset, torch, transformer_lens; print(\"OK torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
    nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "pod setup complete"'
  ;;
train)
  ssh "$POD" "cat > /workspace/train_seeds.sh" <<EOS
#!/bin/bash
# datasets first (one process per seed, serial), then 6 trainings 2 at a time
cd /workspace/maze_injection
for s in $SEEDS; do
  python -u train_fast.py --inject none --seed \$s --base_path /workspace/runs/fast-s\$s --pretok_only 2>&1 | tee /workspace/runs/pretok-s\$s.log
done
jobs=""
for s in $SEEDS; do for m in none fixed trainable; do jobs="\$jobs \$s:\$m"; done; done
echo \$jobs | tr ' ' '\n' | xargs -P 2 -I{} bash -c '
  s=\${1%%:*}; m=\${1##*:}
  cd /workspace/maze_injection
  python -u train_fast.py --inject \$m --seed \$s --base_path /workspace/runs/fast-s\$s > /workspace/runs/train-s\$s-\$m.log 2>&1
  echo "finished s\$s \$m rc=\$?"' _ {}
echo ALL_TRAINING_DONE
EOS
  ssh "$POD" 'mkdir -p /workspace/runs && chmod +x /workspace/train_seeds.sh && tmux new-session -d -s train "bash /workspace/train_seeds.sh 2>&1 | tee /workspace/runs/train_seeds.log" && tmux ls'
  ;;
copy-pretok)  # copy-pretok <seed> <src-alias> <src-ip> <src-port>: pull the seed's pretok file from another pod (pod->pod, agent forwarding)
  SEED=$2; SRC=$3
  F=$(ssh "$SRC" "grep -l -- \"\" /dev/null; ls /workspace/runs/baseline/pretok-*.pt" | sed -n "$( [ "$SEED" = 1 ] && echo 1 || echo 2 )p")
  case "$SEED" in 1) F=/workspace/runs/baseline/pretok-dfs-g6-g6-n2.0M-a_dfs-h61792.pt;; 2) F=/workspace/runs/baseline/pretok-dfs-g6-g6-n2.0M-a_dfs-h99047.pt;; esac
  DIP=$(ssh -G "$POD" | awk '/^hostname /{print $2}'); DPORT=$(ssh -G "$POD" | awk '/^port /{print $2}')
  ssh "$POD" 'mkdir -p /workspace/runs/baseline /workspace/runs'
  ssh -A "$SRC" "scp -q -P $DPORT -o StrictHostKeyChecking=no $F root@$DIP:/workspace/runs/baseline/"
  ssh "$POD" "ls -la /workspace/runs/baseline/"
  ;;
run1)  # run1 <seed> <none|fixed|trainable>: one training run in tmux (pretok must already be on the pod)
  SEED=$2; MODE=$3
  ssh "$POD" "mkdir -p /workspace/runs && tmux new-session -d -s train \"cd /workspace/maze_injection && python -u train_fast.py --inject $MODE --seed $SEED --base_path /workspace/runs/fast-s$SEED > /workspace/runs/train-s$SEED-$MODE.log 2>&1; echo finished s$SEED $MODE >> /workspace/runs/train_seeds.log; echo ALL_TRAINING_DONE >> /workspace/runs/train_seeds.log\" && tmux ls"
  ;;
status)
  ssh "$POD" 'tail -n 3 /workspace/runs/train_seeds.log 2>/dev/null; for f in /workspace/runs/train-s*.log; do echo "== $f"; grep -E "iter [0-9]+/" "$f" | tail -1; grep -E "done|Error|error" "$f" | tail -1; done; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
  ;;
eval)
  MODELS='{"baseline-s0": "final_models/baseline", "fixed-s0": "final_models/complete-fixed", "trainable-s0": "final_models/complete-trainable-retrain"'
  for s in $SEEDS; do
    MODELS="$MODELS, \"baseline-s$s\": \"/workspace/runs/fast-s$s/$RUN_BASE\", \"fixed-s$s\": \"/workspace/runs/fast-s$s/$RUN_FIX\", \"trainable-s$s\": \"/workspace/runs/fast-s$s/$RUN_TRN\""
  done
  MODELS="$MODELS}"
  echo "$MODELS" | ssh "$POD" 'cat > /workspace/runs/next_token_models.json'
  ssh "$POD" "cd /workspace/maze_injection && tmux new-session -d -s eval \"python -u next_token_probs.py --models /workspace/runs/next_token_models.json --out /workspace/runs/next_token --n_mazes ${N_MAZES:-2000} 2>&1 | tee /workspace/runs/next_token.log\" && tmux ls"
  ;;
pull)
  mkdir -p "$HERE/results"
  rsync -avz --exclude 'pretok-*' "$POD":/workspace/runs/next_token/ "$HERE/results/"
  rsync -avz "$POD":/workspace/runs/next_token.log "$HERE/results/"
  ;;
*)
  echo "usage: POD=runpod-nt bash $0 setup|train|status|eval|pull"; exit 1;;
esac
