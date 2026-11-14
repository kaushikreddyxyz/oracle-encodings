#!/bin/bash
# Set up a fresh RunPod pod (alias $POD, default runpod-maze) for maze-transformer
# training. Run from the Mac, from the oracle-encodings repo root.
set -euo pipefail
POD="${POD:-runpod-maze}"

# code: local maze-transformer clone (our working copy) + launcher scripts.
# The maze/ tree lands at /workspace/maze; scripts locate the repo root by
# walking up to the dir containing maze/core (see maze/_pathsetup.py), so pod
# commands must be run from /workspace, e.g.
#   cd /workspace && python -u maze/legacy/train_fast.py ...
rsync -az --exclude .git --exclude examples --exclude notebooks --exclude tests \
    maze-transformer "$POD":/workspace/
rsync -az --exclude __pycache__ maze "$POD":/workspace/

# wandb credentials from the local netrc, piped over stdin (never in argv)
grep -A2 "api.wandb.ai" ~/.netrc | awk '/password/{print $2}' | ssh "$POD" '
    read -r WKEY
    mkdir -p ~/.config
    printf "machine api.wandb.ai\n  login user\n  password %s\n" "$WKEY" > ~/.netrc
    chmod 600 ~/.netrc
'

ssh "$POD" '
    set -e
    cd /workspace/maze-transformer
    pip install -q --break-system-packages -e . 2>&1 | tail -3
    python -c "import maze_transformer, maze_dataset, torch; print(\"maze_transformer OK, torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
    command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
    echo "pod setup complete"
'
