#!/usr/bin/env bash
# Run once on every fresh RunPod container.
#
# Why: the container disk resets on Stop->Start, so /root/.gitconfig vanishes.
# With no git identity set, VS Code's GitHub integration fills the gap with
# whatever account it is signed into — which misattributes commits. Only
# /workspace persists, so identity must be re-seeded per container (globally)
# and stamped into each repo's own .git/config (which does persist).
set -euo pipefail

GIT_NAME="kaushikreddyxyz"
GIT_EMAIL="kaushikreddyxyz@gmail.com"

git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

# Route HTTPS GitHub remotes over SSH so pushes use the forwarded agent
# (the laptop's kaushikreddyxyz key) instead of any editor-injected token.
git config --global url."git@github.com:".insteadOf "https://github.com/"

# Stamp identity into every repo on the persistent volume. Repo-local config
# lives under /workspace, so it survives even if this script isn't re-run.
find /workspace -maxdepth 4 -name .git \( -type d -o -type f \) 2>/dev/null |
while read -r gitdir; do
    repo=$(dirname "$gitdir")
    git -C "$repo" config user.name "$GIT_NAME"
    git -C "$repo" config user.email "$GIT_EMAIL"
    echo "identity set: $repo"
done

echo "global identity: $(git config --global user.name) <$(git config --global user.email)>"
