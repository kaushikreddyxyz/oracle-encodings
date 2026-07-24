#!/usr/bin/env bash
# Add (or refresh) a ~/.ssh/config alias for a RunPod pod, so `ssh runpod-<name>`
# works directly. Run this after the `runpod` MCP server's pod-creation tool
# returns connection details — MCP has no access to this machine's filesystem,
# so this step can't be automated by the MCP server itself.
#
# Usage: ./register-ssh-alias.sh <name> <ip> <port>

set -euo pipefail

NAME="${1:?Usage: $0 <name> <ip> <port>}"
HOST="${2:?Missing ip}"
PORT="${3:?Missing port}"
ALIAS="runpod-${NAME}"

mkdir -p ~/.ssh && touch ~/.ssh/config

# Remove any stale block for this alias first (re-registration).
python3 - <<PY
import re, pathlib
p = pathlib.Path.home() / ".ssh" / "config"
txt = p.read_text() if p.exists() else ""
pattern = re.compile(r"(?ms)^Host " + re.escape("$ALIAS") + r"\s*\n(?:[ \t].*\n?)*")
txt = pattern.sub("", txt).rstrip() + "\n"
p.write_text(txt)
PY

# Matches the convention already in use for every pod on this machine.
# ~/.ssh/runpod is the real key in use (not any runpodctl-auto-generated key).
cat >> ~/.ssh/config <<CFG

Host $ALIAS
  HostName $HOST
  Port $PORT
  User root
  IdentityFile ~/.ssh/runpod
  AddKeysToAgent yes
  UseKeychain yes
  ForwardAgent yes
  StrictHostKeyChecking no
CFG

echo "Alias added: ssh $ALIAS"
