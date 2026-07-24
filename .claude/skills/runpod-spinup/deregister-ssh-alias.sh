#!/usr/bin/env bash
# Remove a RunPod pod's ~/.ssh/config alias. Run this whenever you delete a
# pod via the `runpod` MCP server — otherwise the alias points at a dead IP
# that RunPod may reassign to someone else's pod later.
#
# Usage: ./deregister-ssh-alias.sh <name>

set -euo pipefail

NAME="${1:?Usage: $0 <name>}"
ALIAS="runpod-${NAME}"

[ -f ~/.ssh/config ] || exit 0

python3 - <<PY
import re, pathlib
p = pathlib.Path.home() / ".ssh" / "config"
txt = p.read_text()
new = re.sub(r"(?ms)^Host " + re.escape("$ALIAS") + r"\s*\n(?:[ \t].*\n?)*", "", txt).rstrip() + "\n"
if new != txt:
    p.write_text(new)
    print("Removed ~/.ssh/config alias: $ALIAS")
else:
    print("No alias found for $ALIAS (nothing to remove)")
PY
