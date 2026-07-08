#!/usr/bin/env bash
# Delete a RunPod pod.
# Usage: ./cleanup-pod.sh <pod-id>
#
# Find pod IDs with: runpodctl pod list

set -euo pipefail

POD_ID="${1:?Usage: $0 <pod-id>}"

# Look up the pod name before deletion so we can strip its ~/.ssh/config entry.
POD_NAME=$(runpodctl pod get "$POD_ID" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('name',''))" 2>/dev/null || true)

echo "==> Deleting pod: $POD_ID"
runpodctl pod delete "$POD_ID" -o json
echo "    Pod deleted."

if [ -n "$POD_NAME" ] && [ -f ~/.ssh/config ]; then
  python3 - "$POD_NAME" <<'PY'
import re, sys, pathlib
name = sys.argv[1]
p = pathlib.Path.home() / ".ssh" / "config"
txt = p.read_text()
new = re.sub(r"(?ms)^# RunPod pod: " + re.escape(name) + r" .*?(?=^Host |\Z)", "", txt).rstrip() + "\n"
if new != txt:
    p.write_text(new)
    print(f"    Removed ~/.ssh/config alias: runpod-{name}")
PY
fi
