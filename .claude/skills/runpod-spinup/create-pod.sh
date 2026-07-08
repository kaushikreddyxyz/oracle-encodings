#!/usr/bin/env bash
# Create a RunPod pod with sensible defaults.
# Usage: ./create-pod.sh <name> <gpu-id> [cloud-type] [template-id] [gpu-count]
#
# Examples:
#   ./create-pod.sh test "NVIDIA GeForce RTX 4090"
#   ./create-pod.sh h100 "NVIDIA H100 80GB HBM3" COMMUNITY
#   ./create-pod.sh a100x8 "NVIDIA A100-SXM4-80GB" SECURE runpod-torch-v21 8
#
# Defaults:
#   cloud-type:  SECURE
#   template-id: runpod-torch-v21  (PyTorch 2.1, CUDA 11.8)
#   gpu-count:   1
#
# What this does:
#   1. Calls `runpodctl pod create` with the given args
#   2. Prints pod id, GPU, hourly cost, and SSH command
#   3. Falls back from COMMUNITY -> SECURE on resource errors if cloud-type was COMMUNITY

set -euo pipefail

NAME="${1:?Usage: $0 <name> <gpu-id> [cloud-type] [template-id] [gpu-count] [container-disk-gb]}"
GPU_ID="${2:?Missing gpu-id (e.g. 'NVIDIA GeForce RTX 4090')}"
CLOUD_TYPE="${3:-SECURE}"
TEMPLATE_ID="${4:-runpod-torch-v21}"
GPU_COUNT="${5:-1}"
# Default 20GB matches runpodctl's default and is tiny — barely enough for
# a venv. Override when you need to download anything substantial: a single
# Llama-3.1-8B HF snapshot is ~16GB; 14B is ~28GB; 70B is ~140GB. The HF
# cache lives on the container disk unless you separately mount a volume,
# so size this for the largest model you intend to download.
CONTAINER_DISK_GB="${6:-20}"

create_pod() {
  local cloud="$1"
  runpodctl pod create \
    --name "$NAME" \
    --gpu-id "$GPU_ID" \
    --cloud-type "$cloud" \
    --template-id "$TEMPLATE_ID" \
    --gpu-count "$GPU_COUNT" \
    --container-disk-in-gb "$CONTAINER_DISK_GB" \
    -o json 2>&1
}

echo "==> Creating pod '$NAME' on $CLOUD_TYPE cloud: ${GPU_COUNT}× $GPU_ID (template: $TEMPLATE_ID)"
OUT=$(create_pod "$CLOUD_TYPE" || true)

if echo "$OUT" | grep -q 'does not have the resources'; then
  if [ "$CLOUD_TYPE" = "COMMUNITY" ]; then
    echo "    Community cloud exhausted; retrying on SECURE cloud..."
    OUT=$(create_pod "SECURE")
  else
    echo "$OUT" >&2
    exit 1
  fi
fi

# If still an error, fail loudly
if ! echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); sys.exit(0 if 'id' in d else 1)" 2>/dev/null; then
  echo "$OUT" >&2
  exit 1
fi

POD_ID=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
COST=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('costPerHr','?'))")
GPU_DISPLAY=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('machine',{}).get('gpuDisplayName','?'))")
LOC=$(echo "$OUT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('machine',{}).get('location','?'))")

echo ""
echo "==> Pod created!"
echo "    Pod ID:    $POD_ID"
echo "    GPU:       ${GPU_COUNT}× $GPU_DISPLAY ($LOC)"
echo "    Cost:      \$${COST}/hr"
echo ""
echo "    SSH:       runpodctl ssh connect $POD_ID"
echo "    Stop:      runpodctl pod stop $POD_ID    # storage still billed"
echo "    Delete:    runpodctl pod delete $POD_ID  # or ./cleanup-pod.sh $POD_ID"

# --- Append an ~/.ssh/config alias once the pod has an SSH endpoint ---
# Pattern: `ssh runpod-<name>` (matches `Host runpod-*` defaults block).
# Waits up to ~2min for runpodctl to expose IP/port, then appends the entry.
ALIAS="runpod-${NAME}"
for _ in $(seq 1 24); do
  SSH_CMD=$(runpodctl pod get "$POD_ID" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('ssh') or {}).get('ssh_command') or '')" 2>/dev/null || true)
  [ -n "$SSH_CMD" ] && break
  sleep 5
done

if [ -n "${SSH_CMD:-}" ]; then
  HOST=$(echo "$SSH_CMD" | grep -oP 'root@\K[0-9.]+')
  PORT=$(echo "$SSH_CMD" | grep -oP -- '-p \K[0-9]+')
  if [ -n "$HOST" ] && [ -n "$PORT" ]; then
    mkdir -p ~/.ssh && touch ~/.ssh/config
    # Remove any stale entry for this alias, then append fresh.
    python3 - <<PY
import re, pathlib
p = pathlib.Path.home() / ".ssh" / "config"
txt = p.read_text() if p.exists() else ""
pattern = re.compile(r"(?ms)^# RunPod pod: " + re.escape("$NAME") + r" .*?(?=^Host |\Z)")
txt = pattern.sub("", txt).rstrip() + "\n"
p.write_text(txt)
PY
    cat >> ~/.ssh/config <<CFG

# RunPod pod: $NAME ($POD_ID)
Host $ALIAS
    HostName $HOST
    Port $PORT
CFG
    echo "    Alias:     ssh $ALIAS   (added to ~/.ssh/config; agent forwarding enabled via Host runpod-*)"
  fi
fi
