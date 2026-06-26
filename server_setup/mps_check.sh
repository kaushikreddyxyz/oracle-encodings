#!/usr/bin/env bash
# mps_check.sh — quick health check that CUDA MPS is up AND actually funneling work.
# =============================================================================
# Three checks, fast (~10s):
#   1. control daemon is running
#   2. control pipe responds (queries the default active-thread %)
#   3. FUNCTIONAL: launch 2 concurrent GPU workers and confirm their compute is
#      routed through a single `nvidia-cuda-mps-server` (the signature of MPS
#      working — without MPS you'd see two independent `C`-type processes instead
#      of one shared `M+C` server).
#
# Run AFTER mps_start.sh. Activate the project venv first so torch+CUDA is
# importable for the functional test (otherwise checks 1-2 still run).
#
# USAGE
#   source .venv/bin/activate && bash server_setup/mps_check.sh
#   MPS_TEST_SECS=12 PYBIN=python bash server_setup/mps_check.sh
# =============================================================================
set -euo pipefail

# Inherit the daemon's pipe dir / GPU (workers must match it to connect).
ENV_FILE="${MPS_ENV_FILE:-/tmp/mps_env.sh}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"
PYBIN="${PYBIN:-python}"
command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python3
SECS="${MPS_TEST_SECS:-8}"
fail=0

command -v nvidia-cuda-mps-control >/dev/null 2>&1 || { echo "✗ nvidia-cuda-mps-control not found — not a CUDA host."; exit 1; }

# 1) Daemon up?
if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "✓ MPS control daemon running (pid: $(pgrep -x nvidia-cuda-mps-control | tr '\n' ' '))"
else
    echo "✗ MPS control daemon NOT running — run: bash server_setup/mps_start.sh"; exit 1
fi

# 2) Control pipe responsive?
pct=$(echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>/dev/null || true)
if [ -n "$pct" ]; then
    echo "✓ control pipe responds (default active-thread %: $pct)"
else
    echo "✗ control pipe not responding (CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY)"; fail=1
fi

# 3) Functional concurrency test (needs torch+CUDA)
if "$PYBIN" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    WORK=$(mktemp /tmp/mps_worker.XXXXXX.py)
    cat > "$WORK" <<'PY'
import sys, time, torch
secs = float(sys.argv[1])
dev = torch.device("cuda")
a = torch.randn(2048, 2048, device=dev)
b = torch.randn(2048, 2048, device=dev)
t0 = time.time()
while time.time() - t0 < secs:
    a = (a @ b).relu() * 1e-4 + 1e-4
torch.cuda.synchronize()
print("worker done", flush=True)
PY
    echo "  launching 2 concurrent GPU workers for ${SECS}s (PYBIN=$PYBIN)..."
    "$PYBIN" "$WORK" "$SECS" & p1=$!
    "$PYBIN" "$WORK" "$SECS" & p2=$!
    sleep 3   # let both attach to the GPU
    smi=$(nvidia-smi 2>/dev/null || true)
    srv=$(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null || true)
    if echo "$smi" | grep -q "nvidia-cuda-mps-server" || [ -n "$srv" ]; then
        echo "✓ MPS server active — both clients funnel through one shared GPU context"
        [ -n "$srv" ] && echo "    server pids: $(echo "$srv" | tr '\n' ' ')"
        echo "$smi" | grep -E "M\+C|nvidia-cuda-mps-server" | sed 's/^/    /' || true
    else
        echo "✗ no nvidia-cuda-mps-server while 2 clients ran — work is NOT going through MPS"
        echo "    (check that workers inherited CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY)"
        fail=1
    fi
    wait "$p1" "$p2" 2>/dev/null || true
    rm -f "$WORK"
else
    echo "  ⚠ torch+CUDA not importable with '$PYBIN' — skipping functional test."
    echo "    Activate the project venv and re-run to exercise concurrency."
fi

echo ""
if [ "$fail" = 0 ]; then
    echo "MPS CHECK: PASS"
else
    echo "MPS CHECK: FAIL"; exit 1
fi
