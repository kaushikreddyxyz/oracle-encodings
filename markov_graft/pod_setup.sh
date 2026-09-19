#!/usr/bin/env bash
# One-time pod setup: python deps (the runpod torch image already has torch+cu128) and the CUDA MPS daemon
# (tiny-kernel runs only overlap on one GPU under MPS; without it 4 concurrent runs are slower than serial).
set -euo pipefail
pip install -q --break-system-packages wandb pandas 2>&1 | grep -v WARNING | tail -1 || true
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "mps started"
python -c "import torch, numpy, wandb; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
