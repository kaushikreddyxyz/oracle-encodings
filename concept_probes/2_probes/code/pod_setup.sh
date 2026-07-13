#!/bin/bash
# One-time pod environment setup (idempotent). Requires HF_TOKEN in env
# (gemma-2-2b is gated; the token also authorizes uploads).
set -euo pipefail
pip install -q --upgrade transformers huggingface_hub hf_transfer scipy pyyaml 2>&1 | tail -1
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('gpu:', torch.cuda.get_device_name(0))"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf auth whoami || huggingface-cli whoami
