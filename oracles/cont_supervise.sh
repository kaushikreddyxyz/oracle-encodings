#!/bin/bash
# Stage-7 continuation supervisor (runs under nohup on the pod):
#   0. download resume checkpoint from HF
#   1. start prefetch_shards.py in background (pidfile, idempotent)
#   2. run the trainer in foreground
#   3. run cont_teardown.py (upload+verify+self-terminate; HOLDs on problems)
# Config comes from /workspace/.env; secrets from /workspace/.hf_token,
# /workspace/.wandb_key, /workspace/.rp_key (all chmod 600, staged via scp).
set -u
set -a; source /workspace/.env; set +a
export HF_TOKEN=$(cat /workspace/.hf_token)
export WANDB_API_KEY=$(cat /workspace/.wandb_key)
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/code
mkdir -p /workspace/data /workspace/climbmix /workspace/run /workspace/resume

echo "$(date -u +%FT%TZ) supervisor start layer=${LAYER}" >> /workspace/supervise.log

python3 - <<PY
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download("${HF_REPO}", "${RESUME_PATH}", repo_type="model",
                    local_dir="/workspace/resume")
print("resume ckpt staged:", p, os.path.getsize(p), flush=True)
PY
if [ $? -ne 0 ]; then
  echo "$(date -u +%FT%TZ) FATAL: resume checkpoint download failed" >> /workspace/supervise.log
  exit 1
fi

if [ ! -f /workspace/prefetch.pid ] || ! kill -0 "$(cat /workspace/prefetch.pid)" 2>/dev/null; then
  nohup python3 -u prefetch_shards.py --dest /workspace/data \
    --climbmix-dest /workspace/climbmix --shards "${PREFETCH_SHARDS}" \
    > /workspace/prefetch.log 2>&1 &
  echo $! > /workspace/prefetch.pid
  echo "$(date -u +%FT%TZ) prefetch started pid $(cat /workspace/prefetch.pid)" >> /workspace/supervise.log
fi

python3 -u train_oracle_perlayer.py \
  --layer "${LAYER}" \
  --scores /workspace/data --climbmix-dir /workspace/climbmix \
  --train-shards "${TRAIN_SHARDS}" --val-shards "353,354" \
  --resume "/workspace/resume/${RESUME_PATH}" \
  --max-tokens "${MAX_TOKENS}" --max-hours "${MAX_HOURS}" \
  --cont-anchor-mult "${ANCHOR_MULT}" --cont-end-step "${END_STEP}" \
  --wandb-name "${WANDB_NAME}" --hf-subdir "${HF_SUBDIR}" \
  --out /workspace/run > /workspace/train.log 2>&1
echo "$(date -u +%FT%TZ) trainer exited rc=$?" >> /workspace/supervise.log

python3 -u cont_teardown.py >> /workspace/teardown.log 2>&1
echo "$(date -u +%FT%TZ) teardown script exited rc=$?" >> /workspace/supervise.log
