#!/bin/bash
# Poll trainer pod for expB_learn completion, then pull its metrics and retro-log to wandb.
set -e
ROOT=/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i $HOME/.runpod/ssh/runpodctl-ssh-key root@31.24.80.36 -p 10617"
SCP="scp -o StrictHostKeyChecking=no -o BatchMode=yes -i $HOME/.runpod/ssh/runpodctl-ssh-key -P 10617"
OUT=$ROOT/concept_probes/stage7_oracle/out/retro_metrics
for i in $(seq 1 90); do
  STEP=$($SSH 'tail -1 /workspace/expB_learn/metrics.jsonl 2>/dev/null | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get(\"step\",0))" 2>/dev/null' 2>/dev/null || echo 0)
  echo "$(date +%H:%M) expB_learn step=$STEP"
  if [ "$STEP" -ge 1282 ] 2>/dev/null; then
    $SCP root@31.24.80.36:/workspace/expB_learn/metrics.jsonl $OUT/expB_learn_metrics.jsonl
    cd $ROOT
    set -a; . ./.env 2>/dev/null; set +a; export WANDB_API_KEY="$WANDB_TOKEN"
    .venv/bin/python concept_probes/stage7_oracle/code/wandb_retrolog.py \
      --metrics $OUT/expB_learn_metrics.jsonl --name expB-learn --project stage7-oracle \
      --config mode=expB-learn --config encoder_from=expA_prod/best.pt --config encoder=fine-tune \
      --config lr=3e-3 --config bsz_docs=64 --config grad_accum=4 --config max_steps=1282 \
      --tags expB,learn --notes "Exp B learn: encoder-learning v* head" 2>&1 | grep -E "\[retrolog\]|View run .* at:"
    echo "expB-learn retro-logged to wandb"
    exit 0
  fi
  sleep 120
done
echo "expB_learn watcher timed out after 3h"
