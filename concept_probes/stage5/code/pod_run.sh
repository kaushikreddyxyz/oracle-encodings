#!/bin/bash
# Per-pod Stage 5 driver: download family data -> extract -> train -> evaluate
# -> upload artifacts. Usage:  bash pod_run.sh <family> [<family> ...]
# Env: LAYERS (csv), ROOT (default /workspace/stage5), NATSTATS (npz path),
#      UPLOAD=1 to push probes+metrics to the HF weights repo as they finish.
set -euo pipefail
cd "$(dirname "$0")"
LAYERS="${LAYERS:-1,3,6,8,10,12,14,16,18,20,23,25}"
ROOT="${ROOT:-/workspace/stage5}"
DATA="$ROOT/stage4_data"
NATSTATS="${NATSTATS:-$ROOT/natstats.npz}"
WREPO="kaushikreddyxyz/concept-probes-gemma2-2b"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$ROOT" "$ROOT/metrics"

test -f "$NATSTATS" || { echo "FATAL: natstats not found at $NATSTATS"; exit 1; }

for FAM in "$@"; do
  echo "=== [$FAM] download stage4 data ==="
  hf download kaushikreddyxyz/concept-probes-stage4-data --repo-type dataset \
     --include "data/$FAM/final/*" --local-dir "$DATA" > /dev/null
  echo "=== [$FAM] extract ==="
  python extract.py --family "$FAM" --stage4 "$DATA/data" \
     --out "$ROOT/cache/$FAM" --layers "$LAYERS"
  echo "=== [$FAM] train ==="
  python train.py --family "$FAM" --stage4 "$DATA/data" --cache "$ROOT/cache/$FAM" \
     --natstats "$NATSTATS" --layers "$LAYERS" --out "$ROOT/probes/$FAM"
  echo "=== [$FAM] evaluate ==="
  python evaluate.py --family "$FAM" --stage4 "$DATA/data" --cache "$ROOT/cache/$FAM" \
     --probes "$ROOT/probes/$FAM" --natstats "$NATSTATS" --layers "$LAYERS" \
     --out "$ROOT/metrics/$FAM.json"
  if [ -f "$ROOT/nat_eval/$FAM.jsonl" ]; then
    echo "=== [$FAM] score natural ==="
    python score_natural.py --family "$FAM" --eval "$ROOT/nat_eval/$FAM.jsonl" \
       --probes "$ROOT/probes/$FAM" --natstats "$NATSTATS" \
       --gen-cache "$ROOT/cache/$FAM" --cache "$ROOT/natcache/$FAM" \
       --layers "$LAYERS" --out "$ROOT/natscores"
  else
    echo "=== [$FAM] no nat_eval file — skipping natural scoring ==="
  fi
  if [ "${UPLOAD:-0}" = "1" ]; then
    echo "=== [$FAM] upload ==="
    hf upload "$WREPO" "$ROOT/probes/$FAM" "families/$FAM" --commit-message "stage5 $FAM probes" > /dev/null
    hf upload "$WREPO" "$ROOT/metrics/$FAM.json" "metrics/$FAM.generated.json" \
       --commit-message "stage5 $FAM generated-split metrics" > /dev/null
    test -f "$ROOT/natscores/$FAM.natscores.npz" && \
      hf upload "$WREPO" "$ROOT/natscores/$FAM.natscores.npz" "natscores/$FAM.natscores.npz" \
         --commit-message "stage6 $FAM natural scores" > /dev/null
  fi
  echo "=== [$FAM] DONE ==="
done
echo "ALL FAMILIES DONE: $*"
