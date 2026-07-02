#!/bin/bash
# Pilot: full 26-layer sweep for january / harmfulness / europe (§0.2) +
# natural standardization stats + the §0.1 read-position (+1) ablation.
# Run on the pilot pod from stage5/code. Requires standardization_sample.jsonl
# at $ROOT/standardization_sample.jsonl (scp'd from the mac).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="${ROOT:-/workspace/stage5}"
DATA="$ROOT/stage4_data"
ALL26=$(seq -s, 0 25)
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$ROOT/metrics"

echo "=== natstats (all 26 layers) ==="
test -f "$ROOT/natstats26.npz" || python natstats.py \
  --passages "$ROOT/standardization_sample.jsonl" \
  --layers "$ALL26" --out "$ROOT/natstats26.npz"

pilot() {  # family, class-subset (empty = all)
  local FAM=$1 CLS=$2
  local CARG=(); [ -n "$CLS" ] && CARG=(--classes "$CLS")
  echo "=== [$FAM/$CLS] download ==="
  hf download kaushikreddyxyz/concept-probes-stage4-data --repo-type dataset \
     --include "data/$FAM/final/*" --local-dir "$DATA" > /dev/null
  echo "=== [$FAM/$CLS] extract 26 layers ==="
  test -f "$ROOT/cache26/$FAM/index.json" || python extract.py --family "$FAM" \
     --stage4 "$DATA/data" "${CARG[@]}" --out "$ROOT/cache26/$FAM" --layers "$ALL26"
  echo "=== [$FAM/$CLS] train ==="
  python train.py --family "$FAM" --stage4 "$DATA/data" "${CARG[@]}" \
     --cache "$ROOT/cache26/$FAM" --natstats "$ROOT/natstats26.npz" \
     --layers "$ALL26" --out "$ROOT/probes26/$FAM"
  echo "=== [$FAM/$CLS] evaluate ==="
  python evaluate.py --family "$FAM" --stage4 "$DATA/data" "${CARG[@]}" \
     --cache "$ROOT/cache26/$FAM" --probes "$ROOT/probes26/$FAM" \
     --natstats "$ROOT/natstats26.npz" --layers "$ALL26" \
     --out "$ROOT/metrics/pilot_$FAM.json"
  echo "=== [$FAM/$CLS] read-shift=1 ablation (train+evaluate) ==="
  python train.py --family "$FAM" --stage4 "$DATA/data" "${CARG[@]}" \
     --cache "$ROOT/cache26/$FAM" --natstats "$ROOT/natstats26.npz" \
     --layers "$ALL26" --read-shift 1 --out "$ROOT/probes26_shift1/$FAM"
  python evaluate.py --family "$FAM" --stage4 "$DATA/data" "${CARG[@]}" \
     --cache "$ROOT/cache26/$FAM" --probes "$ROOT/probes26_shift1/$FAM" \
     --natstats "$ROOT/natstats26.npz" --layers "$ALL26" --read-shift 1 \
     --out "$ROOT/metrics/pilot_${FAM}_shift1.json"
}

pilot months january
pilot harmfulness ""
pilot continents europe
echo "PILOT COMPLETE"
