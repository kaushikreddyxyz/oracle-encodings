#!/bin/bash
# Judge lane v2: skips families already judged; waits for gen reports.
set -uo pipefail
cd "$(dirname "$0")"
for f in months seasons color_wheel location_type directions moon_phases continents \
         costliness physical_size lovingness duration harmfulness weekdays; do
  if [ -f "../data/$f/judged/judge_report.json" ]; then
    echo "=== SKIP $f (already judged) ==="; continue
  fi
  waited=0
  until [ -f "../data/$f/raw_gen/gen_report.json" ]; do
    sleep 30; waited=$((waited+30))
    if [ "$waited" -ge 14400 ]; then echo "TIMEOUT waiting for $f gen"; break; fi
  done
  if [ -f "../data/$f/raw_gen/gen_report.json" ]; then
    echo "=== JUDGE $f $(date -u +%H:%M:%SZ) ==="
    python3 judge.py --family "$f" --cap-usd 30 || echo "!!! JUDGE FAILED: $f"
  fi
done
echo "=== JUDGE LANE DONE $(date -u +%H:%M:%SZ) ==="
