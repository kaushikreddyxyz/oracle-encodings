#!/bin/bash
# Finalize lane: as each family's judging completes (judge_report.json appears),
# run curation (§4.7) and assembly (§4.4/§0.5/§4.6). Local compute only, no API.
set -uo pipefail
cd "$(dirname "$0")"
for f in months weekdays seasons color_wheel directions moon_phases continents \
         location_type costliness physical_size lovingness duration harmfulness; do
  waited=0
  until [ -f "../data/$f/judged/judge_report.json" ]; do
    sleep 60; waited=$((waited+60))
    if [ "$waited" -ge 21600 ]; then echo "!!! TIMEOUT waiting for $f judging"; break; fi
  done
  if [ -f "../data/$f/judged/judge_report.json" ]; then
    echo "=== FINALIZE $f $(date -u +%H:%M:%SZ) ==="
    python3 curate.py --family "$f" && python3 assemble.py --family "$f" \
      || echo "!!! FINALIZE FAILED: $f"
  fi
done
echo "=== FINALIZE LANE DONE $(date -u +%H:%M:%SZ) ==="
