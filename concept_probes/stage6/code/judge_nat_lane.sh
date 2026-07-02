#!/bin/bash
# Stage 6 natural-data judging lane: sequential families through the unchanged
# Stage-4 judge (mercury-2, K=3, --tag nat). Resume-safe; per-family cost cap.
set -uo pipefail
cd "$(dirname "$0")/../../stage4/code"
LOG="../../stage6/data/natural/judge_lane.log"
FAMS="${FAMS:-months weekdays seasons color_wheel directions moon_phases continents location_type costliness physical_size lovingness duration harmfulness}"
for FAM in $FAMS; do
  echo "=== $(date '+%H:%M:%S') judging $FAM (nat) ===" >> "$LOG"
  python judge.py --family "$FAM" --tag nat --cap-usd 5.0 >> "$LOG" 2>&1
  echo "=== $(date '+%H:%M:%S') $FAM exit=$? ===" >> "$LOG"
done
echo "=== $(date '+%H:%M:%S') NAT JUDGING LANE COMPLETE ===" >> "$LOG"
