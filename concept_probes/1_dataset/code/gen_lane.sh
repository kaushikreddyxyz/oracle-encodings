#!/bin/bash
# Generation lane: runs every remaining family sequentially on gpt-oss-120b.
# months runs separately (already in flight when this lane starts).
set -uo pipefail
cd "$(dirname "$0")"
for f in weekdays seasons color_wheel directions moon_phases continents \
         location_type costliness physical_size lovingness duration harmfulness; do
  echo "=== GEN $f $(date -u +%H:%M:%SZ) ==="
  python3 generate.py --family "$f" --cap-usd 10 || echo "!!! GEN FAILED: $f"
done
echo "=== GEN LANE DONE $(date -u +%H:%M:%SZ) ==="
