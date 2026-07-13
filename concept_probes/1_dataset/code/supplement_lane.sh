#!/bin/bash
# Post-sweep repairs, run AFTER all 13 assemblies exist (never competes with
# the main judge lane): (1) hazard-surface hardfix, (2) form-holdout supplement.
set -uo pipefail
cd "$(dirname "$0")"
FAMS="months weekdays seasons color_wheel directions moon_phases continents location_type costliness physical_size lovingness duration harmfulness"
while true; do
  missing=0
  for f in $FAMS; do [ -f "../data/$f/final/assembly_report.json" ] || missing=1; done
  [ "$missing" = 0 ] && break
  sleep 120
done
echo "=== all assemblies present $(date -u +%H:%M:%SZ) ==="
python3 hardfix_hazards.py --families $FAMS --cap-usd 10 || echo "!!! HARDFIX FAILED"
python3 supplement_forms.py --families $FAMS --cap-usd 10 || echo "!!! SUPPLEMENT FAILED"
echo "=== SUPPLEMENT LANE DONE $(date -u +%H:%M:%SZ) ==="
