#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")"
for f in color_wheel moon_phases costliness physical_size; do
  echo "=== GEN $f $(date -u +%H:%M:%SZ) ==="
  python3 generate.py --family "$f" --cap-usd 10 || echo "!!! GEN FAILED: $f"
done; echo "=== LANE A DONE ==="
