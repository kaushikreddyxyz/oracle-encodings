#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")"
for f in directions continents lovingness duration; do
  echo "=== GEN $f $(date -u +%H:%M:%SZ) ==="
  python3 generate.py --family "$f" --cap-usd 10 || echo "!!! GEN FAILED: $f"
done; echo "=== LANE B DONE ==="
