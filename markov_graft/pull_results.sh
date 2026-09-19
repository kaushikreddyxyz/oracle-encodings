#!/usr/bin/env bash
# Pull results JSON from every pod running a slice of the grid and regenerate figures.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/results"
for ALIAS in ${@:-runpod-mg runpod-mgb runpod-mgc}; do
  rsync -rlt --no-o --no-g "$ALIAS:/workspace/markov_graft/results/" "$HERE/results/" 2>/dev/null || echo "pull from $ALIAS failed"
done
ls "$HERE/results" | wc -l
"$HERE/../.venv/bin/python" "$HERE/analyze.py" | head -1
