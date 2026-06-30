#!/usr/bin/env bash
# Run ONE concept (used by run_all_parallel.sh via xargs -P). Logs to its own file.
# Env: POS, HELDOUT, CONC (per-process OpenRouter concurrency).
c="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python3"
LOG="$ROOT/overnight_run2/logs"
mkdir -p "$LOG"
echo "$(date -u +%H:%M:%S) START $c" >> "$LOG/phaseA_parallel.log"
if "$PY" "$HERE/run_concept.py" --concept "$c" \
      --pos-per-value "${POS:-4000}" --heldout-per-value "${HELDOUT:-40}" \
      --concurrency "${CONC:-24}" > "$LOG/A_$c.log" 2>&1; then
  echo "$(date -u +%H:%M:%S) DONE $c" >> "$LOG/phaseA_parallel.log"
else
  echo "$(date -u +%H:%M:%S) FAILED $c (rc=$?)" >> "$LOG/phaseA_parallel.log"
fi
