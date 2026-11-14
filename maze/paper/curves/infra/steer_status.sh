#!/bin/bash
# bash steer_status.sh  -> per-shard progress of the steering run (reads results/distance/steer_logs/endpoints.txt)
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # curves/ (this script lives in curves/infra/); tot=0; err=0
while read -r sh id ip port; do a=runpod-mzs-$sh; [ -n "${ALT:-}" ] && [ "$sh" = "${ALT%%:*}" ] && a="${ALT##*:}"
  r=$(ssh -n -o ConnectTimeout=10 "$a" 'echo "$(grep -c "^\[g" /workspace/steer.log) $(grep -c SHARD_DONE /workspace/steer.log) $(grep -c Traceback /workspace/steer.log)"' 2>/dev/null)
  set -- $r; echo "$sh conds=${1:-?} done=${2:-?} tracebacks=${3:-?}"; tot=$((tot + ${2:-0})); err=$((err + ${3:-0}))
done < "$HERE/results/distance/steer_logs/${EP:-endpoints.txt}"
echo "TOTAL shards_done=$tot tracebacks=$err"
