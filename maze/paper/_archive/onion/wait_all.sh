#!/bin/bash
# Mac-side: block until every pod's STATUS says WORKER_DONE (or 100 min), then print the status lines.
for i in $(seq 1 200); do
  done_n=0
  for k in 1 2 3 4 5 6; do ssh -o ConnectTimeout=15 runpod-onion-p$k 'grep -q WORKER_DONE /workspace/runs/STATUS' 2>/dev/null && done_n=$((done_n+1)); done
  [ $done_n -eq 6 ] && break
  sleep 30
done
echo "done_n=$done_n"; for k in 1 2 3 4 5 6; do echo "== p$k"; ssh -o ConnectTimeout=15 runpod-onion-p$k 'cat /workspace/runs/STATUS'; done
