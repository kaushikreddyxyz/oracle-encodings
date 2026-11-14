#!/bin/bash
# For every live pod whose worker is done: pull results, verify, print "<pod> <id>" (deletion is done via MCP), and mark it.
cd "$(dirname "$0")"
for k in $(python3 -c "import json; d=json.load(open('pods.json')); print(' '.join(k for k in sorted(d) if not d[k].get('deleted')))"); do
  a=runpod-c${k#curves-}
  st=$(ssh -o ConnectTimeout=15 -o BatchMode=yes $a 'echo "$(grep -c WORKER_DONE /workspace/runs/STATUS 2>/dev/null) $(ls /workspace/runs/*/*/DONE 2>/dev/null | wc -l)"' 2>/dev/null)
  set -- $st; [ "${1:-0}" != 1 ] && continue
  bash curves_pod.sh pull $a >/dev/null 2>&1
  loc=$(find ../results/$a -name DONE 2>/dev/null | wc -l | tr -d ' ')
  if [ "$loc" = "$2" ]; then
    id=$(python3 -c "import json; print(json.load(open('pods.json'))['$k']['id'])")
    echo "READY_TO_DELETE $k $id (remote $2 runs, local $loc)"
  else
    echo "MISMATCH $k remote $2 local $loc"
  fi
done
