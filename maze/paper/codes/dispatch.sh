#!/bin/bash
# Mac-side job dispatcher: when a candidate pod's worker tmux session is gone and its STATUS ends in WORKER_DONE, pull it,
# pop the next "<grid> <seed> <job>" line from the queue, ship that grid's eval sets and launch the job there.
#   bash dispatch.sh <queue-file> <pod-name>...      (logs to stdout; exits when the queue is empty)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; Q=$1; shift; USED=""
while [ -s "$Q" ]; do
  for n in "$@"; do
    case " $USED " in *" $n "*) continue;; esac
    # free = the worker tmux session is gone (guard-run evals may still be finishing in the background; they share the GPU)
    ssh -o ConnectTimeout=12 -o BatchMode=yes "runpod-$n" 'tmux has-session -t worker 2>/dev/null && exit 1; [ -f /workspace/runs/STATUS ]' 2>/dev/null || continue
    USED="$USED $n"; read -r G S JOB < "$Q"; tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    echo "[$(date -u +%H:%M)] $n free -> pulling, then g$G s$S $JOB"
    bash "$HERE/codes_pod.sh" pull "runpod-$n" > /dev/null 2>&1
    bash "$HERE/codes_pod.sh" setup "runpod-$n" "$G" > "/tmp/setup-$n-g$G.log" 2>&1 || { echo "[$(date -u +%H:%M)] $n: setup failed"; continue; }
    ssh -o BatchMode=yes "runpod-$n" 'grep -c WORKER_DONE /workspace/runs/STATUS' | sed "s/^/[$n] WORKER_DONE lines before relaunch: /"
    echo "[$(date -u +%H:%M)] $n: $(bash "$HERE/codes_pod.sh" run "runpod-$n" "$G" "$S" "$JOB" | tail -1)"
    python3 - "$n" "$G" "$S" "$JOB" <<'PY'
import json,sys; n,g,s,job=sys.argv[1:]; f="fleet_ev.json"; d=json.load(open(f)); d["pods"].setdefault(n,{}).setdefault("jobs_added",[]).append(f"g{g} s{s} {job}"); json.dump(d,open(f,"w"),indent=1)
PY
    [ -s "$Q" ] || break
  done
  [ -s "$Q" ] && sleep 120
done
echo "[$(date -u +%H:%M)] queue empty"
