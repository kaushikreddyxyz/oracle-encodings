#!/usr/bin/env python3
"""Harvest pass after an orchestrator/network failure: for every pod, report unit logs (finished / still running), pull all
results, and journal finished units.  python fleet_harvest.py <pods.txt> <state.jsonl>"""
import json, sys, threading
import fleet_ssh as F
from fleet_queue import PULLS, R
pods = [l.split() for l in open(sys.argv[1]) if len(l.split()) >= 2]; state = sys.argv[2]; lock = threading.Lock(); out = {}
def work(name, ep):
    try:
        rc, o, _ = F.run(ep, 'for f in /workspace/unit_fleet-*.log; do [ -f "$f" ] || continue; echo "$(basename $f .log | sed s/unit_//) $(grep -c RUN_DONE $f) $(grep -c UNIT_FINISHED $f) $(grep -c -E "Traceback|out of memory" $f)"; done; echo "BUSY $(pgrep -f eval_steering | wc -l)"', timeout=40)
        units = [l.split() for l in o.strip().splitlines()]
        for rp, lp, ex in PULLS:
            if F.run(ep, f"test -e {rp}")[0] == 0: F.get(ep, rp, str(lp), ex)
        for u in units:
            if u[0] != "BUSY" and u[1] != "0": F.get(ep, f"/workspace/unit_{u[0]}.log", str(R / "sae_logs" / "units"))
        with lock: out[name] = units
    except Exception as e:
        with lock: out[name] = [["ERR", repr(e)]]
ts = [threading.Thread(target=work, args=(p[0], p[-1])) for p in pods]
[t.start() for t in ts]; [t.join() for t in ts]
done = {json.loads(l)["id"] for l in open(state) if json.loads(l).get("ok")}
for name, units in sorted(out.items()):
    print(name, " ".join(f"{u[0]}:{'done' if u[1] != '0' else ('fail' if len(u) > 3 and u[3] != '0' else 'running/partial')}" if u[0] not in ("BUSY", "ERR") else f"{u[0]}={u[1]}" for u in units))
    for u in units:
        if u[0] not in ("BUSY", "ERR") and u[1] != "0" and u[0] not in done:
            open(state, "a").write(json.dumps(dict(id=u[0], pod=name, ok=True, harvested=True)) + "\n"); done.add(u[0])
print(len(done), "units ok")
