#!/usr/bin/env python3
"""Work-queue orchestrator for the steelman rerun (adapted copy of maze/paper/curves/infra/fleet_queue.py; uses its fleet_ssh).
Any number of pods pull units (largest first) until the queue is empty; pods can be added while it runs (append
`name ip:port` lines to the pods file). Each finished unit's outputs are pulled before the pod takes the next one; state is
journaled so a restart skips finished units; a unit is never dropped.

  python fleet_queue.py <units.json> <pods.txt> <state.jsonl>
units.json: [{"id", "weight", "tag", "run": "<dir under curves/results>" | null, "cmd": "<bash>"}]
The pods need no maze library: steelman/ carries a plain-torch copy of the model (model.py).
"""
import json, os, queue, sys, threading, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; CURVES = HERE.parents[1] / "curves"
sys.path.insert(0, str(CURVES / "infra"))          # fleet_ssh.py lives in curves/infra/
import fleet_ssh as F

OUT = HERE / "results"; LOGS = HERE / "logs"
CODE = ["sae_lib.py", "model.py", "maze_distance.py", "common.py", "selection.py", "sae_sweep.py", "model_unit.py", "run_sae.sh", "inputs", "jobs"]
lock = threading.Lock(); UPLOAD = threading.Semaphore(int(os.environ.get("MAX_UPLOADS", "4")))
GUARD_H = float(os.environ.get("GUARD_H", "5")); DONE = set()


def log(*a):
    with lock: print(time.strftime("%H:%M:%S"), *a, flush=True)


def base_deploy(ep):
    assert F.wait(ep), "no sshd"
    if F.run(ep, "test -f /workspace/BASE_OK")[0] == 0: return
    F.run(ep, f"nohup sh -c 'sleep {int(GUARD_H * 3600)}; runpodctl remove pod $RUNPOD_POD_ID' >/dev/null 2>&1 &")
    F.run(ep, 'nohup sh -c "pip install -q --break-system-packages scipy numpy && touch /workspace/PIP_OK" > /workspace/pip.log 2>&1 &')
    F.run(ep, "mkdir -p /workspace/sets /workspace/out /workspace/runs /workspace/pool")
    with UPLOAD: F.put(ep, "/workspace/sets", str(CURVES / "results/distance/sets"), [f"g{g}.npz" for g in range(5, 11)])
    for _ in range(150):
        if F.run(ep, "test -f /workspace/PIP_OK")[0] == 0: break
        time.sleep(4)
    else: raise RuntimeError("pip timeout")
    F.run(ep, "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null); touch /workspace/BASE_OK")


def small_files(tag):
    base = OUT / tag
    return [str(p.relative_to(OUT)) for p in base.rglob("*") if p.is_file() and p.suffix in (".json", ".npz") and p.stat().st_size < 30e6] if base.exists() else []


def worker(name, ep, q, state_path, pinned=()):
    for attempt in range(6):
        try: base_deploy(ep); break
        except Exception as e: log(name, f"deploy attempt {attempt + 1} failed", repr(e)); time.sleep(15)
    else:
        log(name, "DEPLOY FAILED"); return
    pinned = list(pinned)
    while True:
        if pinned: u = pinned.pop(0)
        else:
            deferred = []; u = None
            while True:                                            # a unit with unmet "after" dependencies waits; others go first
                try: _, _, cand = q.get_nowait()
                except queue.Empty: break
                if all(d in DONE for d in cand.get("after", ())): u = cand; break
                deferred.append(cand)
            for d in deferred: q.put((-d["weight"], d["id"], d))
            if u is None:
                if not deferred: log(name, "idle: queue empty"); return
                time.sleep(30); continue
        uid = u["id"]; tag = u["tag"]; t0 = time.time()
        try:
            F.put(ep, "/workspace/steelman", str(HERE), CODE)
            if u.get("run") and F.run(ep, f"test -f /workspace/runs/{u['run']}/model.final.pt")[0] != 0:
                with UPLOAD: F.put(ep, "/workspace/runs", str(CURVES / "results"), [f"{u['run']}/model.final.pt"])
            sf = small_files(tag)
            if sf and u.get("push_out", True): F.put(ep, "/workspace/out", str(OUT), sf)
            script = f"#!/bin/bash\ncd /workspace/steelman; export PYTHONUNBUFFERED=1\n{{ {u['cmd']} ; }} 2>&1 | grep --line-buffered -v -i warn\necho UNIT_FINISHED\n"
            F.run(ep, f"cat > /workspace/unit_{uid}.sh", stdin_data=script)
            F.run(ep, f'tmux kill-session -t u 2>/dev/null; tmux new-session -d -s u "bash /workspace/unit_{uid}.sh > /workspace/unit_{uid}.log 2>&1"')
            log(name, "started", uid); misses = 0
            while True:
                time.sleep(15)
                try: rc, out, _ = F.run(ep, f"grep -c UNIT_FINISHED /workspace/unit_{uid}.log", timeout=40)
                except Exception as e:
                    misses += 1
                    if misses in (5, 50, 200): log(name, f"cannot reach pod while {uid} runs ({misses} misses): {e!r}")
                    time.sleep(20); continue
                misses = 0
                if (out.split() + ["0"])[0] != "0": break
            for attempt in range(8):
                try:
                    F.get(ep, f"/workspace/out/{tag}", str(OUT / tag), ("acts-post0.pt", "dirs", "*.claim", "*.tmp"))
                    F.get(ep, f"/workspace/unit_{uid}.log", str(LOGS)); break
                except Exception as e: log(name, "pull retry", uid, repr(e)); time.sleep(20)
            tail = F.run(ep, f"tail -n 5 /workspace/unit_{uid}.log; grep -c -E 'Traceback|CUDA out of memory' /workspace/unit_{uid}.log")[1]
            ok = "RUN_DONE" in tail and tail.strip().split()[-1] == "0"
            with lock:
                open(state_path, "a").write(json.dumps(dict(id=uid, pod=name, ok=ok, minutes=round((time.time() - t0) / 60, 1))) + "\n")
                if ok: DONE.add(uid)
            log(name, "finished" if ok else "FAILED", uid, f"{(time.time() - t0) / 60:.1f} min", f"queue {q.qsize()}")
            if not ok and u.get("retries", 0) < 1:
                u["retries"] = u.get("retries", 0) + 1; q.put((-u["weight"], uid, u))
        except Exception as e:
            log(name, "ERROR on", uid, repr(e)); u["retries"] = u.get("retries", 0) + 1; q.put((-u["weight"], uid, u))
            time.sleep(min(300, 20 * u["retries"]))
            if u["retries"] % 6 == 0 and not F.wait(ep, tries=3):
                log(name, "pod unreachable, worker exits (unit stays queued)"); return


def main(units_path, pods_path, state_path):
    units = json.load(open(units_path)); done = set(); OUT.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    if os.path.exists(state_path): done = {json.loads(l)["id"] for l in open(state_path) if json.loads(l).get("ok")}
    DONE.update(done)
    q = queue.PriorityQueue()
    pins = {}
    for u in units:
        if u["id"] in done: continue
        if u.get("pod"): pins.setdefault(u["pod"], []).append(u)
        else: q.put((-u["weight"], u["id"], u))
    log(f"{q.qsize()} units queued ({len(done)} already done)"); threads = {}
    while True:
        for line in open(pods_path):
            p = line.split()
            if len(p) >= 2 and p[0] not in threads and ":" in p[-1]:
                t = threading.Thread(target=worker, args=(p[0], p[-1], q, state_path, pins.get(p[0], ())), daemon=True); t.start(); threads[p[0]] = t; log("pod joined:", p[0], p[-1])
        if threads and not any(t.is_alive() for t in threads.values()) and q.empty(): break
        if q.empty() and not any(t.is_alive() for t in threads.values()): break
        time.sleep(10)
    log("ALL_UNITS_DONE")


if __name__ == "__main__":
    main(*sys.argv[1:4])
