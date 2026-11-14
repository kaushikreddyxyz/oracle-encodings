#!/usr/bin/env python3
"""Copy of fleet_queue.py for the local (one-swap) experiment: uploads eval_local.py + the local sets, pulls /workspace/local_out3.
Work-queue orchestrator over fleet_ssh: any number of pods pull units (largest first) until the queue is empty.
Pods can be added while it runs (append `name ip:port` lines to the pods file). Every finished unit is pulled to local
disk before the pod takes the next one; state is journaled so a restart skips finished units.

  MAZETREE=<dir with maze/> python local_queue.py <units.json> <pods.txt> <state.jsonl>
units.json: [{"id", "weight", "files": [paths relative to results/], "cmd": "<bash>"}]
"""
import json, os, queue, sys, threading, time
from pathlib import Path
import fleet_ssh as F

HERE = Path(__file__).resolve().parent.parent  # curves/ (this script lives in curves/infra/)
REPO = HERE.parents[2]; R = HERE / "results" / "distance"
CODE = ["eval_ckpts.py", "eval_local.py", "sae_lib.py"]
L = HERE / "results" / "local"
PULLS = [("/workspace/local_out3", L / "out", ())]
lock = threading.Lock()
UPLOAD = threading.Semaphore(int(os.environ.get('MAX_UPLOADS', '4')))      # the laptop uplink is the bottleneck: stage uploads


def log(*a):
    with lock: print(time.strftime("%H:%M:%S"), *a, flush=True)


def base_deploy(ep, guard_h=3):
    assert F.wait(ep), "no sshd"
    if F.run(ep, "test -f /workspace/BASE_OK")[0] == 0: return
    with UPLOAD:
        F.put(ep, "/workspace", str(REPO), ["maze-transformer"], excludes=("__pycache__", ".git", "*.pyc", ".DS_Store", "examples", "notebooks", "tests"))
        F.put(ep, "/workspace", os.environ["MAZETREE"], ["maze"])
    F.run(ep, 'cd /workspace/maze-transformer && nohup sh -c "pip install -q --break-system-packages -e . && pip install -q --break-system-packages fire zanj && touch /workspace/PIP_OK" > /workspace/pip.log 2>&1 &')
    F.run(ep, f"nohup sh -c 'sleep {int(guard_h * 3600)}; runpodctl remove pod $RUNPOD_POD_ID' >/dev/null 2>&1 &")
    for g in range(5, 11):
        d = f"/workspace/distsets/g{g}-s4321-n3000"; F.run(ep, f"mkdir -p {d}")
        c = F.connect(ep); s = c.open_sftp(); s.put(str(R / f"sets/g{g}.npz"), f"{d}/distset_K12_w0.05.npz"); c.close()
    F.put(ep, "/workspace/localsets", str(L / "sets"), [f"g{g}.npz" for g in range(5, 11)])
    for _ in range(150):
        if F.run(ep, "test -f /workspace/PIP_OK")[0] == 0: break
        time.sleep(4)
    else:
        raise RuntimeError("pip timeout")
    F.run(ep, "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null); touch /workspace/BASE_OK")


def worker(name, ep, q, state_path, done_ids):
    for attempt in range(4):
        try:
            base_deploy(ep); break
        except Exception as e:
            log(name, f"deploy attempt {attempt + 1} failed", repr(e)); time.sleep(15)
    else:
        log(name, "DEPLOY FAILED"); return
    while True:
        try:
            _, _, u = q.get_nowait()
        except queue.Empty:
            log(name, "idle: queue empty"); return
        uid = u["id"]; t0 = time.time()
        try:
            F.put(ep, "/workspace/maze/paper/curves", str(HERE), CODE)                     # always the current code
            missing = [f for f in u["files"] if F.run(ep, f"test -f /workspace/runs/{f}")[0] != 0]
            if missing:
                with UPLOAD: F.put(ep, "/workspace/runs", str(HERE / "results"), missing)
            script = f"#!/bin/bash\ncd /workspace; export PYTHONUNBUFFERED=1\n{u['cmd']} 2>&1 | grep --line-buffered -v -i warn\necho UNIT_FINISHED\n"
            F.run(ep, f"cat > /workspace/unit_{uid}.sh", stdin_data=script)
            F.run(ep, f'tmux kill-session -t u 2>/dev/null; tmux new-session -d -s u "bash /workspace/unit_{uid}.sh > /workspace/unit_{uid}.log 2>&1"')
            log(name, "started", uid)
            misses = 0
            while True:
                time.sleep(12)
                try:
                    rc, out, _ = F.run(ep, f"grep -c UNIT_FINISHED /workspace/unit_{uid}.log; grep -c -E 'Traceback|CUDA out of memory' /workspace/unit_{uid}.log", timeout=40)
                except Exception as e:
                    misses += 1
                    if misses in (5, 50, 200): log(name, f"cannot reach pod while {uid} runs ({misses} misses): {e!r}")
                    time.sleep(20); continue
                misses = 0; fin, tb = (out.split() + ["0", "0"])[:2]
                if fin != "0": break
            for rp, lp, ex in PULLS:
                if F.run(ep, f"test -e {rp}")[0] == 0: F.get(ep, rp, str(lp), ex)
            F.get(ep, f"/workspace/unit_{uid}.log", str(L / "logs" / "units"))
            ok = tb == "0" and "RUN_DONE" in F.run(ep, f"tail -n 3 /workspace/unit_{uid}.log")[1]
            with lock:
                open(state_path, "a").write(json.dumps(dict(id=uid, pod=name, ok=ok, minutes=round((time.time() - t0) / 60, 1))) + "\n")
            log(name, "finished" if ok else "FAILED", uid, f"{(time.time() - t0) / 60:.1f} min", f"queue {q.qsize()}")
            if not ok and u.get("retries", 0) < 1:
                u["retries"] = u.get("retries", 0) + 1; q.put((-u["weight"], uid, u))
        except Exception as e:
            log(name, "ERROR on", uid, repr(e)); u["retries"] = u.get("retries", 0) + 1
            q.put((-u["weight"], uid, u))                       # never drop a unit: it goes back to the queue
            time.sleep(min(300, 20 * u["retries"]))
            if u["retries"] % 6 == 0 and not F.wait(ep, tries=3):
                log(name, "pod unreachable, worker exits (unit stays queued)"); return


def main(units_path, pods_path, state_path):
    units = json.load(open(units_path)); done = set()
    if os.path.exists(state_path):
        done = {json.loads(l)["id"] for l in open(state_path) if json.loads(l).get("ok")}
    q = queue.PriorityQueue()
    for u in units:
        if u["id"] not in done: q.put((-u["weight"], u["id"], u))
    log(f"{q.qsize()} units queued ({len(done)} already done)")
    (L / "logs" / "units").mkdir(parents=True, exist_ok=True)
    threads = {}
    while True:
        for line in open(pods_path):
            p = line.split()
            if len(p) >= 2 and p[0] not in threads and ":" in p[-1]:
                t = threading.Thread(target=worker, args=(p[0], p[-1], q, state_path, done), daemon=True); t.start(); threads[p[0]] = t
                log("pod joined:", p[0], p[-1])
        alive = [n for n, t in threads.items() if t.is_alive()]
        if threads and not alive and q.empty(): break
        time.sleep(10)
    log("ALL_UNITS_DONE")


if __name__ == "__main__":
    main(*sys.argv[1:4])
