#!/usr/bin/env python3
"""Second-stage scheduler for the local experiment when the laptop uplink is the bottleneck (it fell to ~0.4 MB/s).

Pods that already hold model files (`old`, md5-verified against the laptop copies) get a second queue made only of models
they hold intact, so they need no upload and start at once. Freshly deployed pods (`new`: code + pip done, no data) are fed
from the laptop over the same paramiko channel, one large-grid base model each (most GPU minutes per uploaded byte),
together with that grid's eval sets only. Results of every second queue go to /workspace/local_out2.
  python local_fleet2.py plan|apply <fleet.txt>
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
import fleet_ssh as F, local_deploy as D, local_fleet as LF

UPLOAD_MIN = 2.5                                                     # one base model + stats over the shared uplink
FEED_DELAY = {"h01": 2, "h02": 4, "h04": 6, "n01": 8, "h03": 10}     # minutes until a new pod can start (upload order)


def files_of(R, t):
    g, s, k = t; rel = R[t]
    return [f"{rel}/model.final.pt", f"{rel}/config_holder.json", f"distance/steer_gs_prep/g{g}-s{s}/stats_sae.pt" if k == "base" else f"{rel}/injector.final.pt"]


def plan(path):
    R = D.runs(); pods = LF.fleet(path); local = json.load(open(D.TMP / "local_md5.json")); remote = json.load(open(D.TMP / "remote_md5.json"))
    with ThreadPoolExecutor(8) as ex:
        state = dict(zip([p[0] for p in pods], ex.map(lambda p: LF.pod_state(p[0], p[1], p[4]) if p[2] == "old" else (set(), None, 0.0), pods)))
    done = set().union(*[s[0] for s in state.values()]); running = {s[1] for s in state.values() if s[1]}
    tasks = sorted((k for k in R if k not in done and k not in running), key=lambda k: -LF.cost(k[0], k[2]))
    intact = {n: {f for f, h in remote.get(n, {}).items() if local.get(f) == h} for n in state}
    load = {p[0]: (state[p[0]][2] if p[2] == "old" else 0.0) for p in pods}; speed = {p[0]: float(p[3]) for p in pods}
    kind = {p[0]: p[2] for p in pods}; assign = {p[0]: [] for p in pods}
    for t in tasks:                                        # 1. everything to a pod that already holds the files (longest first, earliest finish)
        ok = [n for n in load if kind[n] == "old" and all(f in intact[n] for f in files_of(R, t))]
        if not ok: continue
        best = min(ok, key=lambda n: load[n] + LF.cost(t[0], t[2])); assign[best].append(t); load[best] += LF.cost(t[0], t[2])
    orphans = [t for t in tasks if not any(t in a for a in assign.values())]
    clock = 0.0; news = [n for n in load if kind[n] == "new"]; load.update({n: 0.0 for n in news})
    def give(t, n):
        nonlocal clock
        clock += UPLOAD_MIN; load[n] = max(load[n], clock) + LF.cost(t[0], t[2]) / speed[n]; assign[n].append(t)
    for t in orphans: give(t, min(news, key=lambda n: load[n]))
    while news:                                            # 2. offload base models from the busiest holder to the fed pods while that shortens the makespan
        src = max((n for n in load if kind[n] == "old"), key=lambda n: load[n]); dst = min(news, key=lambda n: load[n])
        cand = [t for t in assign[src] if t[2] == "base"]
        if not cand: break
        t = max(cand, key=lambda t: LF.cost(t[0], t[2]))
        if max(load[dst], clock + UPLOAD_MIN) + LF.cost(t[0], t[2]) / speed[dst] >= load[src]: break
        assign[src].remove(t); load[src] -= LF.cost(t[0], t[2]); give(t, dst)
    for p in pods: print(f"{p[0]:4s} {p[2]} running={state[p[0]][1]} -> {len(assign[p[0]])} tasks, finish ~{load[p[0]]:.0f} min: {assign[p[0]]}", flush=True)
    print(f"{len(tasks)} tasks; makespan ~{max(load.values()):.0f} min", flush=True)
    return R, pods, state, assign, local, remote


def start_queue(R, pod, state, tasks, wait_running=True):
    name, ep = pod[0], pod[1]; _, job = LF.job_lines(R, tasks); head = ["#!/bin/bash", "cd /workspace; export PYTHONUNBUFFERED=1"]; wait = []
    if pod[2] == "old":
        shard = pod[4]; done, running, _ = state
        skip = [f"g{g}__{k}__g{g}-s{s}.npz" for (g, s, k) in R if (g, s, k) not in done and (g, s, k) != running]
        F.run(ep, "tmux kill-session -t x" + shard[1:] + " 2>/dev/null; mkdir -p /workspace/local_out; cd /workspace/local_out; for f in " + " ".join(skip) + "; do [ -e $f ] || : > $f; done")
        if running: wait = [f"until grep -aq 'RUN_DONE g{running[0]}__{running[2]}__g{running[0]}-s{running[1]}' /workspace/job_{shard}.log; do sleep 5; done"]
    F.run(ep, "cat > /workspace/job_q2.sh", stdin_data="\n".join(head + wait + job + ["echo SHARD_DONE"]) + "\n")
    F.run(ep, 'tmux kill-session -t q2 2>/dev/null; tmux new-session -d -s q2 "bash /workspace/job_q2.sh 2>&1 | tee /workspace/job_q2.log"')
    return f"{name}: queue of {len(tasks)} started"


def marker(t): return f"/workspace/fed_g{t[0]}_s{t[1]}_{t[2]}"


def start_fed_queue(R, pod, tasks):
    """Queue on a fed pod: every task waits for the marker that feed_task() sets once its files are uploaded and md5-verified."""
    ep = pod[1]; head = ["#!/bin/bash", "cd /workspace; export PYTHONUNBUFFERED=1"]; job = []
    for t in tasks: job += [f"until [ -e {marker(t)} ]; do sleep 3; done", LF.job_lines(R, [t])[1][0]]
    F.run(ep, "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)")
    F.run(ep, "cat > /workspace/job_q2.sh", stdin_data="\n".join(head + job + ["echo SHARD_DONE"]) + "\n")
    F.run(ep, 'tmux kill-session -t q2 2>/dev/null; tmux new-session -d -s q2 "bash /workspace/job_q2.sh 2>&1 | tee /workspace/job_q2.log"')


def feed_task(R, pod, t, local):
    name, ep = pod[0], pod[1]; t0 = time.time(); g = t[0]
    if F.run(ep, f"test -e /workspace/localsets/g{g}.npz -a -e /workspace/distsets/g{g}-s4321-n3000/distset_K12_w0.05.npz")[0] != 0:
        F.put(ep, "/workspace/localsets", str(D.RES / "local/sets"), [f"g{g}.npz"])
        d = f"/workspace/distsets/g{g}-s4321-n3000"; F.run(ep, f"mkdir -p {d}")
        c = F.connect(ep); s = c.open_sftp(); s.put(str(D.RES / f"distance/sets/g{g}.npz"), f"{d}/distset_K12_w0.05.npz"); c.close()
    files = LF.job_lines(R, [t])[0]
    def md5s():
        out = F.run(ep, "cd /workspace/runs 2>/dev/null && md5sum " + " ".join(files) + " 2>/dev/null")[1]
        return {l.split(None, 1)[1].strip(): l.split()[0] for l in out.splitlines()}
    need = [f for f in files if md5s().get(f) != local[f]]
    if need: F.put(ep, "/workspace/runs", str(D.RES), need)
    have = md5s(); assert all(have.get(f) == local[f] for f in files), (name, t)
    F.run(ep, f"touch {marker(t)}")
    return f"{name}: {t} fed ({len(need)} files, {time.time() - t0:.0f}s)"


def supplement(path, conds, tag):
    """Extra conditions for every base model, on pods that already hold that model intact (no upload). Queued behind q2."""
    R = D.runs(); pods = LF.fleet(path); local = json.load(open(D.TMP / "local_md5.json")); base = [t for t in R if t[2] == "base"]
    def inv(p):
        out = F.run(p[1], "cd /workspace/runs && find . -type f \\( -name '*.pt' -o -name '*.json' \\) -print0 | xargs -0 md5sum", timeout=600)[1]
        return {l.split(None, 1)[1].strip()[2:] for l in out.splitlines() if local.get(l.split(None, 1)[1].strip()[2:]) == l.split()[0]}
    with ThreadPoolExecutor(12) as ex: have = dict(zip([p[0] for p in pods], ex.map(inv, pods)))
    load = {p[0]: 0.0 for p in pods}; speed = {p[0]: float(p[3]) for p in pods}; assign = {p[0]: [] for p in pods}
    for t in sorted(base, key=lambda t: -LF.COST[t[0]]):
        ok = [n for n in load if all(f in have[n] for f in files_of(R, t))]; assert ok, t
        best = min(ok, key=lambda n: load[n] + LF.COST[t[0]] / speed[n]); assign[best].append(t); load[best] += LF.COST[t[0]] / speed[best]
    for p in pods:
        job = ["#!/bin/bash", "cd /workspace; export PYTHONUNBUFFERED=1", "until grep -q SHARD_DONE /workspace/job_q2.log; do sleep 5; done"]
        job += [l.replace(" 2>&1 |", f" --conds {conds} --tag {tag} 2>&1 |") for l in LF.job_lines(R, assign[p[0]])[1]]
        F.run(p[1], "cat > /workspace/job_q3.sh", stdin_data="\n".join(job + ["echo SHARD_DONE"]) + "\n")
        F.run(p[1], 'tmux kill-session -t q3 2>/dev/null; tmux new-session -d -s q3 "bash /workspace/job_q3.sh 2>&1 | tee /workspace/job_q3.log"')
        print(f"{p[0]}: {len(assign[p[0]])} supplementary base models queued: {assign[p[0]]}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "supp": supplement(sys.argv[2], sys.argv[3], sys.argv[4]); sys.exit()
    act, path = sys.argv[1], sys.argv[2]; R, pods, state, assign, local, remote = plan(path)
    if act == "apply":
        for p in pods:                                     # old pods first: nothing to upload
            if p[2] == "old":
                bad = [f for f, h in remote.get(p[0], {}).items() if f in local and local[f] != h]
                if bad: F.run(p[1], "cd /workspace/runs && rm -f " + " ".join(bad))       # partial files left by the interrupted upload
                print(start_queue(R, p, state[p[0]], assign[p[0]]), flush=True)
        new = sorted((p for p in pods if p[2] == "new" and assign[p[0]]), key=lambda p: -float(p[3]))
        for p in new: start_fed_queue(R, p, assign[p[0]])
        order = [(p, t) for rank in range(max(len(assign[p[0]]) for p in new)) for p in new if rank < len(assign[p[0]]) for t in [assign[p[0]][rank]]]
        with ThreadPoolExecutor(3) as ex:                  # every pod's first task first
            for r in ex.map(lambda pt: feed_task(R, pt[0], pt[1], local), order): print(r, flush=True)
