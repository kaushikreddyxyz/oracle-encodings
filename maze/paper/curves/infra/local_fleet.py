#!/usr/bin/env python3
"""Rebalance the local experiment over a heterogeneous fleet, at model granularity (one task = one final model).

  python local_fleet.py plan|apply <fleet.txt>      fleet.txt lines: <name> <ip:port> <old|new> <relative speed> [first-job shard]
  python local_fleet.py status|pull <fleet.txt>
`old` pods are already running a shard job (local_deploy.py): whatever model they are evaluating right now is left alone,
every other model of their job is turned into a skip (zero-byte placeholder in /workspace/local_out, never pulled) and a
second queue starts when the running model is done. `new` pods get the full deploy. All second-queue results go to
/workspace/local_out2. Tasks are assigned longest-first to the pod that would finish them earliest.
"""
import os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fleet_ssh as F
import local_deploy as D

COST = {10: 25, 9: 18, 8: 12.4, 7: 8.4, 6: 5.6, 5: 3.6}        # A100 minutes for a base model (12 conditions); a grafted model has 4
OUT2 = "/workspace/local_out2"


def cost(g, k): return (COST[g] if k == "base" else COST[g] / 3) + 1.0


def fleet(path): return [l.split() for l in open(path) if l.strip() and not l.startswith("#")]


def pod_state(name, ep, shard):
    """finished models, the running model, minutes it still needs (from the conditions already printed)."""
    rc, out, _ = F.run(ep, f"grep -a '^\\[g\\|RUN_DONE' /workspace/job_{shard}.log | cut -c1-60", timeout=40)
    done, running, n_cond = set(), None, 0
    for l in out.splitlines():
        if l.startswith("RUN_DONE"):
            g, k, s = re.match(r"RUN_DONE g(\d+)__(\w+)__g\d+-s(\d+)", l).groups(); done.add((int(g), int(s), k)); running = None; n_cond = 0
        else:
            g, k, s = re.match(r"\[g(\d+) (\w+) g\d+-s(\d+)", l).groups(); running = (int(g), int(s), k); n_cond += 1
    left = 0.0
    if running: left = max(0.5, cost(running[0], running[2]) * (1 - n_cond / (12 if running[2] == "base" else 4)))
    return done, running, left


def plan(path):
    R = D.runs(); pods = fleet(path); state = {}
    with ThreadPoolExecutor(16) as ex:
        for p, st in zip(pods, ex.map(lambda p: pod_state(p[0], p[1], p[4]) if p[2] == "old" else (set(), None, 0.0), pods)): state[p[0]] = st
    done = set().union(*[s[0] for s in state.values()]); running = {s[1] for s in state.values() if s[1]}
    tasks = sorted((k for k in R if k not in done and k not in running), key=lambda k: -cost(k[0], k[2]))
    load = {p[0]: (state[p[0]][2] if p[2] == "old" else 6.0) for p in pods}; speed = {p[0]: float(p[3]) for p in pods}; assign = {p[0]: [] for p in pods}
    for t in tasks:
        best = min(load, key=lambda n: load[n] + cost(*t[::2]) / speed[n]); assign[best].append(t); load[best] += cost(*t[::2]) / speed[best]
    for p in pods: print(f"{p[0]:5s} {p[2]} x{p[3]} running={state[p[0]][1]} done={len(state[p[0]][0])} -> {len(assign[p[0]])} tasks, finish ~{load[p[0]]:.0f} min: {assign[p[0]]}")
    print(f"{len(tasks)} tasks, {len(done)} done, {len(running)} running; makespan ~{max(load.values()):.0f} min")
    return R, pods, state, assign


def job_lines(R, tasks):
    files, job = [], []
    for g, s, k in tasks:
        rel = R[(g, s, k)]; files += [f"{rel}/model.final.pt", f"{rel}/config_holder.json"]
        cmd = f"python -u maze/paper/curves/eval_local.py run --out_dir {OUT2} --run_dir /workspace/runs/{rel}"
        if k == "base":
            files.append(f"distance/steer_gs_prep/g{g}-s{s}/stats_sae.pt"); cmd += f" --stats /workspace/runs/distance/steer_gs_prep/g{g}-s{s}/stats_sae.pt"
        else:
            files.append(f"{rel}/injector.final.pt")
        job.append(cmd + " 2>&1 | grep --line-buffered -v -i warn")
    return files, job


def apply_one(R, pod, state, tasks):
    name, ep, kind = pod[0], pod[1], pod[2]; files, job = job_lines(R, tasks); head = ["#!/bin/bash", "cd /workspace; export PYTHONUNBUFFERED=1"]
    try:
        if kind == "new":
            (D.TMP / f"{name}.files").write_text("\n".join(files) + "\n"); (D.TMP / f"{name}.job.sh").write_text("\n".join(head + job + ["echo SHARD_DONE"]) + "\n")
            D.deploy(name, ep, 2.0)
        else:
            shard = pod[4]; done, running, _ = state
            skip = [f"g{g}__{k}__g{g}-s{s}.npz" for (g, s, k) in R if (g, s, k) not in done and (g, s, k) != running]
            F.run(ep, "tmux kill-session -t x" + shard[1:] + " 2>/dev/null; mkdir -p /workspace/local_out; cd /workspace/local_out; for f in " + " ".join(skip) + "; do [ -e $f ] || : > $f; done")
            if files: F.put(ep, "/workspace/runs", str(D.RES), files)
            wait = [f"until grep -aq 'RUN_DONE g{running[0]}__{running[2]}__g{running[0]}-s{running[1]}' /workspace/job_{shard}.log; do sleep 5; done"] if running else []
            F.run(ep, "cat > /workspace/job_q2.sh", stdin_data="\n".join(head + wait + job + ["echo SHARD_DONE"]) + "\n")
            F.run(ep, 'tmux kill-session -t q2 2>/dev/null; tmux new-session -d -s q2 "bash /workspace/job_q2.sh 2>&1 | tee /workspace/job_q2.log"')
        return f"{name}: ok ({len(tasks)} tasks)"
    except BaseException as e:
        return f"{name}: FAILED {type(e).__name__} {e}"


def status(path):
    def st(p):
        try:
            rc, out, _ = F.run(p[1], "echo $(cat /workspace/job_*.log | grep -ac RUN_DONE) done, $(ls /workspace/local_out /workspace/local_out2 2>/dev/null | grep -c npz) files, "
                                     "$(grep -c eval_local /workspace/job_q2.sh /workspace/job.sh 2>/dev/null | tr '\\n' ' '); grep -ah 'Error\\|Traceback' /workspace/job_*.log | head -2; "
                                     "grep -l SHARD_DONE /workspace/job_*.log | tr '\\n' ' '; tail -qn 1 /workspace/job_*.log | cut -c1-70 | tail -n 2", timeout=40)
            return f"{p[0]:5s} " + out.strip().replace("\n", " | ")
        except Exception as e:
            return f"{p[0]:5s} ERR {e}"
    with ThreadPoolExecutor(20) as ex:
        for r in ex.map(st, fleet(path)): print(r)


def pull(path):
    def one(p):
        try:
            F.run(p[1], "find /workspace/local_out /workspace/local_out2 -name '*.npz' -size 0 -delete 2>/dev/null")
            n = 0
            for d in ("local_out", "local_out2"):
                if F.run(p[1], f"test -d /workspace/{d}")[0] == 0: n += F.get(p[1], f"/workspace/{d}", str(D.RES / "local/out"))
            (D.RES / "local/logs" / p[0]).mkdir(parents=True, exist_ok=True)
            for lf in F.run(p[1], "ls /workspace/job_*.log")[1].split(): F.get(p[1], lf, str(D.RES / "local/logs" / p[0]))
            return f"{p[0]}: {n} new files"
        except Exception as e:
            return f"{p[0]}: ERR {e}"
    with ThreadPoolExecutor(20) as ex:
        for r in ex.map(one, fleet(path)): print(r)


if __name__ == "__main__":
    act, path = sys.argv[1], sys.argv[2]
    if act == "plan": plan(path)
    elif act == "apply":
        R, pods, state, assign = plan(path)
        with ThreadPoolExecutor(20) as ex:
            for r in ex.map(lambda p: apply_one(R, p, state[p[0]], assign[p[0]]), pods): print(r, flush=True)
    elif act == "status": status(path)
    elif act == "pull": pull(path)
