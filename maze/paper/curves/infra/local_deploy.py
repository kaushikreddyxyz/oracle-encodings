#!/usr/bin/env python3
"""Deploy / poll / pull one shard of the local (single-decision) experiment over fleet_ssh (paramiko).
The 30 (grid, seed) units (base / fixed / trainable final models each) are bin-packed into N_PODS shards p0..; `pilot` = 5x5 seed 0 + two 10x10 models, 3,000 items, with --verify.

  python local_deploy.py shards                          write shard file lists + job scripts to $LOCAL_TMP
  python local_deploy.py deploy <shard> <ip:port> [--guard_h 3]
  python local_deploy.py status <ip:port>
  python local_deploy.py pull   <ip:port>                -> results/local/out, results/local/logs
Env: MAZETREE=<dir containing maze/core>, LOCAL_TMP=<scratch dir>.
"""
import argparse, glob, os, re, sys, time
from pathlib import Path
import fleet_ssh as F

HERE = Path(__file__).resolve().parent.parent  # curves/ (this script lives in curves/infra/)
REPO = HERE.parents[2]; RES = HERE / "results"
N_PODS = 8
TMP = Path(os.environ.get("LOCAL_TMP", "/tmp/local_agent")); CODE = ["eval_ckpts.py", "eval_local.py"]


def runs():
    """(grid, seed, kind) -> run dir relative to results/ (first copy in pod order, as every earlier eval did)."""
    out = {}
    for p in sorted(glob.glob(str(RES / "runpod-c*/g*-s*/fast-*/model.final.pt"))):
        rel = Path(p).parent.relative_to(RES); g, s = map(int, re.match(r"g(\d+)-s(\d+)", rel.parts[1]).groups())
        kind = "fixed" if "inj-fixed" in rel.name else "trainable" if "inj-trainable" in rel.name else "base"
        out.setdefault((g, s, kind), rel)
    assert len(out) == 90, len(out)
    return out


def write_shards(extra=""):
    R = runs(); TMP.mkdir(parents=True, exist_ok=True)
    cost = {10: 35, 9: 26, 8: 18, 7: 12, 6: 8, 5: 5}; bins = [[0, []] for _ in range(N_PODS)]      # minutes per (grid, seed) unit, from the pilot
    for g, s in sorted(((g, s) for g in cost for s in range(5)), key=lambda u: -cost[u[0]]):          # longest-first bin packing
        b = min(bins, key=lambda b: b[0]); b[0] += cost[g]; b[1] += [(g, s, k) for k in ("base", "fixed", "trainable")]
    shards = {f"p{i}": keys for i, (_, keys) in enumerate(bins)}
    print({f"p{i}": (m, sorted({(g, s) for g, s, _ in keys})) for i, (m, keys) in enumerate(bins)})
    shards["pilot"] = [(5, 0, k) for k in ("base", "fixed", "trainable")] + [(10, 0, "base"), (10, 0, "fixed")]
    for name, keys in shards.items():
        files, job = [], ["#!/bin/bash", "cd /workspace; export PYTHONUNBUFFERED=1"]
        for g, s, k in keys:
            rel = R[(g, s, k)]; files += [f"{rel}/model.final.pt", f"{rel}/config_holder.json"]
            cmd = f"python -u maze/paper/curves/eval_local.py run --run_dir /workspace/runs/{rel}"
            if k == "base":
                files.append(f"distance/steer_gs_prep/g{g}-s{s}/stats_sae.pt"); cmd += f" --stats /workspace/runs/distance/steer_gs_prep/g{g}-s{s}/stats_sae.pt"
            else:
                files.append(f"{rel}/injector.final.pt")
            if name == "pilot": cmd += " --verify True --limit 3000 --out_dir /workspace/local_pilot"
            job.append(cmd + " 2>&1 | grep --line-buffered -v -i warn")
        for f in files: assert (RES / f).exists(), f
        (TMP / f"{name}.files").write_text("\n".join(files) + "\n"); (TMP / f"{name}.job.sh").write_text("\n".join(job + ["echo SHARD_DONE"]) + "\n")
    print("shards:", ", ".join(shards))


def deploy(shard, ep, guard_h):
    tree = os.environ["MAZETREE"]
    assert F.wait(ep), f"{shard}: no sshd"
    F.run(ep, f"nohup sh -c 'sleep {int(guard_h * 3600)}; runpodctl remove pod $RUNPOD_POD_ID' >/dev/null 2>&1 &")      # dead-man switch first
    F.put(ep, "/workspace", str(REPO), ["maze-transformer"], excludes=("__pycache__", ".git", "*.pyc", ".DS_Store", "examples", "notebooks", "tests"))
    F.put(ep, "/workspace", tree, ["maze"])
    F.put(ep, "/workspace/maze/paper/curves", str(HERE), CODE)
    F.run(ep, 'cd /workspace/maze-transformer && nohup sh -c "pip install -q --break-system-packages -e . && pip install -q --break-system-packages fire zanj && touch /workspace/PIP_OK" > /workspace/pip.log 2>&1 &')
    t0 = time.time()
    F.put(ep, "/workspace/runs", str(RES), [l.strip() for l in open(TMP / f"{shard}.files") if l.strip()])
    print(f"{shard}: models uploaded in {time.time() - t0:.0f}s", flush=True)
    F.put(ep, "/workspace/localsets", str(RES / "local/sets"), [f"g{g}.npz" for g in range(5, 11)])
    for g in range(5, 11):
        d = f"/workspace/distsets/g{g}-s4321-n3000"; F.run(ep, f"mkdir -p {d}")
        c = F.connect(ep); s = c.open_sftp(); s.put(str(RES / f"distance/sets/g{g}.npz"), f"{d}/distset_K12_w0.05.npz"); c.close()
    for _ in range(120):
        if F.run(ep, "test -f /workspace/PIP_OK")[0] == 0: break
        time.sleep(5)
    else:
        sys.exit(f"{shard}: pip timeout: {F.run(ep, 'tail -3 /workspace/pip.log')[1]}")
    F.run(ep, "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)")
    F.run(ep, "cat > /workspace/job.sh", stdin_data=(TMP / f"{shard}.job.sh").read_text())
    F.run(ep, f'tmux kill-session -t job 2>/dev/null; tmux new-session -d -s job "bash /workspace/job.sh 2>&1 | tee /workspace/job_{shard}.log"')
    print(f"{shard}: job started on {ep}", flush=True)


def extra(name, ep, first, units):
    """Queue more (grid, seed) units on a pod that is already running shard `first`: upload their models now, run them
    once the first job has printed SHARD_DONE (one job per GPU)."""
    R = runs(); files, job = [], ["#!/bin/bash", "cd /workspace; export PYTHONUNBUFFERED=1", f"until grep -q SHARD_DONE /workspace/job_{first}.log; do sleep 20; done"]
    for u in units.split(","):
        g, s = map(int, u.split("-"))
        for k in ("base", "fixed", "trainable"):
            rel = R[(g, s, k)]; files += [f"{rel}/model.final.pt", f"{rel}/config_holder.json"]
            cmd = f"python -u maze/paper/curves/eval_local.py run --run_dir /workspace/runs/{rel}"
            if k == "base":
                files.append(f"distance/steer_gs_prep/g{g}-s{s}/stats_sae.pt"); cmd += f" --stats /workspace/runs/distance/steer_gs_prep/g{g}-s{s}/stats_sae.pt"
            else:
                files.append(f"{rel}/injector.final.pt")
            job.append(cmd + " 2>&1 | grep --line-buffered -v -i warn")
    t0 = time.time(); F.put(ep, "/workspace/runs", str(RES), files); print(f"{name}: models uploaded in {time.time() - t0:.0f}s", flush=True)
    F.run(ep, f"cat > /workspace/job_{name}.sh", stdin_data="\n".join(job + ["echo SHARD_DONE"]) + "\n")
    F.run(ep, f'tmux new-session -d -s {name} "bash /workspace/job_{name}.sh 2>&1 | tee /workspace/job_{name}.log"')
    print(f"{name}: queued on {ep} behind {first}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("act"); ap.add_argument("args", nargs="*"); ap.add_argument("--guard_h", type=float, default=3)
    a = ap.parse_args()
    if a.act == "shards": write_shards()
    elif a.act == "deploy": deploy(a.args[0], a.args[1], a.guard_h)
    elif a.act == "extra": extra(*a.args[:4])
    elif a.act == "status":
        print(F.run(a.args[0], "grep -c RUN_DONE /workspace/job_*.log; grep -h 'SHARD_DONE\\|Error\\|Traceback' /workspace/job_*.log | head; tail -n 4 /workspace/job_*.log | cut -c1-260; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")[1])
    elif a.act == "pull":
        ep = a.args[0]; sub = a.args[1] if len(a.args) > 1 else "local_out"
        print(F.get(ep, f"/workspace/{sub}", str(RES / "local" / ("out" if sub == "local_out" else sub))), "result files")
        (RES / "local/logs").mkdir(parents=True, exist_ok=True)
        rc, out, _ = F.run(ep, "ls /workspace/job_*.log")
        for lf in out.split(): F.get(ep, lf, str(RES / "local/logs"))
