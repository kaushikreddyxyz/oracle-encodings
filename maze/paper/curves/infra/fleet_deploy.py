#!/usr/bin/env python3
"""Deploy one shard to a pod over fleet_ssh (paramiko): code, pip, the shard's base models, the standard eval sets, a
pod-side auto-delete guard, and (optionally) a tmux job. Python twin of steer_deploy_one.sh for sessions without `ssh`.

  MAZETREE=<dir with maze/> SHARDS=<dir with <shard>.files/.dirs> \
  python fleet_deploy.py <shard> <ip:port> [--job '<bash lines, {dir} expands to each run dir of the shard>'] [--guard_h 4]
"""
import argparse, os, sys, time
from pathlib import Path
import fleet_ssh as F

HERE = Path(__file__).resolve().parent.parent  # curves/ (this script lives in curves/infra/)
REPO = HERE.parents[2]
CODE = ["eval_ckpts.py", "eval_distance.py", "maze_distance.py", "eval_steering.py", "eval_steering_graftsite.py", "sae_lib.py"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("shard"); ap.add_argument("ep"); ap.add_argument("--job", default=""); ap.add_argument("--guard_h", type=float, default=4)
    ap.add_argument("--extra_code", default="", help="comma-separated extra files under maze/paper/curves to upload")
    a = ap.parse_args(); ep = a.ep; shards = Path(os.environ["SHARDS"]); tree = os.environ["MAZETREE"]
    assert F.wait(ep), f"{a.shard}: no sshd"
    F.put(ep, "/workspace", str(REPO), ["maze-transformer"], excludes=("__pycache__", ".git", "*.pyc", ".DS_Store", "examples", "notebooks", "tests"))
    F.put(ep, "/workspace", tree, ["maze"])
    F.put(ep, "/workspace/maze/paper/curves", str(HERE), CODE + [x for x in a.extra_code.split(",") if x])
    F.run(ep, 'cd /workspace/maze-transformer && nohup sh -c "pip install -q --break-system-packages -e . && pip install -q --break-system-packages fire zanj && touch /workspace/PIP_OK" > /workspace/pip.log 2>&1 &')
    F.run(ep, f"nohup sh -c 'sleep {int(a.guard_h * 3600)}; runpodctl remove pod $RUNPOD_POD_ID' >/dev/null 2>&1 &")      # dead-man switch
    F.put(ep, "/workspace/runs", str(HERE / "results"), [l.strip() for l in open(shards / f"{a.shard}.files") if l.strip()])
    for g in range(5, 11):
        d = f"/workspace/distsets/g{g}-s4321-n3000"; F.run(ep, f"mkdir -p {d}")
        c = F.connect(ep); s = c.open_sftp(); s.put(str(HERE / f"results/distance/sets/g{g}.npz"), f"{d}/distset_K12_w0.05.npz"); c.close()
    for _ in range(120):
        if F.run(ep, "test -f /workspace/PIP_OK")[0] == 0: break
        time.sleep(5)
    else:
        sys.exit(f"{a.shard}: pip timeout: {F.run(ep, 'tail -3 /workspace/pip.log')[1]}")
    F.run(ep, "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)")
    if a.job:
        dirs = [l.strip() for l in open(shards / f"{a.shard}.dirs") if l.strip()]
        lines = ["#!/bin/bash", "cd /workspace; export PYTHONUNBUFFERED=1"] + [a.job.replace("{dir}", d) for d in dirs] + ["echo SHARD_DONE"]
        F.run(ep, "cat > /workspace/job.sh", stdin_data="\n".join(lines) + "\n")
        F.run(ep, 'tmux kill-session -t job 2>/dev/null; tmux new-session -d -s job "bash /workspace/job.sh 2>&1 | tee /workspace/job.log"')
    print(f"{a.shard}: ready on {ep}", flush=True)


if __name__ == "__main__":
    main()
