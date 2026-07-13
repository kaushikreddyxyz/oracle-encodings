#!/usr/bin/env python3
"""Laptop-side launcher for the climbmix-scored attribution waves.

Usage (wave 1):
  python3 launch_attrib_wave.py --seed-repos
  python3 launch_attrib_wave.py --shards 0-61  --pod-prefix attrib-w1 --launch first
  python3 launch_attrib_wave.py --shards 0-61  --pod-prefix attrib-w1 --launch rest
  python3 launch_attrib_wave.py --pod-prefix attrib-w1 --status
Wave 2: --shards 62-123 --pod-prefix attrib-w2 (same flow; repos already seeded)
Wave 3: --shards 124-184 --pod-prefix attrib-w3

Handles the known RunPod/HF quirks: ports shift on fresh pods (re-resolve
every round), pods may expose MULTIPLE public :22 mappings (try all), no
pgrep guards (pidfiles), secrets staged by scp never argv. Pods self-
terminate ONLY on /workspace/DONE_SCORE.txt (clean completion); HOLD leaves
the pod alive as evidence.
"""
import argparse
import concurrent.futures as cf
import io
import json
import pathlib
import re
import subprocess
import time
import urllib.request

HOME = pathlib.Path.home()
KEY_SSH = str(HOME / ".runpod/ssh/runpodctl-ssh-key")
RP_API = re.search(r"apikey\s*=\s*'([^']+)'", (HOME / ".runpod/config.toml").read_text()).group(1)
REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
CODE = f"{REPO_ROOT}/attribution"
OUT = f"{REPO_ROOT}/attribution/out"
CREATE_POD = str(HOME / ".claude/skills/runpod-spinup/create-pod.sh")
U = "kaushikreddyxyz"
K = 54
DET_REPOS = [f"{U}/climbmix-scored"] + [f"{U}/climbmix-scored-overflow" + ("" if i == 1 else f"-{i}")
                                          for i in range(1, 8)]
DET_PER_REPO, N_ALL = 25, 185   # full-coverage shards ~10-10.8GB -> 25/repo
GPU = "NVIDIA H100 80GB HBM3"


def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {RP_API}", "User-Agent": "curl/8.4.0"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def endpoints(pod_id):
    """ALL public :22 mappings for a pod (they shift; some are dead)."""
    for p in gql("query{myself{pods{id runtime{ports{ip isIpPublic privatePort publicPort}}}}}")["data"]["myself"]["pods"]:
        if p["id"] == pod_id:
            return [(x["ip"], x["publicPort"]) for x in (p["runtime"] or {}).get("ports") or []
                    if x["isIpPublic"] and x["privatePort"] == 22]
    return []


def ssh_try(pod_id, cmd, timeout=120, input_text=None):
    """Run cmd on the first responsive endpoint. Returns stdout or None."""
    for ip, port in endpoints(pod_id):
        try:
            base = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                    "-o", "BatchMode=yes", "-i", KEY_SSH, "-p", str(port), f"root@{ip}"]
            if input_text is None:
                r = subprocess.run(base + ["-n", cmd], capture_output=True, text=True, timeout=timeout)
            else:
                r = subprocess.run(base + [cmd], input=input_text, capture_output=True,
                                   text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout
        except Exception:  # noqa: BLE001
            continue
    return None


def scp_try(pod_id, local, remote):
    for ip, port in endpoints(pod_id):
        try:
            r = subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
                                "-i", KEY_SSH, "-P", str(port), local, f"root@{ip}:{remote}"],
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def parse_range(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def partition(sids, n):
    """Contiguous in-order chunks, larger chunks first (9,9,...,8)."""
    base, extra = divmod(len(sids), n)
    out, i = [], 0
    for j in range(n):
        size = base + (1 if j < extra else 0)
        out.append(sids[i:i + size])
        i += size
    return out


# ------------------------------------------------------------------ seeding
def seed_repos():
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    meta = {}
    for f in ("quant.json", "corpus_stats.json", "columns.json"):
        meta[f] = open(hf_hub_download(f"{U}/corpus-scores", f, repo_type="dataset",
                                        local_dir="/tmp/attrib_seed"), "rb").read()
    assignment = {"det": {str(s): DET_REPOS[s // DET_PER_REPO] for s in range(N_ALL)},
                  "det_per_repo": DET_PER_REPO,
                  "spec": "full-coverage-det-only-2026-07-10"}
    fm = ("---\npretty_name: {pn}\ntags:\n- interpretability\n- probing\n- gemma-2\n"
          "size_categories:\n- 1B<n<10B\n---\n")

    def det_readme(i):
        lo, hi = i * DET_PER_REPO, min((i + 1) * DET_PER_REPO - 1, N_ALL - 1)
        return fm.format(pn=f"ClimbMix Scored — Concept-Probe Detections (shards {lo}-{hi})") + f"""# {DET_REPOS[i].split('/')[-1]}

Concept-probe detection scores for **ClimbMix-shuffle shards {lo}-{hi}** (nanochat
training data), gemma-2-2b layers 6/8/14 co-located per token, **FULL TOKEN
COVERAGE**: every parquet row appears in `docs_<sid>.jsonl`, no min-length
filter, no truncation — docs longer than 2048 gemma tokens are scored in
consecutive non-overlapping 2048-token windows (window-boundary context
truncation is an accepted artifact), so `sum(n)` equals the shard's total
gemma token count. Probes and quantization/standardization are FROZEN from
[corpus-scores](https://huggingface.co/datasets/{U}/corpus-scores) (calibrated
once on shard 320 — never refit), so the two datasets are directly comparable
(note corpus-scores itself is first-2048-window only). Shard->repo map:
`assignment.json` ({DET_PER_REPO} shards/repo, strictly in order across
{len(DET_REPOS)} repos). No DoM scores in this dataset family.

- `scores_<sid>.npy` int8 `[n_tokens, 3, 54]` — axis1: **0=L6, 1=L8, 2=L14**;
  axis2 = concept per `columns.json`. Dequant: `score = int8*scale[l][c]+zero[l][c]`.
- `tokens_<sid>.npy` int32 — full BOS-free gemma ids, docs in parquet order
- `docs_<sid>.jsonl` {{doc, start, n}} — n = full doc token count (0 allowed)
- Scoring: gemma-2-2b bf16 EAGER attention; probes from
  [concept-probes-gemma2-2b](https://huggingface.co/{U}/concept-probes-gemma2-2b) gold_probes.
"""

    for i, rid in enumerate(DET_REPOS):
        api.create_repo(rid, repo_type="dataset", exist_ok=True)
        ops = [CommitOperationAdd("README.md", io.BytesIO(det_readme(i).encode())),
               CommitOperationAdd("columns.json", io.BytesIO(meta["columns.json"])),
               CommitOperationAdd("quant.json", io.BytesIO(meta["quant.json"])),
               CommitOperationAdd("corpus_stats.json", io.BytesIO(meta["corpus_stats.json"])),
               CommitOperationAdd("assignment.json", io.BytesIO(json.dumps(assignment, indent=1).encode()))]
        api.create_commit(repo_id=rid, repo_type="dataset", operations=ops,
                          commit_message="seed: README + frozen metadata + assignment (full-coverage spec)")
        print(f"seeded {rid}")


# ------------------------------------------------------------------- pods
def create_pod(name):
    r = subprocess.run(["bash", CREATE_POD, name, GPU, "SECURE", "runpod-torch-v240", "1", "200"],
                       capture_output=True, text=True, timeout=300)
    m = re.search(r"Pod ID:\s+(\S+)", r.stdout)
    if not m:
        raise RuntimeError(f"create {name} failed: {r.stdout[-300:]} {r.stderr[-200:]}")
    return m.group(1)


LAUNCH_SH = r'''
set -u
cd /workspace
pip install -q transformers pyarrow 2>&1 | tail -1
python3 - <<'CHK'
import transformers, pyarrow, torch, numpy
print("DEPS_OK", transformers.__version__, torch.__version__)
CHK
export HF_TOKEN=$(cat /workspace/.hf_token)
export ONLY_SHARDS="__SHARDS__" BATCH_SIZE=56 __VALIDATE__
nohup python3 -u /workspace/code/score_climbmix_stacked.py > /workspace/score.nohup 2>&1 &
echo $! > /workspace/score.pid
cat > /workspace/selfterm.sh <<'EOF'
#!/bin/bash
while true; do
  if [ -f /workspace/DONE_SCORE.txt ]; then
    sleep 120
    KEY=$(cat /workspace/.rp_key)
    curl -s -A "curl/8.4.0" -H "Content-Type: application/json" \
      -d '{"query":"mutation{podTerminate(input:{podId:\"__POD__\"})}"}' \
      "https://api.runpod.io/graphql?api_key=$KEY" >> /workspace/selfterm.log 2>&1
    exit 0
  fi
  sleep 60
done
EOF
chmod +x /workspace/selfterm.sh
nohup bash /workspace/selfterm.sh > /dev/null 2>&1 &
echo $! > /workspace/selfterm.pid
sleep 8
kill -0 $(cat /workspace/score.pid) && echo SCORER_ALIVE || { echo SCORER_DEAD; tail -5 /workspace/score.nohup; }
'''


def stage_and_launch(pod_id, shards, validate_first):
    files = [(f"{CODE}/score_corpus.py", "/workspace/code/score_corpus.py"),
             (f"{CODE}/score_climbmix_stacked.py", "/workspace/code/score_climbmix_stacked.py"),
             (f"{OUT}/probe_set.json", "/workspace/meta/probe_set.json"),
             (f"{OUT}/probe_set_arrays.npz", "/workspace/meta/probe_set_arrays.npz"),
             (f"{OUT}/quant.json", "/workspace/meta/quant216.json"),
             ("/tmp/.fleet_hf_token", "/workspace/.hf_token"),
             ("/tmp/.fleet_rp_key", "/workspace/.rp_key")]
    for rnd in range(40):
        if ssh_try(pod_id, "echo ready", timeout=25) is None:
            time.sleep(15)
            continue
        ssh_try(pod_id, "mkdir -p /workspace/code /workspace/meta /workspace/scores")
        if not all(scp_try(pod_id, l, r) for l, r in files):
            time.sleep(10)
            continue
        ssh_try(pod_id, "chmod 600 /workspace/.hf_token /workspace/.rp_key")
        script = (LAUNCH_SH.replace("__SHARDS__", ",".join(map(str, shards)))
                  .replace("__POD__", pod_id)
                  .replace("__VALIDATE__", "VALIDATE_FIRST=1" if validate_first else ""))
        if ssh_try(pod_id, "cat > /workspace/launch_all.sh", input_text=script) is None:
            time.sleep(10)
            continue
        out = ssh_try(pod_id, "bash /workspace/launch_all.sh", timeout=900)
        if out and "SCORER_ALIVE" in out:
            return f"{pod_id}: LAUNCHED shards {shards[0]}-{shards[-1]} ({out.splitlines()[-1]})"
        print(f"{pod_id} rnd {rnd}: launch output: {out!r}")
        time.sleep(15)
    return f"{pod_id}: LAUNCH FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-repos", action="store_true")
    ap.add_argument("--shards", help="e.g. 0-61")
    ap.add_argument("--pod-prefix", default="attrib-w1")
    ap.add_argument("--n-pods", type=int, default=7)
    ap.add_argument("--launch", choices=["first", "rest", "all"])
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.seed_repos:
        seed_repos()
        return

    if args.status:
        pods = {p["name"]: p["id"] for p in gql(
            "query{myself{pods{id name}}}")["data"]["myself"]["pods"]
            if p["name"].startswith(args.pod_prefix)}
        for name in sorted(pods):
            out = ssh_try(pods[name],
                          "cat /workspace/hb_score.txt 2>/dev/null; echo; "
                          "grep -c ': OK ->' /workspace/score.log 2>/dev/null; "
                          "ls /workspace/HOLD_SCORE.txt /workspace/DONE_SCORE.txt 2>/dev/null; "
                          "tail -2 /workspace/score.log 2>/dev/null", timeout=60)
            print(f"== {name} ({pods[name]}) ==\n{(out or 'UNREACHABLE').strip()}")
        return

    sids = parse_range(args.shards)
    parts = partition(sids, args.n_pods)
    existing = {p["name"]: p["id"] for p in gql(
        "query{myself{pods{id name}}}")["data"]["myself"]["pods"]}
    todo = (range(0, 1) if args.launch == "first"
            else range(1, args.n_pods) if args.launch == "rest" else range(args.n_pods))
    results = []
    with cf.ThreadPoolExecutor(max_workers=7) as ex:
        futs = {}
        for j in todo:
            name = f"{args.pod_prefix}-{j}"
            pid = existing.get(name) or create_pod(name)
            print(f"{name}: pod {pid}, shards {parts[j][0]}-{parts[j][-1]}")
            futs[ex.submit(stage_and_launch, pid, parts[j], validate_first=(j == 0))] = name
        for f in cf.as_completed(futs):
            results.append(f"{futs[f]}: {f.result()}")
    print("\n".join(results))


if __name__ == "__main__":
    main()
