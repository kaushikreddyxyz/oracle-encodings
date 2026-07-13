#!/usr/bin/env python3
"""Laptop-side launcher for the L6/L8 oracle-encoder continuation runs
(2026-07-10 handoff). One H100 pod per layer, warm-started from
kaushikreddyxyz/oracle-encoders layerXX/best_stripped.pt, pushing to
layerXX/cont1/ (originals are never overwritten). L14 is deliberately absent.

Usage:
  python3 cont_launch.py --create            # create both pods (idempotent by name)
  python3 cont_launch.py --deploy all        # stage code+secrets+env, start supervisor
  python3 cont_launch.py --status            # heartbeats + last evals + DONE/HOLD
  python3 cont_launch.py --terminate L6      # manual teardown escape hatch (s7cont-* ONLY)

Boundary math (measured kept-ratio 0.99996, per-shard token counts from HF
tokens_*.npy sizes): L6 resumed at 690.17M gathered tokens = shards
320,352..340 fully + 80.4% of 339 (skipped mid-run, re-queued at tail);
L8 resumed at 701.10M = through 339 fully + 3.7% of 338 (338 re-included
at head). Epoch = all 41 non-val shards; val stays {353,354}.

RunPod quirks handled (all bitten before): ports shift on fresh pods
(re-resolve every round), multiple public :22 mappings (try all), pidfiles
not pgrep, secrets via scp'd files never argv, Cloudflare 403s python UA.
NEVER touches pods whose name doesn't start with s7cont- (the ~21 attrib-w*
pods belong to another task).
"""
import argparse
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
CODE = f"{REPO_ROOT}/oracles"
ATTRIB = f"{REPO_ROOT}/attribution"
STAGE6 = f"{REPO_ROOT}/concept_probes/3_validation/code"
CREATE_POD = str(HOME / ".claude/skills/runpod-spinup/create-pod.sh")
GPU = "NVIDIA H100 80GB HBM3"
HF_REPO = "kaushikreddyxyz/oracle-encoders"
DISK_GB = "400"

BASE_SHARDS = list(range(338, 320, -1)) + [355] + list(range(356, 363))
CFG = {
    "L6": {
        "pod_name": "s7cont-L6", "layer": 6,
        "train_shards": BASE_SHARDS + [339],       # 339 was 80% consumed -> tail
        "max_tokens": "1949522528", "end_step": "71183",
        "anchor_mult": "0.4528",                    # lr_lambda(25200, 300, 47272)
        "resume_path": "layer06/best_stripped.pt",
        "hf_subdir": "layer06/cont1", "wandb_name": "oracle-L6-perlayer-cont1",
    },
    "L8": {
        "pod_name": "s7cont-L8", "layer": 8,
        "train_shards": BASE_SHARDS,                # 338 only 3.7% consumed -> head
        "max_tokens": "1913669162", "end_step": "69876",
        "anchor_mult": "0.4395",                    # lr_lambda(25600, 300, 47272)
        "resume_path": "layer08/best_stripped.pt",
        "hf_subdir": "layer08/cont1", "wandb_name": "oracle-L8-perlayer-cont1",
    },
}
MAX_HOURS = "26"
VAL = [353, 354]


def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {RP_API}", "User-Agent": "curl/8.4.0"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def my_pods():
    return gql("query{myself{pods{id name desiredStatus}}}")["data"]["myself"]["pods"]


def endpoints(pod_id):
    for p in gql("query{myself{pods{id runtime{ports{ip isIpPublic privatePort publicPort}}}}}")["data"]["myself"]["pods"]:
        if p["id"] == pod_id:
            return [(x["ip"], x["publicPort"]) for x in (p["runtime"] or {}).get("ports") or []
                    if x["isIpPublic"] and x["privatePort"] == 22]
    return []


def ssh_try(pod_id, cmd, timeout=120, input_text=None):
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
                               capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def create_pod(name):
    r = subprocess.run(["bash", CREATE_POD, name, GPU, "SECURE", "runpod-torch-v240", "1", DISK_GB],
                       capture_output=True, text=True, timeout=300)
    m = re.search(r"Pod ID:\s+(\S+)", r.stdout)
    if not m:
        raise RuntimeError(f"create {name} failed: {r.stdout[-300:]} {r.stderr[-200:]}")
    return m.group(1)


def write_secret_tmpfiles():
    """Stage secrets as /tmp files for scp (never argv). wandb key from ~/.netrc."""
    hf_tok = (HOME / ".cache/huggingface/token").read_text().strip()
    netrc = (HOME / ".netrc").read_text()
    m = re.search(r"machine api\.wandb\.ai.*?password\s+(\S+)", netrc, re.S)
    if not m:
        raise RuntimeError("wandb key not found in ~/.netrc")
    paths = {}
    for fn, content in ((".s7_hf_token", hf_tok), (".s7_wandb_key", m.group(1)),
                        (".s7_rp_key", RP_API)):
        p = f"/tmp/{fn}"
        with open(p, "w") as f:
            f.write(content + "\n")
        subprocess.run(["chmod", "600", p], check=True)
        paths[fn] = p
    return paths


def env_file(key, pod_id):
    c = CFG[key]
    prefetch = VAL + c["train_shards"]
    return "\n".join([
        f"LAYER={c['layer']}",
        f"TRAIN_SHARDS={','.join(map(str, c['train_shards']))}",
        f"PREFETCH_SHARDS={','.join(map(str, prefetch))}",
        f"MAX_TOKENS={c['max_tokens']}",
        f"MAX_HOURS={MAX_HOURS}",
        f"ANCHOR_MULT={c['anchor_mult']}",
        f"END_STEP={c['end_step']}",
        f"WANDB_NAME={c['wandb_name']}",
        f"HF_SUBDIR={c['hf_subdir']}",
        f"RESUME_PATH={c['resume_path']}",
        f"HF_REPO={HF_REPO}",
        f"POD_ID={pod_id}",
    ]) + "\n"


DEPS_SH = (
    "cd /workspace && pip install -q transformers pyarrow scipy wandb "
    "hf_transfer huggingface_hub tqdm 2>&1 | tail -1 && "
    "python3 -c 'import transformers,pyarrow,scipy,wandb,torch;"
    "print(\"DEPS_OK\", transformers.__version__, torch.__version__)'"
)


def deploy(key):
    c = CFG[key]
    pods = {p["name"]: p["id"] for p in my_pods()}
    pod_id = pods.get(c["pod_name"])
    if pod_id is None:
        raise RuntimeError(f"pod {c['pod_name']} does not exist — run --create first")
    secrets = write_secret_tmpfiles()
    files = [
        (f"{CODE}/train_oracle_perlayer.py", "/workspace/code/train_oracle_perlayer.py"),
        (f"{CODE}/train_encoder.py", "/workspace/code/train_encoder.py"),
        (f"{ATTRIB}/align.py", "/workspace/code/align.py"),
        (f"{ATTRIB}/_align_fallback.py", "/workspace/code/_align_fallback.py"),
        (f"{STAGE6}/nat_common.py", "/workspace/code/nat_common.py"),
        (f"{CODE}/prefetch_shards.py", "/workspace/code/prefetch_shards.py"),
        (f"{CODE}/cont_teardown.py", "/workspace/code/cont_teardown.py"),
        (f"{CODE}/cont_supervise.sh", "/workspace/cont_supervise.sh"),
        (secrets[".s7_hf_token"], "/workspace/.hf_token"),
        (secrets[".s7_wandb_key"], "/workspace/.wandb_key"),
        (secrets[".s7_rp_key"], "/workspace/.rp_key"),
    ]
    for rnd in range(60):
        if ssh_try(pod_id, "echo ready", timeout=25) is None:
            print(f"{key}: pod {pod_id} not sshable yet (round {rnd})")
            time.sleep(15)
            continue
        ssh_try(pod_id, "mkdir -p /workspace/code")
        if not all(scp_try(pod_id, l, r) for l, r in files):
            print(f"{key}: scp incomplete, retrying")
            time.sleep(10)
            continue
        if ssh_try(pod_id, "cat > /workspace/.env", input_text=env_file(key, pod_id)) is None:
            time.sleep(10)
            continue
        ssh_try(pod_id, "chmod 600 /workspace/.hf_token /workspace/.wandb_key /workspace/.rp_key")
        dep = ssh_try(pod_id, DEPS_SH, timeout=600)
        if dep is None or "DEPS_OK" not in dep:
            print(f"{key}: deps install failed: {dep!r}")
            time.sleep(10)
            continue
        # idempotence: never double-start (pidfile check, not pgrep)
        out = ssh_try(pod_id,
            "if [ -f /workspace/supervise.pid ] && kill -0 $(cat /workspace/supervise.pid) 2>/dev/null; "
            "then echo ALREADY_RUNNING; else "
            "nohup bash /workspace/cont_supervise.sh > /workspace/supervise.out 2>&1 & "
            "echo $! > /workspace/supervise.pid; sleep 5; "
            "kill -0 $(cat /workspace/supervise.pid) && echo SUPERVISOR_ALIVE; fi")
        if out and ("SUPERVISOR_ALIVE" in out or "ALREADY_RUNNING" in out):
            return f"{key}: {out.strip()} on {pod_id} ({dep.strip().splitlines()[-1]})"
        print(f"{key}: launch attempt output {out!r}")
        time.sleep(15)
    return f"{key}: DEPLOY FAILED"


def status():
    pods = {p["name"]: (p["id"], p["desiredStatus"]) for p in my_pods()
            if p["name"].startswith("s7cont-")}
    if not pods:
        print("no s7cont-* pods exist")
    for name in sorted(pods):
        pid, st = pods[name]
        layer = CFG[name.split("-")[1]]["layer"]
        out = ssh_try(pid,
            f"cat /workspace/hb_L{layer}.txt 2>/dev/null; echo ---; "
            f"tail -1 /workspace/run/metrics.jsonl 2>/dev/null | head -c 400; echo; echo ---; "
            f"ls /workspace/DONE.json /workspace/HOLD_REASON.txt 2>/dev/null; "
            f"tail -3 /workspace/supervise.log 2>/dev/null; "
            f"tail -2 /workspace/prefetch.log 2>/dev/null; "
            f"df -h /workspace | tail -1; "
            f"tail -2 /workspace/train.log 2>/dev/null", timeout=60)
        print(f"== {name} ({pid}, {st}) ==\n{(out or 'UNREACHABLE').strip()}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--deploy", choices=["L6", "L8", "all"])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", choices=["L6", "L8"])
    args = ap.parse_args()

    if args.create:
        existing = {p["name"]: p["id"] for p in my_pods()}
        for key, c in CFG.items():
            if c["pod_name"] in existing:
                print(f"{key}: {c['pod_name']} already exists -> {existing[c['pod_name']]}")
            else:
                pid = create_pod(c["pod_name"])
                print(f"{key}: created {c['pod_name']} -> {pid}")

    if args.deploy:
        for key in (["L6", "L8"] if args.deploy == "all" else [args.deploy]):
            print(deploy(key))

    if args.status:
        status()

    if args.terminate:
        name = CFG[args.terminate]["pod_name"]
        assert name.startswith("s7cont-")
        pods = {p["name"]: p["id"] for p in my_pods()}
        if name not in pods:
            print(f"{name} not found")
            return
        print(gql('mutation{podTerminate(input:{podId:"%s"})}' % pods[name]))


if __name__ == "__main__":
    main()
