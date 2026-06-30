"""
runpod_lib.py — RunPod pod lifecycle + budget/spend guards for the overnight run.

Thin wrapper over `runpodctl` (v2.6) and direct SSH (key ~/.ssh/runpod). Every pod is
created with a hard `--terminate-after` backstop so it can NEVER outlive the budget window
even if the orchestrator process dies. Never prints secrets; `.env` is moved by scp only.
"""
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

SSH_KEY = os.path.expanduser("~/.ssh/runpod")
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]
# MUST create from a TEMPLATE, not a bare --image: official runpod/pytorch images do NOT
# auto-start sshd; only the template's start script does (verified the hard way — a bare
# --image pod sits RUNNING with runtime=null forever). v280 = torch 2.8 / cu12.8 / ub24.04,
# recent enough for vLLM + transformers.
DEFAULT_TEMPLATE = "runpod-torch-v280"


def sh(cmd, check=True, timeout=120, capture=True):
    """Run a shell command (list or str). Returns stdout (str)."""
    r = subprocess.run(cmd, shell=isinstance(cmd, str), check=False,
                       capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\nSTDERR: {r.stderr[:500]}")
    return r.stdout if capture else ""


def _json(cmd, timeout=60):
    out = sh(cmd, timeout=timeout)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


# --------------------------------------------------------------------------- #
# Account / spend
# --------------------------------------------------------------------------- #
def balance():
    """{'clientBalance':float,'currentSpendPerHr':float,'spendLimit':float}"""
    b = _json(["runpodctl", "user", "-o", "json"])
    if not isinstance(b, dict):
        raise RuntimeError(f"unexpected `runpodctl user` output: {str(b)[:200]}")
    return b


def spent_since(start_balance):
    b = balance()
    return round(start_balance - float(b["clientBalance"]), 4), float(b.get("currentSpendPerHr", 0.0))


def projected_spend(start_balance, hours_remaining):
    spent, rate = spent_since(start_balance)
    return spent + rate * max(0.0, hours_remaining), spent, rate


def list_pods():
    p = _json(["runpodctl", "pod", "list", "-o", "json"])
    return p if isinstance(p, list) else []


def teardown_all():
    """Emergency: delete every pod. Returns list of deleted ids."""
    ids = []
    for p in list_pods():
        pid = p.get("id") or p.get("podId")
        if pid:
            try:
                sh(["runpodctl", "pod", "delete", pid], check=False)
                ids.append(pid)
            except Exception:
                pass
    return ids


# --------------------------------------------------------------------------- #
# GPU selection
# --------------------------------------------------------------------------- #
# `runpodctl gpu list` has NO price field -> approximate secure-cloud $/hr per GPU
# (per the runpod-spinup skill table; used only for cost-ranking, the real cost is
# confirmed from the create response). Match by substring of the `gpuId`.
PRICE = {
    "RTX 4090": 0.69, "RTX 5090": 0.94, "L40S": 0.99, "RTX A6000": 0.79,
    "A100 80GB PCIe": 1.39, "A100-SXM4-80GB": 1.69, "A100 80GB": 1.39,
    "H100 80GB HBM3": 2.39, "H100 NVL": 2.59, "H100 PCIe": 2.39, "H100": 2.69,
    "H200": 3.59, "B200": 4.99, "MI300X": 2.49,
}


def _price_of(gpu_id):
    for k, v in sorted(PRICE.items(), key=lambda kv: -len(kv[0])):  # longest match first
        if k.lower() in gpu_id.lower():
            return v
    return 999.0


def gpu_list():
    g = _json(["runpodctl", "gpu", "list", "-o", "json"])
    return g if isinstance(g, list) else []


def pick_gpu(min_vram_gb, prefer=None):
    """Return (gpu_id, est_price) for the cheapest AVAILABLE gpu with >= min_vram_gb.
    `prefer` = ordered substrings to try first (else cheapest by PRICE table)."""
    cands = []
    for g in gpu_list():
        gid = g.get("gpuId") or g.get("displayName") or ""
        vram = g.get("memoryInGb") or 0
        if g.get("available", True) and vram >= min_vram_gb and g.get("stockStatus") != "Unavailable":
            cands.append((gid, _price_of(gid), vram))
    if prefer:
        for pat in prefer:
            for gid, price, _ in sorted(cands, key=lambda x: x[1]):
                if pat.lower() in gid.lower():
                    return (gid, price)
    if cands:
        gid, price, _ = sorted(cands, key=lambda x: x[1])[0]
        return (gid, price)
    return (None, None)


# --------------------------------------------------------------------------- #
# Pod lifecycle
# --------------------------------------------------------------------------- #
def create_pod(name, gpu_id, gpu_count=1, disk_gb=80, cloud="SECURE",
               template_id=DEFAULT_TEMPLATE, image=None,
               terminate_after_hours=6, ports="8000/http,22/tcp"):
    """Create a pod with a hard auto-terminate backstop. Returns pod dict (incl id).
    Uses a TEMPLATE by default so sshd actually starts (see DEFAULT_TEMPLATE note)."""
    term = (datetime.now(timezone.utc) + timedelta(hours=terminate_after_hours)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = ["runpodctl", "pod", "create", "--name", name, "--gpu-id", gpu_id,
           "--gpu-count", str(gpu_count), "--cloud-type", cloud,
           "--container-disk-in-gb", str(disk_gb),
           "--ports", ports, "--terminate-after", term, "-o", "json"]
    if image:                       # explicit image override (rare)
        cmd += ["--image", image]
    else:                           # normal path: template (starts sshd)
        cmd += ["--template-id", template_id]
    info = _json(cmd, timeout=180)
    if isinstance(info, dict):
        info["_terminate_after"] = term
    return info


def pod_get(pid):
    return _json(["runpodctl", "pod", "get", pid, "-o", "json"], timeout=60)


def ssh_endpoint(pid):
    """Return (host, port) for direct SSH, parsed from pod get."""
    info = pod_get(pid)
    if not isinstance(info, dict):
        return None, "22"
    # try the explicit ssh_command first
    sc = (info.get("ssh", {}) or {}).get("ssh_command")
    if sc:
        toks = sc.split()
        host, port = None, "22"
        for i, t in enumerate(toks):
            if "@" in t:
                host = t.split("@")[-1]
            if t == "-p" and i + 1 < len(toks):
                port = toks[i + 1]
        if host:
            return host, port
    # fallback: publicIp + ports mapping
    ip = info.get("publicIp") or info.get("ip")
    port = "22"
    for pm in (info.get("portMappings") or info.get("ports") or []):
        if str(pm.get("privatePort")) == "22" or pm.get("type") == "tcp":
            port = str(pm.get("publicPort") or pm.get("port") or port)
    return ip, port


def wait_ssh(pid, timeout=600):
    """Poll direct SSH until the pod answers. Returns (host, port)."""
    t0 = time.time()
    host = port = None
    while time.time() - t0 < timeout:
        try:
            host, port = ssh_endpoint(pid)
            if host:
                r = subprocess.run(["ssh", *SSH_OPTS, "-i", SSH_KEY, "-p", str(port),
                                    f"root@{host}", "echo ready"],
                                   capture_output=True, text=True, timeout=20)
                if "ready" in r.stdout:
                    return host, port
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError(f"pod {pid} ssh not up after {timeout}s (host={host} port={port})")


def ssh_run(host, port, cmd, timeout=3600, check=True):
    r = subprocess.run(["ssh", *SSH_OPTS, "-i", SSH_KEY, "-p", str(port),
                        f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh cmd failed ({r.returncode}): {cmd[:120]}\nSTDERR:{r.stderr[:500]}")
    return r.stdout


def scp_to(host, port, local, remote):
    """Copy a local file to the pod. Used to move .env securely (never printed)."""
    subprocess.run(["scp", *SSH_OPTS, "-i", SSH_KEY, "-P", str(port),
                    local, f"root@{host}:{remote}"], check=True, capture_output=True, text=True)


def scp_from(host, port, remote, local):
    subprocess.run(["scp", *SSH_OPTS, "-i", SSH_KEY, "-P", str(port),
                    f"root@{host}:{remote}", local], check=True, capture_output=True, text=True)


def delete_pod(pid):
    sh(["runpodctl", "pod", "delete", pid], check=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        b = balance()
        print(json.dumps({"balance": b, "pods": [p.get("id") for p in list_pods()]}, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "gpus":
        print(json.dumps(gpu_list(), indent=2)[:3000])
    else:
        print("usage: runpod_lib.py [status|gpus]")
