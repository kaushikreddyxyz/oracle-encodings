#!/usr/bin/env python3
"""
pod_up.py NAME MIN_VRAM [PREFER_CSV] — create a PERSISTENT pod, wait for ssh, provision
it (ship code+prompts+concepts+config+data, .env, and the verified HF cached token), and
save its endpoint to overnight_run/.pod_<NAME>.json so later steps can ssh in.

Env: DISK (gb, default 120), TERM (auto-terminate hours, default 7).
Teardown later with:  python pod_down.py NAME   (or runpodctl pod delete <id>)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpod_lib as R

ROOT = Path(__file__).resolve().parent.parent

name = sys.argv[1]
vram = int(sys.argv[2])
prefer = sys.argv[3].split(",") if len(sys.argv) > 3 else None
disk = int(os.environ.get("DISK", "120"))
term = int(os.environ.get("TERM", "7"))

gid, price = R.pick_gpu(vram, prefer=prefer)
if not gid:
    print(f"NO GPU >= {vram}GB available"); sys.exit(1)
print(f"[pod_up:{name}] creating {gid} ~${price}/hr disk={disk}GB term+{term}h", flush=True)
info = R.create_pod(name, gid, disk_gb=disk, terminate_after_hours=term)
pid = info.get("id") if isinstance(info, dict) else None
if not pid:
    print(f"[pod_up:{name}] FAIL create: {str(info)[:300]}"); sys.exit(1)
print(f"[pod_up:{name}] pod {pid} created; waiting for ssh (up to 10m)...", flush=True)
host, port = R.wait_ssh(pid, timeout=600)
print(f"[pod_up:{name}] SSH UP {host}:{port}", flush=True)

# provision
R.ssh_run(host, port, "mkdir -p /workspace/run /root/.cache/huggingface", timeout=60)
subprocess.run(["tar", "czf", "/tmp/payload.tgz", "-C", str(ROOT),
                "code", "prompts", "concepts.py", "data"], check=True, capture_output=True)
R.scp_to(host, port, "/tmp/payload.tgz", "/workspace/run/payload.tgz")
R.ssh_run(host, port, "cd /workspace/run && tar xzf payload.tgz", timeout=120)
env_path = ROOT.parent / ".env"
if env_path.exists():
    R.scp_to(host, port, str(env_path), "/workspace/run/.env")
hf_tok = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(hf_tok):
    R.scp_to(host, port, hf_tok, "/root/.cache/huggingface/token")
# sanity: confirm GPU + HF auth on the pod
chk = R.ssh_run(host, port,
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
                "export HF_TOKEN=$(cat /root/.cache/huggingface/token); "
                "python -c 'from huggingface_hub import whoami; print(\"hf:\", whoami(token=open(\"/root/.cache/huggingface/token\").read().strip())[\"name\"])' 2>&1 | tail -1",
                timeout=120, check=False)
print(f"[pod_up:{name}] pod checks:\n{chk}", flush=True)

out = {"pid": pid, "host": host, "port": port, "gid": gid, "price": price}
(ROOT / f".pod_{name}.json").write_text(json.dumps(out))
print(f"[pod_up:{name}] POD UP & PROVISIONED -> {json.dumps(out)}", flush=True)
