#!/usr/bin/env python3
"""drive.py NAME 'CMD' [TIMEOUT_S] — run a shell CMD on persistent pod NAME via ssh.
Reads .pod_<NAME>.json for host/port. Robust (argv list, no shell-quoting hazards).
CMD can read the HF token at /root/.cache/huggingface/token."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_lib as R

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1]
cmd = sys.argv[2]
timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
ep = json.loads((ROOT / f".pod_{name}.json").read_text())
out = R.ssh_run(ep["host"], ep["port"], cmd, timeout=timeout, check=False)
sys.stdout.write(out)
