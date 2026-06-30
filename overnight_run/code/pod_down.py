#!/usr/bin/env python3
"""pod_down.py NAME — delete the persistent pod recorded in .pod_<NAME>.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_lib as R

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1]
f = ROOT / f".pod_{name}.json"
if f.exists():
    pid = json.loads(f.read_text())["pid"]
    R.delete_pod(pid)
    f.unlink()
    print(f"deleted pod {pid} ({name})")
else:
    print(f"no .pod_{name}.json; listing live pods:")
    print([(p.get("id"), p.get("name")) for p in R.list_pods()])
