#!/usr/bin/env python3
"""parse_pods.py <saved mcp list-pods output> [name-prefix] -> lines `name id ip:port gpu cost` for pods with a direct SSH endpoint."""
import json, sys
d = json.load(open(sys.argv[1])); pre = sys.argv[2] if len(sys.argv) > 2 else ""
for p in d["pods"]:
    if not p["name"].startswith(pre): continue
    s = (p.get("ssh") or {}).get("direct")
    print(p["name"], p["id"], f"{s['host']}:{s['port']}" if s else "PENDING", p["gpu"]["id"].replace(" ", "_"), p["cost"])
