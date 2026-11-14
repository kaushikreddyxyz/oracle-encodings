#!/usr/bin/env python3
"""Print per-checkpoint retention / autopilot / steering for every run under a results root."""
import glob, json, sys
root = sys.argv[1] if len(sys.argv) > 1 else "."
for f in sorted(glob.glob(f"{root}/s*/*/eval_fade_traj/metrics.json")):
    rows = json.load(open(f))["rows"]
    print(f.split("/")[-3].replace("_jirpyish-d256-l12_dfs-g6-n2000000", ""))
    for r in rows:
        o = r["retention"]["origin"]; c = r["conditions"]
        nan = float("nan")
        best = max(((v.get("route_changed") or 0), k) for k, v in c.items() if k.startswith("delta")) if any(k.startswith("delta") for k in c) else (nan, "-")
        print(f"  it {r['iteration']:>5} gain {o['gain']:+.3f} cos {o['cos_gen']:+.3f} zero-fit {o['acc_zero_fit']:.3f} probe {o['acc_probe']:.3f}"
              f" | off {c['off'].get('same_maze', nan):.3f} on {c.get('on', {}).get('same_maze', nan):.3f}/{c.get('on', {}).get('route_changed', nan):.3f}"
              f" best delta {best[0]:.3f} ({best[1]})")
