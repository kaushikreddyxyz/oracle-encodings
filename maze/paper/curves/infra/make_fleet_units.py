#!/usr/bin/env python3
"""Fleet units from the pilot selection: one unit per (base model, site); 5 SAE kinds each, the selected spec + steering setting."""
import glob, json, os
from pathlib import Path
H = Path(__file__).resolve().parent.parent  # curves/ (this script lives in curves/infra/)
R = H / "results"
sel = json.load(open(R / "distance/sae_logs/sae_selection.json"))["selection"]
seen = {}
for done in sorted(glob.glob(str(R / "runpod-*/g*-s*/fast-baseline*/DONE"))):
    rd = os.path.relpath(os.path.dirname(done), R); g, s = rd.split("/")[1][1:].split("-s")
    seen.setdefault((int(g), int(s)), rd)
assert len(seen) == 30
units = []
for (g, s), rd in sorted(seen.items()):
    files = [f"{rd}/model.final.pt", f"{rd}/config_holder.json"]; d = f"/workspace/runs/{rd}"
    for site, script, extra in (("adj", "eval_steering.py", ""), ("gs", "eval_steering_graftsite.py", "--sae_layers 6 --out_dir /workspace/steer_out")):
        ks = [sel[f"{site}:{k}"] for k in ("relu", "topk", "btopk", "mat", "mp") if f"{site}:{k}" in sel]
        specs = ",".join(k["spec"] for k in ks); conds = ",".join(k["cond"] for k in ks)
        units.append(dict(id=f"fleet-{site}-g{g}-s{s}", weight=g * 10 + (1 if site == "adj" else 0), files=files,
                          cmd=f"python -u maze/paper/curves/{script} run --run_dir {d} --tag _f --saes '{specs}' --conds '{conds}' --batch 3000 {extra}"))
json.dump(units, open(R / "distance/sae_logs/fleet_units.json", "w"), indent=1); print(len(units), "units;", units[0]["cmd"][-260:])
