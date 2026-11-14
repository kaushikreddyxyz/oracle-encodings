#!/usr/bin/env python3
"""Pick, per site and SAE kind, the (spec, steering setting) with the best tuning-set route-changed solve rate
(mean over the 6x6 and 9x9 seed-0 models). Reads the pulled pilot unit logs; writes sae_selection.json + a table."""
import glob, json, re, collections
from pathlib import Path
R = Path(__file__).resolve().parent / "results" / "distance" / "sae_logs"
val = collections.defaultdict(dict); meta = collections.defaultdict(dict)
for f in glob.glob(str(R / "units" / "unit_pilot-*.log")):
    site = "adj" if "-adj-" in f else "gs"
    for line in open(f, errors="replace"):
        m = re.match(r"\[g(\d+) \S+ (\S+)\] .*?(?:route-changed mean|route-changed: solved target maze) ([0-9.]+)", line)
        if m: val[(site, m.group(2))][int(m.group(1))] = float(m.group(3))
        m = re.match(r"\[prep g(\d+)\] SAE (?:L\d+:)?(\S+): n_lat (\d+) heldout (\{.*?\}) loss_recovered ([0-9.\-]+)", line)
        if m:
            h = eval(m.group(4)); meta[(site, m.group(2))][int(m.group(1))] = dict(n_lat=int(m.group(3)), fvu=h["fvu"], L0=h["L0"], dead=h["dead_frac"], loss_rec=float(m.group(5)))
rows = []
for (site, cond), v in val.items():
    if not cond.startswith("sae:") or len(v) < 2: continue
    spec = next(a[1:] for a in cond.split(":")[1:] if a.startswith("s")); kind = spec.split("-")[0]
    rows.append(dict(site=site, kind=kind, spec=spec, cond=cond, g6=v[6], g9=v[9], mean=(v[6] + v[9]) / 2, **{f"{k}_g9": x for k, x in meta.get((site, spec), {}).get(9, {}).items()}))
sel = {}
for site in ("adj", "gs"):
    print(f"== {site}: references", {c: v for (s, c), v in val.items() if s == site and not c.startswith("sae:")})
    for kind in ("relu", "topk", "btopk", "mat", "mp"):
        rs = sorted([r for r in rows if r["site"] == site and r["kind"] == kind], key=lambda r: -r["mean"])
        if not rs: print("  ", kind, "NO DATA"); continue
        sel[f"{site}:{kind}"] = rs[0]
        print(f"  {kind:6s} n_cfg {len(rs):2d} best {rs[0]['cond']:34s} g6 {rs[0]['g6']:.3f} g9 {rs[0]['g9']:.3f} | n_lat {rs[0].get('n_lat_g9')} L0 {rs[0].get('L0_g9', float('nan')):.1f} "
              f"fvu {rs[0].get('fvu_g9', float('nan')):.4f} dead {rs[0].get('dead_g9', float('nan')):.2f} loss_rec {rs[0].get('loss_rec_g9', float('nan')):.3f} | worst {rs[-1]['mean']:.3f}")
json.dump(dict(selection=sel, all=rows), open(R / "sae_selection.json", "w"), indent=1)
