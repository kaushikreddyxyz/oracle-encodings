#!/usr/bin/env python3
"""Stage-2 SAE shortlist per maze size, from the stage-1 sweeps of the seed-0 models (both grafts pooled).
Uses the TRUE alignment (steelman / oracle) and the fair label-correlation criterion; recorded in jobs/shortlist.json.
  per graft: top-3 (spec, variant) by steelman score among ctr/raw, top-2 among the coding-aware open/closed variants,
             top-2 by the fair criterion, and the best config of each cheap kind (relu, topk, btopk, mat).
Matching pursuit is not re-run in stage 2 (unrolled MP costs k encoder passes per step); its stage-1 numbers stay in sae_metrics.csv.
post0 shortlist: the ctr/raw members of the same list, SAE seed 0."""
import json, sys
from pathlib import Path
from selection import FAMILIES
HERE = Path(__file__).resolve().parent; ROOT = HERE / "results"; rec = {}
steel = lambda r: max(r[f"{f}_id"] for f in FAMILIES)
for n in range(5, 11):
    keep = {}
    for cond in ("fixed", "trainable"):
        d = ROOT / f"g{n}-s0-{cond}" / "sae-injection"; rows = [json.loads(p.read_text()) for p in d.glob("*_s0.json")]; rows = [r for r in rows if "error" not in r]
        assert len(rows) > 100, (n, cond, len(rows))
        unsup = [r for r in rows if r["variant"] in ("ctr", "raw")]; aware = [r for r in rows if r["variant"] in ("open", "closed")]
        picks = sorted(unsup, key=steel)[-3:] + sorted(aware, key=steel)[-2:] + sorted(unsup, key=lambda r: r["fair_corr"])[-2:]
        picks += [max((r for r in rows if r["kind"] == k), key=steel) for k in ("relu", "topk", "btopk", "mat")]
        for r in picks:
            if r["kind"] == "mp": continue
            keep.setdefault((r["spec"], r["variant"]), []).append(dict(cond=cond, steel=round(steel(r), 4), fair_corr=round(r["fair_corr"], 4), fair_id=round(r["fair_id"], 4)))
    jobs = [dict(spec=s, variant=v, seed=sd) for (s, v) in keep for sd in (0, 1)]
    json.dump(jobs, open(HERE / "jobs" / f"short-g{n}.json", "w")); post = [dict(spec=s, variant=v, seed=0) for (s, v) in keep if v in ("ctr", "raw")]
    json.dump(post, open(HERE / "jobs" / f"short-post0-g{n}.json", "w")); rec[f"g{n}"] = {f"{s}|{v}": why for (s, v), why in keep.items()}
    print(n, len(jobs), len(post), sorted(keep))
json.dump(rec, open(HERE / "jobs" / "shortlist.json", "w"), indent=1)
