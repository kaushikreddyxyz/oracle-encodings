#!/usr/bin/env python3
"""Gather every run's eval_fade output into flat CSVs (numpy/stdlib only).

  python maze/experiments/08_seed_and_fade/scripts/collect.py [--root results]
reads   <root>/s{seed}/<run>/{fade_config.json, eval_fade*/metrics.json}
writes  <root>/conditions.csv   run x eval x iteration x condition
        <root>/retention.csv    run x eval x iteration x position
        <root>/retention_by_layer.csv
and prints the final-checkpoint tables.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def arm_of(c):
    if c["mode"] == "none":
        return "base"
    if c["fade"] == "none":
        return "permanent"
    a = "fade" if c["fade"] == "bernoulli" else "fade-scale"
    if c["mode"] == "trainable":
        a += "-trainableW"
    if c["freeze_readers"]:
        a += "+frozen" + (f"(until {c['unfreeze_at']})" if c["unfreeze_at"] >= 0 else "")
    if c["aux_weight"] > 0:
        a += "+aux"
    if c.get("blank_prompt"):
        a += "+blank"
    return a


def main(root="results"):
    root = Path(root) if Path(root).is_absolute() else HERE / root
    cond_rows, ret_rows, lay_rows = [], [], []
    for cfg_f in sorted(root.glob("s*/*/fade_config.json")):
        run = cfg_f.parent
        c = json.load(open(cfg_f))
        arm, seed = arm_of(c), c["seed"]
        for mf in sorted(run.glob("eval_fade*/metrics.json")):
            ev = mf.parent.name
            for row in json.load(open(mf))["rows"]:
                key = dict(seed=seed, arm=arm, site=c["layer"], edit_layer=row["layer"], eval=ev,
                           iteration=row["iteration"], run=f"s{seed}/{run.name}")
                for cond, v in row["conditions"].items():
                    cond_rows.append({**key, "condition": cond, **v})
                for pos, v in row["retention"].items():
                    ret_rows.append({**key, "position": pos, **v})
                for L, byp in row.get("retention_by_layer", {}).items():
                    for pos, v in byp.items():
                        lay_rows.append({**key, "layer": int(L), "position": pos, **v})
    for name, rows in (("conditions", cond_rows), ("retention", ret_rows), ("retention_by_layer", lay_rows)):
        if not rows:
            continue
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(root / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        print(f"{name}.csv: {len(rows)} rows")

    # final-checkpoint table, one line per run and site
    fin = lambda r: r["iteration"] == -1 and r["eval"] != "eval_fade_traj"
    runs = sorted({(r["seed"], r["site"], r["arm"], r["run"], r["eval"]) for r in cond_rows if fin(r)})
    print("\nseed site arm @edit-layer | retention@origin: cos gain zero-fit probe major | solve rates: "
          "off(same) on(same) on(rc) best-delta(rc) delta_perm(rc) ablate ablate_rand | replaceA1 best-replace(rc)")
    for seed, site, arm, run, ev in runs:
        cr = {r["condition"]: r for r in cond_rows if fin(r) and r["run"] == run and r["eval"] == ev}
        rr = next((r for r in ret_rows if fin(r) and r["run"] == run and r["eval"] == ev and r["position"] == "origin"), None)
        g = lambda cnd, k: (f"{cr[cnd][k]:.3f}" if cnd in cr and cr[cnd].get(k) is not None else "  -  ")
        deltas = {k: v.get("route_changed") for k, v in cr.items() if k.startswith("delta") and k != "delta_perm"}
        for k in list(cr):
            if k.startswith("ablate_rand") and k != "ablate_rand":
                cr.pop(k)
        best = max(deltas, key=lambda k: deltas[k] if deltas[k] is not None else -1) if deltas else None
        ret = (f"{rr['cos_gen']:+.3f} {rr['gain']:+.3f} {rr['acc_zero_fit']:.3f} {rr['acc_probe']:.3f} {rr['acc_majority']:.3f}"
               if rr else "")
        reps = {k: v.get("route_changed") for k, v in cr.items() if k.startswith("replace") and not k.startswith("replaceA")}
        rbest = max(reps, key=lambda k: reps[k] if reps[k] is not None else -1) if reps else None
        el = f"L{cr[next(iter(cr))]['edit_layer']}" + ("R" if ev.endswith("replace") else "")
        print(f"s{seed} L{site} {arm:24s} @{el:4s}| {ret} | {g('off','same_maze')} {g('on','same_maze')} {g('on','route_changed')} "
              f"{(f'{deltas[best]:.3f} ({best})' if best else '  -  ')} {g('delta_perm','route_changed')} "
              f"{g('ablate','same_maze')} {g('ablate_rand','same_maze')} | {g('replaceA1','same_maze')} "
              f"{(f'{reps[rbest]:.3f} ({rbest})' if rbest else '  -  ')}")


if __name__ == "__main__":
    main(*sys.argv[1:2]) if len(sys.argv) < 2 or not sys.argv[1].startswith("--") else main(sys.argv[2])
