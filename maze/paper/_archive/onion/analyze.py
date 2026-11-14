#!/usr/bin/env python3
"""Tables for the onion-graft pilot (6x6, seed 0). Numpy only; run from anywhere.

Inputs
  results/runs/<seed_tag>/<run>/eval/metrics.json      eval_ckpts.py  (every checkpoint; true + phantom world)
  results/dist_out/g6__<kind>__<seed_tag>.npz          eval_distance.py on the standard set
  results/local_out/g6__<kind>__<seed_tag>.npz         eval_local.py on the one-swap set
  results/runs/<seed_tag>/<run>/injector.final.pt      onion_depth -> layer of every connection
Reference = the edge grafts of the curves campaign (5 seeds): ../curves/tables/all_metrics.csv,
../curves/results/distance/out, ../curves/results/local/out.

Outputs: onion_final.csv, onion_distance.csv, onion_depth.csv (+ the same tables printed as markdown).

Depth of a connection = the shallowest onion layer that stores it (0 = outermost, a threshold on the tile's
direction; 3 = innermost). The edge-graft reference is stratified by the SAME labels as a control: the labels mean
nothing to an edge graft, so any trend there is item difficulty, not the code.
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CUR = HERE.parent / "curves"
RES = HERE / "results"
G = 6


def load_injector(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def runs():
    out = {}
    for d in sorted((RES / "runs").glob(f"g{G}-s*")):
        for r in d.glob("fast-inj-*"):
            if (r / "eval" / "metrics.json").exists():
                kind = "fixed" if "inj-fixed" in r.name else "trainable"
                code = d.name.split("-", 2)[2]  # e.g. onion-checker, onion-full-dummy, edge-dummy
                out[f"{code} {kind}"] = dict(dir=r, seed_tag=d.name, kind=kind, code=code)
    return out


def min_depth(depth):
    d = np.where(depth < 0, 99, depth).min(0)
    return np.where(d == 99, -1, d)


def final_rows(R):
    rows = []
    for name, r in R.items():
        for m in json.load(open(r["dir"] / "eval" / "metrics.json"))["rows"]:
            if m["iteration"] == -1:
                rows.append(dict(model=name, world=m["world"], p_sp=m["p_sp"], p_valid=m["p_valid"],
                                 p_invalid=m["p_invalid"], solved=m["solved"], reached=m["reached"]))
    ref = {}
    for row in csv.DictReader(open(CUR / "tables" / "all_metrics.csv")):
        if int(row["grid"]) == G and row["final"] == "True":
            ref.setdefault((row["cond"], row["world"]), []).append(row)
    for (cond, world), rs in sorted(ref.items()):
        f = lambda k: float(np.mean([float(x[k]) for x in rs]))
        rows.append(dict(model=f"REF {cond} (mean of {len(rs)} seeds)", world=world, p_sp=f("p_sp"), p_valid=f("p_valid"),
                         p_invalid=f("p_invalid"), solved=f("solved"), reached=f("reached")))
    return rows


def distance_rows(R):
    S = np.load(CUR / "results/distance/sets" / f"g{G}.npz")
    rc = (~S["path_same"]) & (S["level"] > 0)
    same = S["level"] == 0
    rows = []

    def add(label, files):
        a = [np.load(f) for f in files]
        if a:
            rows.append(dict(model=label, n_models=len(a),
                             same_maze=float(np.mean([x["solved_graft"][same].mean() for x in a])),
                             route_changed=float(np.mean([x["solved_graft"][rc].mean() for x in a])),
                             route_changed_follows_prompt=float(np.mean([x["solved_token"][rc].mean() for x in a]))))
    for name, r in R.items():
        add(name, list((RES / "dist_out").glob(f"g{G}__{r['kind']}__{r['seed_tag']}.npz")))
    for kind in ("fixed", "trainable"):
        add(f"REF edge {kind}", sorted((CUR / "results/distance/out").glob(f"g{G}__{kind}__g{G}-s?.npz")))
    return rows


def local_items(npz, N):
    g = lambda c, k: npz[f"{c}__{k}"][:N]
    return dict(p_new=np.exp(g("graft_B", "lp_b")), p_old=np.exp(g("graft_B", "lp_a")),
                top1_new=g("graft_B", "argmax") == npz["tid_b"][:N],
                gain_nats=g("graft_B", "lp_b") - g("graft_A", "lp_b"), drop_nats=g("graft_A", "lp_a") - g("graft_B", "lp_a"),
                used_door=g("graft_B", "used"), crossed_wall=g("graft_B", "crossed"), solved_new=g("graft_B", "solved_b"))


def depth_rows(R):
    S = np.load(CUR / "results/local/sets" / f"g{G}.npz")
    rows = []
    onion = {n: r for n, r in R.items() if r["code"].startswith("onion") and not r["code"].endswith("dummy")}
    for name, r in onion.items():
        f = RES / "local_out" / f"g{G}__{r['kind']}__{r['seed_tag']}.npz"
        if not f.exists():
            continue
        md = min_depth(load_injector(r["dir"] / "injector.final.pt")["onion_depth"].numpy())
        groups = {name: [np.load(f)]}
        groups[f"REF edge {r['kind']} (same labels)"] = [
            np.load(p) for p in sorted((CUR / "results/local/out").glob(f"g{G}__{r['kind']}__g{G}-s?.npz"))]
        for label, files in groups.items():
            for which, edge in (("door e+", S["add"]), ("wall e-", S["rem"])):
                for k in (0, 1, 2, 3, "all"):
                    vals = []
                    for z in files:
                        N = len(z["tid_a"])
                        m = (S["family"][:N] == 0) & (True if k == "all" else md[edge[:N]] == k)
                        v = local_items(z, N)
                        vals.append({kk: float(np.mean(vv[m])) for kk, vv in v.items()} | {"n": int(m.sum())})
                    rows.append(dict(model=label, code=r["code"], stratified_by=which, depth=k, n_items=vals[0]["n"],
                                     **{kk: float(np.mean([x[kk] for x in vals])) for kk in vals[0] if kk != "n"}))
    return rows


def md_table(rows, cols, fmt="{:.3f}"):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    R = runs()
    print(f"{len(R)} runs: {list(R)}\n")
    fr = final_rows(R)
    write_csv(HERE / "onion_final.csv", fr)
    print(md_table(fr, ["model", "world", "p_sp", "p_valid", "p_invalid", "solved", "reached"]), "\n")
    dr = distance_rows(R)
    write_csv(HERE / "onion_distance.csv", dr)
    print(md_table(dr, ["model", "n_models", "same_maze", "route_changed", "route_changed_follows_prompt"]), "\n")
    pr = depth_rows(R)
    write_csv(HERE / "onion_depth.csv", pr)
    print(md_table(pr, ["model", "stratified_by", "depth", "n_items", "p_new", "top1_new", "gain_nats", "drop_nats",
                        "used_door", "crossed_wall", "solved_new"]))
