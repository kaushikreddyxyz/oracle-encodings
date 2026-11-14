#!/usr/bin/env python3
"""Aggregate campaign results and regenerate canonical learning curves (PDF/SVG).

Run from any directory; --results optionally overrides the raw-results folder.
"""
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"          # summary CSVs (figures go to figures/)
TABLES.mkdir(exist_ok=True)

COND_RE = re.compile(r"fast-(baseline|inj-fixed|inj-trainable)")
COND_MAP = {"baseline": "base", "inj-fixed": "fixed", "inj-trainable": "trainable"}


def collect(results: Path):
    rows, losses = [], []
    seen = {}
    for mj in sorted(results.glob("*/g*-s*/*/eval/metrics.json")):
        run = mj.parent.parent
        if not (run / "DONE").exists():   # pulled while still being scored -> skip until complete
            continue
        m = re.match(r"g(\d+)-s(\d+)", run.parent.name)
        grid, seed = int(m.group(1)), int(m.group(2))
        cond = COND_MAP[COND_RE.search(run.name).group(1)]
        d = json.load(open(mj))
        key = (grid, seed, cond)
        if key in seen:  # the same run finished on two pods (guard raced the worker): keep the first complete one
            print(f"duplicate {key}: keeping {seen[key]}, ignoring {mj.parts[-5]}")
            continue
        seen[key] = mj.parts[-5]
        n_iter = None
        lc = run / "train_loss.csv"
        if lc.exists():
            tl = pd.read_csv(lc)
            n_iter = int(tl["iter"].max()) + 1
            tl["grid"], tl["seed"], tl["cond"] = grid, seed, cond
            losses.append(tl)
        for r in d["rows"]:
            it = r["iteration"] if r["iteration"] >= 0 else (n_iter if n_iter else -1)
            rows.append(dict(grid=grid, seed=seed, cond=cond, iteration=it, final=r["iteration"] < 0, **{k: v for k, v in r.items() if k != "iteration"}))
    df = pd.DataFrame(rows)
    tl = pd.concat(losses) if losses else pd.DataFrame(columns=["iter", "loss", "grid", "seed", "cond"])
    return df, tl


def loss_at_checkpoints(tl, iters):
    """mean train loss over the window ending at each checkpoint step (window = gap to previous ckpt)."""
    out = []
    for (g, s, c), t in tl.groupby(["grid", "seed", "cond"]):
        t = t.sort_values("iter")
        its = sorted(i for i in iters if i <= t["iter"].max() + 1)
        prev = -1
        for i in its:
            lo = max(prev + 1, i - 200)
            w = t[(t["iter"] >= lo) & (t["iter"] <= i)]["loss"]
            if len(w):
                out.append(dict(grid=g, seed=s, cond=c, iteration=i, train_loss=float(w.mean())))
            prev = i
    return pd.DataFrame(out)


def main(results=None):
    """Refresh aggregated data and the two canonical learning-curve figures."""
    from plot_curves_overlay import main as render_curves
    root = Path(__file__).resolve().parent
    df, tl = collect(Path(results) if results else root / "results")
    if df.empty:
        print("no results yet")
        return
    df.to_csv(TABLES / "all_metrics.csv", index=False)
    tlc = loss_at_checkpoints(tl, sorted(df["iteration"].unique()))
    tlc.to_csv(TABLES / "train_loss_by_ckpt.csv", index=False)
    for graft in ("fixed", "trainable"):
        render_curves(graft)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    main(parser.parse_args().results)
