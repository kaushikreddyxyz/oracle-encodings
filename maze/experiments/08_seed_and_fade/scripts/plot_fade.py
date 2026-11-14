#!/usr/bin/env python3
"""Trajectory figure: per checkpoint, gain along W_inj at the origin, autopilot solve rate, steering along
W_inj (best strength), and graft-on route-changed solve rate; one column per site, one line per arm.
  python maze/experiments/08_seed_and_fade/scripts/plot_fade.py [--seed 0] [--sites 2,3,4,6]
Writes fig_trajectory_s{seed}.{pdf,svg} next to results/.
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
COL = {"base": "#2a7a57", "permanent": "#000c7a", "fade": "#7a8fb3", "fade+frozen": "#e07b00", "fade+aux": "#e0a100",
       "fade+blank": "#b03060", "fade+frozen+blank": "#d2691e"}
LAB = {"base": "base model", "permanent": "permanent graft", "fade": "fade", "fade+frozen": "fade, readers frozen",
       "fade+aux": "fade + aux (diagnostic only)", "fade+blank": "fade, blanked-prompt curriculum",
       "fade+frozen+blank": "blanked prompt + readers frozen"}


def arm_of(c):
    if c["mode"] == "none": return "base"
    if c["fade"] == "none": return "permanent"
    a = "fade" if c["fade"] == "bernoulli" else "fade-scale"
    if c["mode"] == "trainable": a += "-trainableW"
    if c["freeze_readers"]: a += "+frozen" + (f"(until {c['unfreeze_at']})" if c["unfreeze_at"] >= 0 else "")
    if c["aux_weight"] > 0: a += "+aux"
    if c.get("blank_prompt"): a += "+blank"
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sites", default="2,3,4,6")
    ap.add_argument("--root", default=str(HERE / "results")); ap.add_argument("--tag", default=""); a = ap.parse_args()
    sites = [int(s) for s in a.sites.split(",")]
    root = Path(a.root)
    runs = {}
    for cf in root.glob(f"s{a.seed}/*/fade_config.json"):
        c = json.load(open(cf)); arm = arm_of(c)
        if arm not in COL: continue
        mf = cf.parent / "eval_fade_traj" / "metrics.json"
        if not mf.exists(): continue
        runs[(arm, c["layer"])] = json.load(open(mf))["rows"]
    base = next((v for (arm, _), v in runs.items() if arm == "base"), None)
    metrics = [("gain along W_inj\n(origin, graft off)", lambda r: r["retention"]["origin"]["gain"]),
               ("autopilot solve rate\n(graft off)", lambda r: r["conditions"]["off"].get("same_maze")),
               ("steer along W_inj\n(route-changed solve, best s)", lambda r: max((v.get("route_changed") or 0) for k, v in r["conditions"].items() if k.startswith("delta")) if any(k.startswith("delta") for k in r["conditions"]) else None),
               ("graft on, code B added\n(route-changed solve)", lambda r: r["conditions"].get("on", {}).get("route_changed"))]
    fig, axes = plt.subplots(len(metrics), len(sites), figsize=(3.2 * len(sites), 2.4 * len(metrics)), sharex=True, sharey="row")
    for j, k in enumerate(sites):
        for i, (title, f) in enumerate(metrics):
            ax = axes[i][j]
            for arm in COL:
                rows = base if arm == "base" else runs.get((arm, k))
                if rows is None: continue
                xs = [r["iteration"] for r in rows]; ys = [f(r) for r in rows]
                pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if not pts: continue
                ax.plot(*zip(*pts), "-o", ms=3, lw=1.4, color=COL[arm], label=LAB[arm])
            ax.axvspan(1500, 4000, color="#dddddd", alpha=0.5, lw=0)
            if i == 0: ax.set_title(f"graft site: resid_pre of block {k}", fontsize=10)
            if j == 0: ax.set_ylabel(title, fontsize=8.5)
            if i == len(metrics) - 1: ax.set_xlabel("training step")
            ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Seed and fade{a.tag}, seed {a.seed}: fade window shaded (graft dropout 0 -> 1), 500 mazes per point", fontsize=10)
    fig.tight_layout()
    name = f"fig_trajectory{a.tag.replace(' ', '_').replace(',', '')}_s{a.seed}"
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}")
    print("wrote", HERE / f"{name}.pdf")


if __name__ == "__main__":
    main()
