"""Cross-model summary for bind_icl results.  usage: python plot_bind_models.py label=dir [label=dir ...] out.png"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_bind import load, acc_curve, decompose, se, parse, BIN_X, COL, LAB, LATE

args = sys.argv[1:]
out = args[-1]; specs = [a.split("=", 1) for a in args[:-1]]
arms = ["rand", "actpc", "native", "unembed"]

fig, axes = plt.subplots(2, len(specs), figsize=(4.4 * len(specs), 6.4), squeeze=False)
for j, (label, d) in enumerate(specs):
    dat = load(os.path.abspath(os.path.join(d, "main.json"))); runs = dat["runs"]
    summ = {}
    for name, rr in runs.items():
        A = acc_curve(rr); L, S = decompose(rr)
        health = np.mean([r["top_is_label"][LATE[0]:LATE[1]].mean() for r in rr])
        summ[name] = dict(late=A[:, -1].mean(), late_se=se(A[:, -1]), L=L, S=S, health=health, curve=A.mean(0), curve_se=se(A[:, -1]))
    best = {}
    for n, s in summ.items():
        if n in ("none", "token") or s["health"] < 0.9: continue
        arm = parse(n)[0]
        if arm not in best or s["late"] > summ[best[arm]]["late"]: best[arm] = n
    ax = axes[0, j]
    ax.plot(BIN_X, summ["none"]["curve"], color=COL["none"], ls=":", label=LAB["none"])
    ax.plot(BIN_X, summ["token"]["curve"], color=COL["token"], ls="--", label=LAB["token"])
    for arm in arms:
        if arm in best:
            n = best[arm]; a, h, w, b = parse(n)
            ax.plot(BIN_X, summ[n]["curve"], color=COL[arm], marker="o", ms=3, label=f"{LAB[arm]}\n  best: hs{h} {w} beta {b}")
    ax.set_xscale("log"); ax.axhline(0.5, color="0.8", lw=0.8); ax.set_ylim(0.3, 1.02); ax.set_title(label)
    ax.set_xlabel("shot index"); ax.legend(fontsize=6, loc="upper left")
    if j == 0: ax.set_ylabel("counterbalanced accuracy")
    ax = axes[1, j]
    names = ["token"] + [best[a] for a in arms if a in best]
    x = np.arange(len(names))
    for comp, off, hatch, alpha in (("L", -0.2, "", 0.9), ("S", 0.2, "//", 0.45)):
        ax.bar(x + off, [summ[n][comp].mean() for n in names], 0.38, yerr=[se(summ[n][comp]) for n in names], capsize=3,
               color=[COL[n if n == "token" else parse(n)[0]] for n in names], hatch=hatch, alpha=alpha,
               label="learned (flips with convention)" if comp == "L" else "steering (fixed sign)")
        for xi, n in zip(x, names):
            ax.scatter(np.full(len(summ[n][comp]), xi + off) + np.random.default_rng(0).normal(0, 0.03, len(summ[n][comp])),
                       summ[n][comp], s=6, color="k", alpha=0.5, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([n if n == "token" else parse(n)[0] for n in names], fontsize=8)
    if j == 0: ax.set_ylabel(f"logit diff, shots {LATE[0]}-{LATE[1]}"); ax.legend(fontsize=7)
fig.suptitle("Frozen models bind an injected bit to an arbitrary label in context; steering component is separated by the convention flip", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=140); print("saved", out)
