"""Unseen-edge analysis + figure (results_edge/edge.json -> figures_qk/story4_unseen_edge.png)."""
import json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_edge"
rows = json.load(open(f"{RES}/edge.json"))["rows"]
conds = ["none", "rand26_b4", "lateanti_b4", "own_b2", "late_b3", "late_b4", "latenon_b4", "lateadj_b4", "lateadj_minus_b4", "z2_hs16_b4", "adj_hs16_b6"]
LAB = {"none": "no injection", "rand26_b4": "random plane", "lateanti_b4": "anti-port", "own_b2": "own plane (calibrated)", "late_b3": "QK-port β3", "late_b4": "QK-port β4",
       "latenon_b4": "port, 4 random late heads", "lateadj_b4": "adjacency port (edge IN kernel)", "lateadj_minus_b4": "adjacency port (edge DELETED)", "z2_hs16_b4": "spectral port @hs16", "adj_hs16_b6": "adjacency port @hs16 β6"}
COL = {"none": "k", "rand26_b4": "tab:purple", "lateanti_b4": "tab:red", "own_b2": "tab:blue", "late_b3": "#74c476", "late_b4": "#006d2c", "latenon_b4": "tab:gray",
       "lateadj_b4": "tab:orange", "lateadj_minus_b4": "#ffbb78", "z2_hs16_b4": "tab:olive", "adj_hs16_b6": "#e6550d"}

# one item = one (context, direction); paired across conditions
keys = sorted({(r["set"], r["seed"], r["src"]) for r in rows})
def get(c, k, m): return [r[m] for r in rows if r["cond"] == c and (r["set"], r["seed"], r["src"]) == k][0]
def vec(c, m): return np.array([get(c, k, m) for k in keys])
base = {m: vec("none", m) for m in ("P_heldout", "P_seen_nbr", "P_d2", "P_d3")}
print(f"n items = {len(keys)} (contexts x directions); mean queries per item = {np.mean([r['n_q'] for r in rows if r['cond']=='none']):.1f}")
print(f"{'condition':32s} P(held-out edge)  P(seen nbr, each)  P(dist-2 node, each)  held-out − dist2 [paired Δ vs none]   held-out/seen")
lines = []
for c in conds:
    if not any(r["cond"] == c for r in rows): continue
    ho, sn, d2 = vec(c, "P_heldout"), vec(c, "P_seen_nbr"), vec(c, "P_d2")
    contrast = ho - d2; dcon = contrast - (base["P_heldout"] - base["P_d2"])
    p = stats.ttest_1samp(dcon, 0).pvalue if c != "none" else stats.ttest_1samp(contrast, 0).pvalue
    line = (f"{LAB[c]:32s} {ho.mean():.3f}±{ho.std(ddof=1)/np.sqrt(len(ho)):.3f}      {sn.mean():.3f}             {d2.mean():.3f}               "
            f"{contrast.mean():+.3f} [{dcon.mean():+.3f}, p={p:.1g}]            {ho.mean()/sn.mean():.2f}")
    print(line); lines.append(line)

# ---- figure
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
show = [c for c in conds if any(r["cond"] == c for r in rows)]
x = np.arange(len(show)); w = 0.27
for j, (m, lab, col, hatch) in enumerate((("P_seen_nbr", "a SEEN neighbor of X (per node)", "#9ecae1", ""), ("P_heldout", "the HELD-OUT neighbor Y (edge never in context)", "tab:green", ""), ("P_d2", "a non-neighbor at distance 2 (per node)", "#d9d9d9", "//"))):
    v = [vec(c, m) for c in show]
    ax[0].bar(x + (j - 1) * w, [a.mean() for a in v], w, yerr=[a.std(ddof=1) / np.sqrt(len(a)) for a in v], color=col, hatch=hatch, capsize=2, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels([LAB[c] for c in show], rotation=35, ha="right", fontsize=7.5)
ax[0].set(ylabel="P(next token) at queries on X", title=f"Unseen edge: what does the model predict after X?  (n = {len(keys)} context×direction items)")
ax[0].legend(fontsize=7.5)
# paired contrast Δ vs none
vals, errs = [], []
for c in show:
    d = (vec(c, "P_heldout") - vec(c, "P_d2")) - (base["P_heldout"] - base["P_d2"]); vals.append(d.mean()); errs.append(d.std(ddof=1) / np.sqrt(len(d)))
ax[1].bar(x, vals, 0.6, yerr=errs, color=[COL[c] for c in show], capsize=3)
ax[1].axhline(0, color="k", lw=0.7); ax[1].set_xticks(x); ax[1].set_xticklabels([LAB[c] for c in show], rotation=35, ha="right", fontsize=7.5)
ax[1].set(ylabel="Δ [P(held-out Y) − P(dist-2 non-neighbor)] vs no injection", title="Does the channel supply the missing edge? (paired vs no injection)")
fig.tight_layout(); fig.savefig(f"{HERE}/figures_qk/story4_unseen_edge.png", dpi=160)
open(f"{HERE}/figures_qk/edge_summary.txt", "w").write("\n".join(lines)); print("figure -> figures_qk/story4_unseen_edge.png")
