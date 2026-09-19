"""Figures for the large-grid replication (results_big/big.json -> figures_big/)."""
import json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_big"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_big"; os.makedirs(FIG, exist_ok=True)
R = json.load(open(f"{RES}/big.json")); rows, un, curves, geo, meta = R["rows"], R["unseen"], R["curves"], R["geometry"], R["meta"]
sizes = sorted({r["N"] for r in rows}); base = {r["ctx"]: r for r in rows if r["cand"] == "none"}; bun = {(u["ctx"], u["X"]): u for u in un if u["cand"] == "none"}
COL = {"none": "k", "own_b2": "tab:blue", "rand_b4": "tab:purple", "plane_b3": "#a1d99b", "plane_matched": "#238b45", "adj_b6": "#fdae6b", "adj_matched": "#d94801"}
LAB = {"none": "no injection", "own_b2": "own plane (calibrated, hs 26, β2)", "rand_b4": "random plane (hs 32, β4)", "plane_b3": "QK plane port (hs 32, β3 fixed)", "plane_matched": "QK plane port (hs 32, β matched)",
       "adj_b6": "adjacency-kernel port (hs 32, β6 fixed)", "adj_matched": "adjacency-kernel port (hs 32, β matched)"}


def m(c, N, which):
    if which == "uptake": return np.array([r["up_early"] - base[r["ctx"]]["up_early"] for r in rows if r["cand"] == c and r["N"] == N])
    if which == "uptake_late": return np.array([r["up_late"] - base[r["ctx"]]["up_late"] for r in rows if r["cand"] == c and r["N"] == N])
    if which == "self": return np.array([r["self_late"] - base[r["ctx"]]["self_late"] for r in rows if r["cand"] == c and r["N"] == N])
    if which == "override": return np.array([r["ov_inj"] - r["ov_chance"] for r in rows if r["cand"] == c and r["N"] == N])
    if which == "ov_true": return np.array([r["ov_true"] - base[r["ctx"]]["ov_true"] for r in rows if r["cand"] == c and r["N"] == N])
    if which == "unseen": return np.array([u["trueX_X"] - bun[(u["ctx"], u["X"])]["P_nbrs_X"] for u in un if u["cand"] == c and u["N"] == N])
    if which == "claimY": return np.array([u["claimY_Y"] - bun[(u["ctx"], u["X"])]["P_nbrs_Y"] for u in un if u["cand"] == c and u["N"] == N])
    if which == "abs_late": return np.array([r["up_late"] for r in rows if r["cand"] == c and r["N"] == N])
    if which == "abs_unseen": return np.array([u["trueX_X"] if c != "none" else u["P_nbrs_X"] for u in un if u["cand"] == c and u["N"] == N])


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
def pv(v): return stats.ttest_1samp(v, 0).pvalue if len(v) > 2 and np.std(v) > 0 else np.nan


# ---------------- summary table
cands = [c for c in dict.fromkeys(r["cand"] for r in rows) if c != "none"]
lines = ["large-grid replication; deltas paired vs no injection; SEM over contexts (unseen: over items)"]
for N in sizes:
    b = [r for r in rows if r["cand"] == "none" and r["N"] == N]; nb = len(b); g = [x for x in geo if x["N"] == N]
    lines.append(f"\nN={N} ({meta['grids'][str(N)][0]}x{meta['grids'][str(N)][1]}), T={meta['T'][str(N)]}, {nb} contexts; baseline early {np.mean([r['up_early'] for r in b]):.3f} late {np.mean([r['up_late'] for r in b]):.3f} unseen-X {np.mean([u['P_nbrs_X'] for u in un if u['cand']=='none' and u['N']==N]):.3f}; "
                 f"own-plane R2 {np.mean([x['r2_own'] for x in g]):.2f} (in top-2 PCs {np.mean([x['r2_in_pcs2'] for x in g]):.2f}, PC2 var {np.mean([x['pc_var2'] for x in g]):.2f}), energy in cone32 {np.mean([x['own_in_cone32'] for x in g]):.2f}, cos plane {np.mean([x['own_cos_plane'] for x in g]):.2f}, own norm/nref {np.mean([x['own_norm'] for x in g]) / meta['nref']:.2f}; "
                 f"matched beta plane {meta['betas'][str(N)]['plane_matched']:.2f} adj {meta['betas'][str(N)]['adj_matched']:.2f}; kernel adj self/nbr/d2/d3/other {meta['kernels'][str(N)]['adj']['self']:.2f}/{meta['kernels'][str(N)]['adj']['nbr']:.2f}/{meta['kernels'][str(N)]['adj']['d2']:.2f}/{meta['kernels'][str(N)]['adj']['d3']:.2f}/{meta['kernels'][str(N)]['adj']['other']:.2f}")
    lines.append(f"  {'strategy':16s} {'early':>22s} {'late':>22s} {'override':>22s} {'unseen X':>22s} {'claim-Y':>22s} {'self':>7s}")
    for c in cands:
        lines.append(f"  {c:16s} " + " ".join(f"{m(c, N, w).mean():+.3f}±{sem(m(c, N, w)):.3f} p{pv(m(c, N, w)):.0e}" for w in ("uptake", "uptake_late", "override", "unseen", "claimY")) + f" {m(c, N, 'self').mean():+.3f}")
open(f"{FIG}/big_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))

# ---------------- Fig 1: learning curves, one panel per grid
show = ["none", "own_b2", "rand_b4", "plane_matched", "adj_matched"]
fig, ax = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4.2), sharey=True)
for j, N in enumerate(sizes):
    T = meta["T"][str(N)]; K = max(10, T // 40); ctxs = sorted({r["ctx"] for r in rows if r["N"] == N})
    for c in show:
        M = np.array([curves[f"{k}|{c}"] for k in ctxs]); mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
        t = np.arange(K, len(mu) + 1); sm = np.convolve(mu, np.ones(K) / K, "valid"); ss = np.convolve(se, np.ones(K) / K, "valid")
        ax[j].plot(t, sm, color=COL[c], lw=2 if c in ("none", "adj_matched") else 1.4, label=LAB[c]); ax[j].fill_between(t, sm - ss, sm + ss, color=COL[c], alpha=0.12)
    if T > 4096: ax[j].axvline(4096, color="gray", ls="--", lw=0.9); ax[j].text(4096, 0.97, " 4096-token local window", fontsize=7, color="gray", va="top")
    ax[j].set(xscale="log", title=f"{N} nodes ({meta['grids'][str(N)][0]}×{meta['grids'][str(N)][1]}), {T} tokens", xlabel="context length (tokens)"); ax[j].grid(alpha=0.3); ax[j].set_ylim(0, 1)
ax[0].set_ylabel("rule-following probability mass"); ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle("In-context learning curves by grid size, with and without injection (mean ± SEM over contexts, smoothed)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/big1_curves.png", dpi=150)

# ---------------- Fig 2: summary vs N (matched ports)
def summary_fig(cs, fname, title):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8)); ax = ax.ravel()
    for j, (which, lab) in enumerate((("uptake", "faster learning: Δ mass, first quarter of context"), ("uptake_late", "late accuracy: Δ mass, last two thirds"), ("override", "override: injected-map mass − chance"), ("unseen", "unseen node: Δ P(true neighbors of X)"))):
        for c in cs:
            v = [m(c, N, which) for N in sizes]; ax[j].errorbar(sizes, [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", color=COL[c], lw=1.8, capsize=3, label=LAB[c])
        ax[j].axhline(0, color="k", lw=0.6); ax[j].set(xscale="log", xlabel="grid size (nodes)", title=lab); ax[j].set_xticks(sizes); ax[j].set_xticklabels([str(s) for s in sizes]); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=7.5); fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=150)


summary_fig(["own_b2", "rand_b4", "plane_matched", "adj_matched"], "big2_summary.png", "Injection strategies vs grid size (paired Δ vs no injection; ports with kernel-matched strength)")
summary_fig(["plane_b3", "plane_matched", "adj_b6", "adj_matched"], "big3_fixed_vs_matched.png", "Fixed strength vs kernel-matched strength for the two weight-defined ports")

# ---------------- Fig 4: geometry / baseline vs N
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
b_late = [np.mean([r["up_late"] for r in rows if r["cand"] == "none" and r["N"] == N]) for N in sizes]; b_early = [np.mean([r["up_early"] for r in rows if r["cand"] == "none" and r["N"] == N]) for N in sizes]
b_un = [np.mean([u["P_nbrs_X"] for u in un if u["cand"] == "none" and u["N"] == N]) for N in sizes]
ax[0].plot(sizes, b_late, marker="o", color="k", label="baseline accuracy, last two thirds"); ax[0].plot(sizes, b_early, marker="s", color="gray", label="baseline accuracy, first quarter"); ax[0].plot(sizes, b_un, marker="^", color="tab:red", label="baseline P(true nbrs) at an unseen node")
ax[0].plot(sizes, [4 / N for N in sizes], ls="--", color="gray", lw=0.8, label="chance (4/N)")
ax[0].set(xscale="log", xlabel="grid size (nodes)", ylabel="probability mass", title="Baseline: how well is each grid learned?"); ax[0].set_xticks(sizes); ax[0].set_xticklabels([str(s) for s in sizes]); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3)
for key, lab, col in (("r2_own", "R² of spectral coords → node means (own plane fit)", "tab:blue"), ("own_in_cone32", "own-plane energy inside the top-32 QK cone", "#006d2c"), ("pc_var2", "variance in the top-2 PCs of node means", "gray")):
    v = [[x[key] for x in geo if x["N"] == N] for N in sizes]; ax[1].errorbar(sizes, [np.mean(a) for a in v], yerr=[sem(a) for a in v], marker="o", color=col, capsize=3, label=lab)
ax[1].set(xscale="log", xlabel="grid size (nodes)", ylim=(0, 1), title="The model's own plane vs grid size"); ax[1].set_xticks(sizes); ax[1].set_xticklabels([str(s) for s in sizes]); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.3)
ax[2].plot(sizes, [meta["betas"][str(N)]["plane_matched"] for N in sizes], marker="o", color="#238b45", label="matched β, plane port (fixed = 3)")
ax[2].plot(sizes, [meta["betas"][str(N)]["adj_matched"] for N in sizes], marker="o", color="#d94801", label="matched β, adjacency port (fixed = 6)")
ax[2].set(xscale="log", xlabel="grid size (nodes)", ylabel="β", title="Matched strength (same neighbor-vs-distance-2 logit contrast)"); ax[2].set_xticks(sizes); ax[2].set_xticklabels([str(s) for s in sizes]); ax[2].legend(fontsize=7.5); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/big4_geometry.png", dpi=150)
# ---------------- Fig 5: unseen node, absolute, vs chance
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for c in ["none", "own_b2", "rand_b4", "plane_matched", "adj_matched"]:
    v = [m(c, N, "abs_unseen") for N in sizes]; ax[0].errorbar(sizes, [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", color=COL[c], lw=1.8, capsize=3, label=LAB[c])
ax[0].plot(sizes, [4 / N for N in sizes], ls="--", color="gray", lw=0.8, label="chance (4/N)")
ax[0].set(xscale="log", yscale="log", xlabel="grid size (nodes)", ylabel="P(true neighbors of X) at the first query on X", title="Unseen node: absolute mass on the true neighbors"); ax[0].set_xticks(sizes); ax[0].set_xticklabels([str(s) for s in sizes]); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3)
for c in ["own_b2", "rand_b4", "plane_matched", "adj_matched"]:
    v = [m(c, N, "claimY") for N in sizes]; ax[1].errorbar(sizes, [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", color=COL[c], lw=1.8, capsize=3, label=LAB[c])
ax[1].axhline(0, color="k", lw=0.6); ax[1].set(xscale="log", xlabel="grid size (nodes)", ylabel="Δ P(neighbors of Y) vs no injection", title="Unseen node with a FAKE coordinate: mass on the fake location's neighbors"); ax[1].set_xticks(sizes); ax[1].set_xticklabels([str(s) for s in sizes]); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/big5_unseen.png", dpi=150)
print(f"figures -> {FIG}")
