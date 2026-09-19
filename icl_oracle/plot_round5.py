"""Figures for zoo round 5 (results_zoo/round5.json + edge5.json -> figures_zoo/r5_*.png)."""
import json, os, sys, re
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))

RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_zoo"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_zoo"; os.makedirs(FIG, exist_ok=True)
from scipy import stats as _st
METRICS = (("uptake", "Δ mass, tokens 100–300", "#74c476"), ("uptake_late", "Δ mass, tokens 400–1200", "#238b45"), ("override", "override: injected-map − chance", "tab:orange"),
           ("unseen", "unseen X: Δ P(true nbrs)", "tab:green"), ("claimY", "fake coord: Δ P(nbrs of Y)", "tab:red"))


def load(name):
    R = json.load(open(f"{RES}/{name}.json")); rows, un = R["rows"], R["unseen"]
    base = {r["ctx"]: r for r in rows if r["cand"] == "none"}; bun = {(u["ctx"], u["X"]): u for u in un if u["cand"] == "none"}
    def m(c, which):
        if which == "uptake": return np.array([r["up_early"] - base[r["ctx"]]["up_early"] for r in rows if r["cand"] == c])
        if which == "uptake_late": return np.array([r["up_late"] - base[r["ctx"]]["up_late"] for r in rows if r["cand"] == c])
        if which == "self": return np.array([r["self_late"] - base[r["ctx"]]["self_late"] for r in rows if r["cand"] == c])
        if which == "override": return np.array([r["ov_inj"] - r["ov_chance"] for r in rows if r["cand"] == c])
        if which == "unseen": return np.array([u["trueX_X"] - bun[(u["ctx"], u["X"])]["P_nbrs_X"] for u in un if u["cand"] == c])
        if which == "claimY": return np.array([u["claimY_Y"] - bun[(u["ctx"], u["X"])]["P_nbrs_Y"] for u in un if u["cand"] == c])
    cands = [c for c in dict.fromkeys(r["cand"] for r in rows) if c != "none"]
    return R, m, cands, len(base), len(bun)


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
def pv(v): return _st.ttest_1samp(v, 0).pvalue if len(v) > 2 and np.std(v) > 0 else np.nan


def table(name, m, cands, nc, nu, extra=()):
    lines = [f"[{name}] n contexts = {nc}; unseen items = {nu}", f"{'condition':26s} " + " ".join(f"{lab[:26]:>28s}" for _, lab, _ in METRICS) + "     self"]
    for c in cands: lines.append(f"{c:26s} " + " ".join(f"{m(c, w).mean():+.3f}±{sem(m(c, w)):.3f} p{pv(m(c, w)):.0e}" for w, _, _ in METRICS) + f"   {m(c, 'self').mean():+.3f}")
    lines += list(extra); open(f"{FIG}/{name}_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))

# ---------------- A: adjacency port grid (layer x beta)
if os.path.exists(f"{RES}/round5.json"):
    R, m, cands, nc, nu = load("round5"); table("r5", m, cands, nc, nu)
    hss = sorted({int(re.match(r"adj_hs(\d+)_b", c).group(1)) for c in cands if c.startswith("adj_hs")}); bs = sorted({float(re.match(r"adj_hs\d+_b([\d.]+)", c).group(1)) for c in cands if c.startswith("adj_hs")})
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.6)); cols = {30: "#9ecae1", 32: "#4292c6", 34: "#08519c", 36: "#081d58"}
    for j, (which, lab, col) in enumerate(METRICS):
        for hs in hss:
            v = [m(f"adj_hs{hs}_b{b:g}", which) for b in bs]
            ax[j].errorbar(bs, [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", color=cols.get(hs, "k"), capsize=3, lw=1.8, label=f"adjacency kernel @hs{hs}")
        for ref, ls, lab2 in (("plane_hs32_b3", "--", "plane port @hs32 β3"), ("qk_late_b4", ":", "4-reader port @hs26 β4 (Part III)"), ("own_b2", "-.", "own plane @hs26 β2")):
            ax[j].axhline(m(ref, which).mean(), color="gray", ls=ls, lw=1, label=lab2)
        ax[j].axhline(0, color="k", lw=0.6); ax[j].set(xlabel="beta", title=lab); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Round 5: adjacency-kernel port (all-late form), injection layer × strength ({nc} contexts)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/r5a_adj_grid.png", dpi=150)

# ---------------- B: unseen edge
if os.path.exists(f"{RES}/edge5.json"):
    rows = json.load(open(f"{RES}/edge5.json"))["rows"]; conds = [c for c in dict.fromkeys(r["cond"] for r in rows)]
    keys = sorted({(r["ctx"], r["src"]) for r in rows})
    def vec(c, k): return np.array([[r[k] for r in rows if r["cond"] == c and (r["ctx"], r["src"]) == key][0] for key in keys])
    base = {k: vec("none", k) for k in ("P_heldout", "P_seen_nbr", "P_d2")}
    LAB = {"none": "no injection", "adj_hs34_b6": "adjacency port @34 β6\n(edge IN kernel)", "adj_hs34_b6_minus": "adjacency port @34 β6\n(edge DELETED from kernel)", "adj_hs34_b4": "adjacency port @34 β4\n(edge IN)",
           "adj_hs34_b4_minus": "adjacency port @34 β4\n(edge DELETED)", "plane_hs32_b3": "plane port @32 β3", "qk_late_b4": "4-reader port @26 β4\n(Part III)", "own_b2": "own plane @26 β2"}
    lines = [f"[edge5] n items = {len(keys)} (contexts x directions)", f"{'condition':22s} P(held-out edge)  P(seen nbr, each)  P(dist-2, each)  held-out − dist2 [paired Δ vs none, p]   held-out/seen   acc≥300"]
    for c in conds:
        ho, sn, d2, acc = vec(c, "P_heldout"), vec(c, "P_seen_nbr"), vec(c, "P_d2"), vec(c, "acc"); con = ho - d2; dcon = con - (base["P_heldout"] - base["P_d2"])
        p = stats.ttest_1samp(dcon, 0).pvalue if c != "none" else stats.ttest_1samp(con, 0).pvalue
        lines.append(f"{c:22s} {ho.mean():.3f}±{sem(ho):.3f}      {sn.mean():.3f}             {d2.mean():.3f}           {con.mean():+.3f} [{dcon.mean():+.3f}, p={p:.1g}]           {ho.mean() / sn.mean():.2f}       {acc.mean():.3f}")
    open(f"{FIG}/edge5_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5)); x = np.arange(len(conds)); w = 0.27
    for j, (k, lab, col, hatch) in enumerate((("P_seen_nbr", "a SEEN neighbor of X (per node)", "#9ecae1", ""), ("P_heldout", "the HELD-OUT neighbor Y (edge never in context)", "tab:green", ""), ("P_d2", "a non-neighbor at distance 2 (per node)", "#d9d9d9", "//"))):
        v = [vec(c, k) for c in conds]; ax[0].bar(x + (j - 1) * w, [a.mean() for a in v], w, yerr=[sem(a) for a in v], color=col, hatch=hatch, capsize=2, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels([LAB.get(c, c) for c in conds], rotation=30, ha="right", fontsize=7.5); ax[0].set(ylabel="P(next token) at queries on X (tokens ≥ 300)", title=f"Unseen edge with the late adjacency port (n = {len(keys)} items)"); ax[0].legend(fontsize=7.5)
    vals = []; errs = []
    for c in conds:
        d = (vec(c, "P_heldout") - vec(c, "P_d2")) - (base["P_heldout"] - base["P_d2"]); vals.append(d.mean()); errs.append(sem(d))
    ax[1].bar(x, vals, 0.6, yerr=errs, color=["k", "#08519c", "#fdae6b", "#4292c6", "#fdd0a2", "#006d2c", "#74c476", "gray"][:len(conds)], capsize=3)
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_xticks(x); ax[1].set_xticklabels([LAB.get(c, c) for c in conds], rotation=30, ha="right", fontsize=7.5)
    ax[1].set(ylabel="Δ [P(held-out Y) − P(dist-2 non-neighbor)] vs no injection", title="Does the channel supply the missing edge? (paired)")
    fig.tight_layout(); fig.savefig(f"{FIG}/r5b_unseen_edge.png", dpi=150)
print(f"figures -> {FIG}")
