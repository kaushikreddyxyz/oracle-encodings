"""Figures for zoo rounds 3 and 4 (results_zoo/round3.json, round4.json -> figures_zoo/r3_*.png, r4_*.png)."""
import json, os, sys, re
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_zoo"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_zoo"; os.makedirs(FIG, exist_ok=True)
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
        if which == "ov_true": return np.array([r["ov_true"] - base[r["ctx"]]["ov_true"] for r in rows if r["cand"] == c])
        if which == "unseen": return np.array([u["trueX_X"] - bun[(u["ctx"], u["X"])]["P_nbrs_X"] for u in un if u["cand"] == c])
        if which == "claimY": return np.array([u["claimY_Y"] - bun[(u["ctx"], u["X"])]["P_nbrs_Y"] for u in un if u["cand"] == c])
    cands = [c for c in dict.fromkeys(r["cand"] for r in rows) if c != "none"]
    return R, m, cands, len(base), len(bun)


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
def pv(v): return stats.ttest_1samp(v, 0).pvalue if len(v) > 2 and np.std(v) > 0 else np.nan


def table(name, m, cands, nc, nu, extra=()):
    lines = [f"[{name}] n contexts = {nc}; unseen items = {nu}", f"{'condition':26s} " + " ".join(f"{lab[:26]:>28s}" for _, lab, _ in METRICS) + "     self"]
    for c in cands: lines.append(f"{c:26s} " + " ".join(f"{m(c, w).mean():+.3f}±{sem(m(c, w)):.3f} p{pv(m(c, w)):.0e}" for w, _, _ in METRICS) + f"   {m(c, 'self').mean():+.3f}")
    lines += list(extra); open(f"{FIG}/{name}_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))


def bars(ax, m, conds, labels, title, metrics=METRICS, w=None):
    x = np.arange(len(conds)); w = w or 0.8 / len(metrics)
    for j, (which, lab, col) in enumerate(metrics):
        v = [m(c, which) for c in conds]; ax.bar(x + (j - (len(metrics) - 1) / 2) * w, [a.mean() for a in v], w, yerr=[sem(a) for a in v], color=col, capsize=2, label=lab)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right"); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)


# ================================================================== round 3
if os.path.exists(f"{RES}/round3.json"):
    R, m, cands, nc, nu = load("round3"); ks = R["meta"]["ks"]; geo = R["geometry"]
    fr = {k: np.mean([geo[c][f"frac_in{k}"] for c in geo]) for k in ks}
    extra = ["own-plane feature energy inside the top-k cone of S_all (mean over contexts): " + " ".join(f"k{k}:{fr[k]:.2f}" for k in ks),
             "own plane QK score: all-late form %.3f, late-reader form %.3f" % (np.mean([geo[c]["own_qk_all"] for c in geo]), np.mean([geo[c]["own_qk_late"] for c in geo])),
             "static geometry: " + json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3) for k, v in R["meta"]["geo_static"].items()})]
    table("r3", m, cands, nc, nu, extra)
    # Fig A: own split
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8), sharey=True)
    for j, k in enumerate(ks):
        conds = ["own_b2", f"own_in{k}_b2", f"own_out{k}_b2", f"own_in{k}_full_b2"]
        bars(ax[j], m, conds, ["own plane (β2)", f"in-cone part only\n({fr[k] * 100:.0f}% of energy)", "out-of-cone part only", "in-cone part, rescaled\nto full norm"], f"split at k = {k} top eigen-directions of S_all")
    ax[0].set_ylabel("Δ vs no injection"); ax[0].legend(fontsize=7)
    fig.suptitle("Dissecting the model's OWN plane: does it work because of its component inside the QK cone?", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/r3a_own_split.png", dpi=150)
    # Fig B: layer bands + act_qk sweep + vocab_qk
    fig, ax = plt.subplots(1, 2, figsize=(19, 4.8), sharey=True)
    conds = ["qk_late", "qk_alllate", "qk_L12_25", "qk_L26_33", "qk_L34_41", "qk_L12_41", "qk_all672"]
    bars(ax[0], m, conds, ["4 reader heads\n(L26,L28)", "all heads L26–41", "all heads L12–25", "all heads L26–33", "all heads L34–41", "all heads L12–41", "all 672 heads\n(L0–41)"], "which heads define the form S (plane injected at hs 26, β4)")
    conds = [c for c in ("act_qk8", "act_qk16", "act_qk32", "act_qk64", "act_qk128", "act_qk64_set1", "act_qk64_Sall", "vocab_qk2048", "vocab_qk3072") if c in cands]
    bars(ax[1], m, conds, [c.replace("act_qk", "act top-").replace("_set1", "\n(word set 1)").replace("_Sall", "\n(S_all form)").replace("vocab_qk", "vocab top-") for c in conds], "QK-best plane inside activation PCs (foreign word set) / vocabulary PCs")
    ax[0].set_ylabel("Δ vs no injection"); ax[0].legend(fontsize=7)
    fig.suptitle("Round 3: where the cone comes from (layers) and how little of the activation subspace is needed", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/r3b_bands_act.png", dpi=150)

# ================================================================== round 4
if os.path.exists(f"{RES}/round4.json"):
    R, m, cands, nc, nu = load("round4")
    table("r4", m, cands, nc, nu, ["geometry: " + json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in R["meta"]["geo"].items()})])
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    single = sorted([(int(re.match(r"all_top2_hs(\d+)$", c).group(1)), c) for c in cands if re.match(r"all_top2_hs\d+$", c)])
    matched = sorted([(int(re.match(r"ge(\d+)_top2_hs\d+", c).group(1)), c) for c in cands if c.startswith("ge")])
    for j, (which, lab, col) in enumerate((METRICS[0], METRICS[1], METRICS[2], METRICS[3])):
        v = [m(c, which) for _, c in single]; ax[j].errorbar([h for h, _ in single], [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", color=col, capsize=3, lw=1.8, label="S_all (L26–41) top-2 plane, β4")
        v = [m(c, which) for _, c in matched]; ax[j].errorbar([h for h, _ in matched], [a.mean() for a in v], yerr=[sem(a) for a in v], marker="s", color="tab:purple", capsize=3, lw=1.2, ls="--", label="plane from heads at layers ≥ injection layer")
        ax[j].axhline(m("qk_late_b4", which).mean(), color="#006d2c", ls="--", lw=1, label="4-reader QK port @hs26"); ax[j].axhline(m("own_b2", which).mean(), color="k", ls=":", lw=1, label="own plane β2 @hs26")
        ax[j].axhline(m("rand_hs34", which).mean(), color="gray", ls="-.", lw=1, label="random plane @hs34")
        ax[j].axhline(0, color="k", lw=0.6); ax[j].set(xlabel="hidden state of injection", title=lab); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("Round 4: the late-layer port (fine layer sweep, 15 contexts)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/r4a_layer_fine.png", dpi=150)
    fig, ax = plt.subplots(1, 2, figsize=(19, 4.8), sharey=True)
    conds = ["all_top2_hs32_b2", "all_top2_hs32_b3", "all_top2_hs32", "all_top2_hs32_b6", "all_top2_hs32_b8", "all_top2_hs34_b2", "all_top2_hs34_b3", "all_top2_hs34", "all_top2_hs34_b6", "all_top2_hs34_b8"]
    conds = [c for c in conds if c in cands]
    bars(ax[0], m, conds, [c.replace("all_top2_", "").replace("_b", " β") + ("" if "_b" in c else " β4") for c in conds], "strength at hs 32 / hs 34 (S_all top-2 plane)")
    conds = [c for c in ("qk_late_b4", "spec_top3_4_hs34", "spec_top9_10_hs34", "adj_all_hs26", "adj_all_hs30", "adj_all_hs34", "adj_all_hs34_b6", "own_b2", "rand_hs34") if c in cands]
    bars(ax[1], m, conds, ["4-reader port\n@26 β4", "rank 3–4 plane\n@34 β4", "rank 9–10 plane\n@34 β4", "adjacency kernel\n@26 β4", "adjacency kernel\n@30 β4", "adjacency kernel\n@34 β4", "adjacency kernel\n@34 β6", "own plane\n@26 β2", "random\n@34 β4"], "cone planes at hs 34; adjacency-kernel port (self ≈ 0)")
    ax[0].set_ylabel("Δ vs no injection"); ax[0].legend(fontsize=7)
    fig.suptitle("Round 4: strength and kernel choice for the late-layer port", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/r4b_strength_kernel.png", dpi=150)
print(f"figures -> {FIG}")
