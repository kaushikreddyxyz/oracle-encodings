"""Figures + summary table for the subspace zoo (results_zoo/zoo.json [+ sweep.json] -> figures_zoo/)."""
import json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_zoo"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_zoo"; os.makedirs(FIG, exist_ok=True)
Z = json.load(open(f"{RES}/zoo.json")); rows, un, geo, meta = Z["rows"], Z["unseen"], Z["geometry"], Z["meta"]
ctxs = sorted({r["ctx"] for r in rows}); betas = meta["betas"]; B = betas[-1]
FAM_COL = {"qk": "#006d2c", "vocab": "tab:blue", "vocab_qk": "#6baed6", "act": "tab:purple", "emb16": "tab:orange", "weight": "tab:brown", "ref": "k"}
FAM_LAB = {"qk": "QK ports (weights)", "vocab": "vocabulary-embedding PCs", "vocab_qk": "QK form restricted to vocab PCs", "act": "activation PCs / foreign plane",
           "emb16": "PCs of the 16 word embeddings", "weight": "MLP-in / attn-out singular dirs", "ref": "references"}
base_row = {r["ctx"]: r for r in rows if r["cand"] == "none"}
base_un = {(u["ctx"], u["X"]): u for u in un if u["cand"] == "none"}
cands = [c for c in dict.fromkeys(r["cand"] for r in rows) if c != "none"]
group = {r["cand"]: r["group"] for r in rows}


def metric(c, b, which):
    """paired per-context (or per-item) deltas vs no injection."""
    if which == "uptake":
        return np.array([r["up_early"] - base_row[r["ctx"]]["up_early"] for r in rows if r["cand"] == c and r["beta"] == b])
    if which == "uptake_late":
        return np.array([r["up_late"] - base_row[r["ctx"]]["up_late"] for r in rows if r["cand"] == c and r["beta"] == b])
    if which == "override":
        return np.array([r["ov_inj"] - r["ov_chance"] for r in rows if r["cand"] == c and r["beta"] == b])
    if which == "ov_true":
        return np.array([r["ov_true"] - base_row[r["ctx"]]["ov_true"] for r in rows if r["cand"] == c and r["beta"] == b])
    if which == "self":
        return np.array([r["self_late"] - base_row[r["ctx"]]["self_late"] for r in rows if r["cand"] == c and r["beta"] == b])
    if which == "unseen":
        return np.array([u["trueX_X"] - base_un[(u["ctx"], u["X"])]["P_nbrs_X"] for u in un if u["cand"] == c and u["beta"] == b])
    if which == "claimY":
        return np.array([u["claimY_Y"] - base_un[(u["ctx"], u["X"])]["P_nbrs_Y"] for u in un if u["cand"] == c and u["beta"] == b])
    raise KeyError(which)


def gmean(c, key):
    v = [geo[k][c][key] for k in ctxs if c in geo[k] and key in geo[k][c]]
    return float(np.mean(v)) if v else np.nan


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def pval(v): return stats.ttest_1samp(v, 0).pvalue if len(v) > 2 and np.std(v) > 0 else np.nan


# ------------------------------------------------------------------ summary table
order = sorted(cands, key=lambda c: (list(FAM_COL).index(group[c]), -metric(c, B, "override").mean()))
lines = [f"n contexts = {len(ctxs)}; unseen items = {len(base_un)}; beta = {B:g}   (deltas are paired vs no injection; override = injected-map mass - chance)",
         f"{'candidate':18s} {'family':9s} {'uptake':>15s} {'override':>15s} {'unseen X':>15s} {'claim-Y':>15s} {'self':>8s} | {'cos own':>7s} {'cos QK':>7s} {'QKscore':>7s} {'leak':>5s} {'embAl':>5s} {'inVoc':>5s}"]
for c in order:
    up, ov, ux, cy, sf = (metric(c, B, w) for w in ("uptake", "override", "unseen", "claimY", "self"))
    cq = np.mean([geo[k]["_cos_pairs"][c].get("qk_late", np.nan) for k in ctxs if c in geo[k]["_cos_pairs"]])
    lines.append(f"{c:18s} {group[c]:9s} {up.mean():+.3f}±{sem(up):.3f} p{pval(up):.0e} {ov.mean():+.3f}±{sem(ov):.3f} p{pval(ov):.0e} {ux.mean():+.3f}±{sem(ux):.3f} p{pval(ux):.0e} "
                 f"{cy.mean():+.3f}±{sem(cy):.3f} p{pval(cy):.0e} {sf.mean():+.3f} | {gmean(c, 'cos_own'):7.2f} {cq:7.2f} {gmean(c, 'qk_late'):7.2f} {gmean(c, 'leak'):5.2f} {gmean(c, 'emb_align'):5.2f} {gmean(c, 'in_vocab64'):5.2f}")
open(f"{FIG}/zoo_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))

# ------------------------------------------------------------------ Fig 1: effects per candidate (beta B), by family
fig, ax = plt.subplots(1, 4, figsize=(20, 0.32 * len(order) + 2.5), sharey=True)
y = np.arange(len(order))[::-1]
for j, (which, lab) in enumerate((("uptake", "faster learning: Δ rule-following mass, tokens 100–300"), ("override", "override: mass on INJECTED-map neighbors − chance"),
                                  ("unseen", "unseen node: Δ P(true neighbors of X)"), ("claimY", "unseen node, fake coordinate: Δ P(neighbors of Y)"))):
    v = [metric(c, B, which) for c in order]
    ax[j].barh(y, [a.mean() for a in v], xerr=[sem(a) for a in v], color=[FAM_COL[group[c]] for c in order], capsize=2, height=0.7)
    ax[j].axvline(0, color="k", lw=0.7)
    for ref, ls in (("qk_late", "--"), ("own", ":"), ("rand", "-.")):
        if ref in order: ax[j].axvline(metric(ref, B, which).mean(), color="gray", ls=ls, lw=0.9)
    ax[j].set_title(lab, fontsize=9); ax[j].grid(axis="x", alpha=0.3)
ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COL[f]) for f in FAM_COL] + [plt.Line2D([0], [0], color="gray", ls=ls) for ls in ("--", ":", "-.")]
ax[3].legend(handles, [FAM_LAB[f] for f in FAM_COL] + ["QK port (reference)", "own plane (calibrated)", "random plane"], fontsize=7, loc="lower right")
fig.suptitle(f"Subspace zoo at hidden state 26, beta {B:g}: which planes accept injected grid coordinates? ({len(ctxs)} contexts, paired vs no injection)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/zoo1_effects.png", dpi=150)

# ------------------------------------------------------------------ Fig 2: effect vs geometry
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for j, (xkey, xlab) in enumerate((("qk_late", "QK score of the plane (uᵀSu / λmax, late-reader form)"), ("cos_own", "cos principal angle vs the context's OWN plane (mean)"), ("emb_align", "mean |cos| with the 16 word embeddings"))):
    for c in order:
        ov = metric(c, B, "override"); x = gmean(c, xkey)
        ax[j].errorbar(x, ov.mean(), yerr=sem(ov), fmt="o", color=FAM_COL[group[c]], ms=5, capsize=2, alpha=0.9)
        if abs(ov.mean()) > 0.03 or c in ("rand", "qk_anti", "mlp_in26", "vocab_pc1_2", "emb16_pc1_2"): ax[j].annotate(c, (x, ov.mean()), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax[j].axhline(0, color="k", lw=0.7); ax[j].set(xlabel=xlab, ylabel="override: injected-map mass − chance"); ax[j].grid(alpha=0.3)
handles = [plt.Line2D([0], [0], marker="o", color=FAM_COL[f], ls="") for f in FAM_COL]
ax[0].legend(handles, [FAM_LAB[f] for f in FAM_COL], fontsize=7)
fig.suptitle("Does causal control track the plane's QK score, its overlap with the model's own plane, or its embedding alignment?", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/zoo2_geometry.png", dpi=150)

# ------------------------------------------------------------------ Fig 3: overlap heatmap (cos principal angles)
show = [c for c in order if group[c] in ("qk", "vocab_qk", "ref") or c in ("vocab_pc1_2", "vocab_pc2_3", "act_pc1_2", "act_pc2_3", "act_qk64", "foreign_set0", "emb16_qk", "emb16_pc1_2", "mlp_in26", "attn_out25")]
agn = [c for c in show if c in meta["cos_pairs"]]
M = np.full((len(show), len(show)), np.nan)
for i, a in enumerate(show):
    for j, b in enumerate(show):
        if a in meta["cos_pairs"] and b in meta["cos_pairs"]: M[i, j] = meta["cos_pairs"][a][b]
        else:
            vals = [geo[k]["_cos_pairs"][a][b] for k in ctxs if a in geo[k]["_cos_pairs"] and b in geo[k]["_cos_pairs"][a]]
            vals += [geo[k]["_cos_pairs"][b][a] for k in ctxs if b in geo[k]["_cos_pairs"] and a in geo[k]["_cos_pairs"][b]]
            if vals: M[i, j] = np.mean(vals)
            elif a == b: M[i, j] = 1.0
fig, ax = plt.subplots(figsize=(0.5 * len(show) + 3, 0.5 * len(show) + 2))
im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis"); ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=90, fontsize=7); ax.set_yticks(range(len(show))); ax.set_yticklabels(show, fontsize=7)
for i in range(len(show)):
    for j in range(len(show)):
        if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if M[i, j] < 0.6 else "k")
plt.colorbar(im, ax=ax, fraction=0.03, label="mean cos of principal angles (1 = same plane, 0 = orthogonal)")
ax.set_title("Overlap between candidate planes (context-specific planes: mean over contexts; blank = not comparable)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/zoo3_overlap.png", dpi=150)

# ------------------------------------------------------------------ Fig 4: beta sweep
if os.path.exists(f"{RES}/sweep.json"):
    S = json.load(open(f"{RES}/sweep.json")); srows, sun, sb = S["rows"], S["unseen"], S["meta"]["betas"]
    sbase = {r["ctx"]: r for r in srows if r["cand"] == "none"}; sbun = {(u["ctx"], u["X"]): u for u in sun if u["cand"] == "none"}
    scands = [c for c in dict.fromkeys(r["cand"] for r in srows) if c != "none"]
    def sm(c, b, which):
        if which == "uptake": return np.array([r["up_early"] - sbase[r["ctx"]]["up_early"] for r in srows if r["cand"] == c and r["beta"] == b])
        if which == "uptake_late": return np.array([r["up_late"] - sbase[r["ctx"]]["up_late"] for r in srows if r["cand"] == c and r["beta"] == b])
        if which == "override": return np.array([r["ov_inj"] - r["ov_chance"] for r in srows if r["cand"] == c and r["beta"] == b])
        if which == "unseen": return np.array([u["trueX_X"] - sbun[(u["ctx"], u["X"])]["P_nbrs_X"] for u in sun if u["cand"] == c and u["beta"] == b])
        if which == "claimY": return np.array([u["claimY_Y"] - sbun[(u["ctx"], u["X"])]["P_nbrs_Y"] for u in sun if u["cand"] == c and u["beta"] == b])
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.5))
    SC = {"own": "k", "qk_late": "#74c476", "rand": "tab:purple", "qk_alllate": "#006d2c", "qk_L26all": "tab:red", "vocab_qk1024": "tab:blue", "act_qk64": "tab:brown"}
    for i, c in enumerate(scands):
        col = SC.get(c, "tab:gray")
        for j, (which, lab) in enumerate((("uptake", "Δ mass tokens 100–300 (faster learning)"), ("uptake_late", "Δ mass tokens 400–1200 (does it hurt late?)"), ("override", "override: injected-map mass − chance"), ("unseen", "unseen X: Δ P(true neighbors)"), ("claimY", "fake coordinate: Δ P(neighbors of Y)"))):
            v = [sm(c, b, which) for b in sb]
            ax[j].errorbar(sb, [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", ms=4, color=col, label=c, capsize=2, lw=1.5 if c in ("qk_late", "own", "rand") else 1)
            ax[j].set(xlabel="beta (injection strength, × reference norm)", title=lab); ax[j].axhline(0, color="k", lw=0.6); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Strength sweep for the best grid-agnostic planes vs references ({len(sbase)} contexts, hidden state 26)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/zoo4_sweep.png", dpi=150)
    print("sweep candidates:", scands)
print(f"figures -> {FIG}")
