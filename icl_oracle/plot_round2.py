"""Figures for zoo round 2 (results_zoo/round2.json -> figures_zoo/r2_*.png)."""
import json, os, sys, re
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_zoo"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_zoo"; os.makedirs(FIG, exist_ok=True)
R = json.load(open(f"{RES}/round2.json")); rows, un, meta = R["rows"], R["unseen"], R["meta"]
base = {r["ctx"]: r for r in rows if r["cand"] == "none"}; bun = {(u["ctx"], u["X"]): u for u in un if u["cand"] == "none"}
cands = [c for c in dict.fromkeys(r["cand"] for r in rows) if c != "none"]; ctxs = sorted(base)


def m(c, which):
    if which == "uptake": return np.array([r["up_early"] - base[r["ctx"]]["up_early"] for r in rows if r["cand"] == c])
    if which == "uptake_late": return np.array([r["up_late"] - base[r["ctx"]]["up_late"] for r in rows if r["cand"] == c])
    if which == "self": return np.array([r["self_late"] - base[r["ctx"]]["self_late"] for r in rows if r["cand"] == c])
    if which == "override": return np.array([r["ov_inj"] - r["ov_chance"] for r in rows if r["cand"] == c])
    if which == "unseen": return np.array([u["trueX_X"] - bun[(u["ctx"], u["X"])]["P_nbrs_X"] for u in un if u["cand"] == c])
    if which == "claimY": return np.array([u["claimY_Y"] - bun[(u["ctx"], u["X"])]["P_nbrs_Y"] for u in un if u["cand"] == c])


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
def pv(v): return stats.ttest_1samp(v, 0).pvalue if len(v) > 2 and np.std(v) > 0 else np.nan


METRICS = (("uptake", "Δ mass, tokens 100–300", "#74c476"), ("uptake_late", "Δ mass, tokens 400–1200", "#238b45"), ("override", "override: injected-map − chance", "tab:orange"),
           ("unseen", "unseen X: Δ P(true nbrs)", "tab:green"), ("claimY", "fake coord: Δ P(nbrs of Y)", "tab:red"))
lines = [f"n contexts = {len(ctxs)}; unseen items = {len(bun)}; beta 4 unless noted", f"{'condition':28s} " + " ".join(f"{lab[:26]:>28s}" for _, lab, _ in METRICS) + "     self"]
for c in cands:
    lines.append(f"{c:28s} " + " ".join(f"{m(c, w).mean():+.3f}±{sem(m(c, w)):.3f} p{pv(m(c, w)):.0e}" for w, _, _ in METRICS) + f"   {m(c, 'self').mean():+.3f}")
lines.append("own-plane energy inside top-k eigen-directions of S_all (mean over contexts) / of the late-reader form; random plane; QK-late plane:")
ks = meta["ks"]
for lab_, src in (("own", None), ("rand", "rand"), ("qk_late", "qk_late")):
    for form in ("all", "late"):
        if src is None: vals = [np.mean([o[form][str(k)] if str(k) in o[form] else o[form][k] for o in R["own_energy"]]) for k in ks]
        else: vals = [meta["energy_ref"][src][form][str(k)] if str(k) in meta["energy_ref"][src][form] else meta["energy_ref"][src][form][k] for k in ks]
        lines.append(f"  {lab_:8s} {form:5s} " + " ".join(f"k{k}:{v:.2f}" for k, v in zip(ks, vals)))
lines.append(f"own plane: QK score (all-late form) {np.mean([o['qk_all'] for o in R['own_energy']]):.3f}, (late-reader form) {np.mean([o['qk_late'] for o in R['own_energy']]):.3f}; cos vs qk_late plane {np.mean([o['cos_qk_late'] for o in R['own_energy']]):.2f}; vs S_all top-2 {np.mean([o['cos_all_top2'] for o in R['own_energy']]):.2f}")
lines.append(f"S_all: {meta['n_pos']} positive eigenvalues of 3584; top eigenvalues {np.round(meta['ev_all_top'][-6:], 3).tolist()}; cos(qk_late, S_all top-2) = {meta['cos_late_all']:.2f}")
open(f"{FIG}/r2_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))

# ---------------------------------------------------------------- Fig A: spectrum walk
spec = [(c, int(re.match(r"spec_top(\d+)_", c).group(1))) for c in cands if c.startswith("spec_top")]
spec.sort(key=lambda t: t[1]); bot = [c for c in cands if c.startswith("spec_bot")]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for j, (which, lab, col) in enumerate((METRICS[0], METRICS[2], METRICS[3])):
    x = [r for _, r in spec]; v = [m(c, which) for c, _ in spec]
    ax[j].errorbar(x, [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", color=col, capsize=3, lw=1.8, label="plane at eigen-rank (r, r+1) of S_all, hs 26, β4")
    for i, c in enumerate(bot):
        v = m(c, which); ax[j].errorbar([4000 * (i + 1)], [v.mean()], yerr=[sem(v)], marker="v", color="tab:red", capsize=3, ls="", label="bottom of spectrum (anti-port)" if i == 0 else None)
    ax[j].axhline(m("qk_late_b4", which).mean(), color="#006d2c", ls="--", lw=1, label="QK port (4 reader heads)")
    ax[j].axhline(m("own_b2", which).mean(), color="k", ls=":", lw=1, label="own plane (calibrated, β2)")
    ax[j].axhline(0, color="k", lw=0.6); ax[j].set(xscale="log", xlabel="eigen-rank from the top of the QK spectrum (all 256 late heads)", title=lab); ax[j].grid(alpha=0.3)
    ax[j].set_xticks([1, 3, 9, 33, 129, 513, 2049]); ax[j].set_xticklabels(["1–2", "3–4", "9–10", "33–34", "129–130", "513–514", "2049–50"], fontsize=7)
ax[0].legend(fontsize=7)
fig.suptitle("Walking down the QK spectrum: does only the top plane work, or the whole positive cone?", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/r2a_spectrum.png", dpi=150)

# ---------------------------------------------------------------- Fig B: injection layer (single + multi)
single = [(c, int(re.match(r"all_top2_hs(\d+)$", c).group(1))) for c in cands if re.match(r"all_top2_hs\d+$", c)] + [("spec_top1_2", 26)]
single.sort(key=lambda t: t[1]); multi = [c for c in cands if "+" in c and c.startswith("all_top2")]
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
for j, (which, lab, col) in enumerate((METRICS[0], METRICS[1], METRICS[2], METRICS[3])):
    x = [h for _, h in single]; v = [m(c, which) for c, _ in single]
    ax[j].errorbar(x, [a.mean() for a in v], yerr=[sem(a) for a in v], marker="o", color=col, capsize=3, lw=1.8, label="S_all top-2 plane, one layer, β4")
    for i, c in enumerate(multi):
        v = m(c, which); ax[j].errorbar([38 + 3 * i], [v.mean()], yerr=[sem(v)], marker="s", color="tab:purple", capsize=3, ls="")
        ax[j].annotate(c.replace("all_top2_", ""), (38 + 3 * i, v.mean()), fontsize=6, rotation=90, xytext=(0, 6), textcoords="offset points", ha="center")
    ax[j].axhline(m("qk_late_b4", which).mean(), color="#006d2c", ls="--", lw=1, label="QK port (4 reader heads) @hs26"); ax[j].axhline(m("own_b2", which).mean(), color="k", ls=":", lw=1, label="own plane β2 @hs26")
    ax[j].axhline(0, color="k", lw=0.6); ax[j].set(xlabel="hidden state of injection (squares: several layers at once)", title=lab); ax[j].grid(alpha=0.3)
ax[0].legend(fontsize=7)
fig.suptitle("Where to inject the weight-defined port: single-layer sweep and multi-layer injection", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/r2b_layer.png", dpi=150)

# ---------------------------------------------------------------- Fig C: own plane inside the QK cone
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for j, form in enumerate(("all", "late")):
    own = np.array([[o[form][str(k)] if str(k) in o[form] else o[form][k] for k in ks] for o in R["own_energy"]])
    ax[j].errorbar(ks, own.mean(0), yerr=own.std(0, ddof=1) / np.sqrt(len(own)), marker="o", color="tab:blue", capsize=3, lw=1.8, label="model's OWN plane (per context)")
    for src, col, lab in (("rand", "tab:purple", "random plane"), ("qk_late", "#006d2c", "QK port plane (4 reader heads)")):
        e = meta["energy_ref"][src][form]; ax[j].plot(ks, [e[str(k)] if str(k) in e else e[k] for k in ks], marker="s", color=col, lw=1.2, label=lab)
    ax[j].plot(ks, np.array(ks) / 3584, color="gray", ls="--", lw=1, label="chance (k / 3584)")
    ax[j].set(xscale="log", xlabel="k (top-k eigen-directions of the QK form)", ylabel="fraction of the plane's energy inside the top-k subspace", title=("all 256 late heads (S_all)" if form == "all" else "4 late reader heads"), ylim=(0, 1.02)); ax[j].grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.suptitle("Is the native position plane inside the QK-positive cone?", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/r2c_own_in_cone.png", dpi=150)

# ---------------------------------------------------------------- Fig D: dimensionality
dims = [c for c in ("spec_top1_2", "all_top12+34", "all_top6_3planes", "all_top3_Z3", "qk_late_b4", "own_b2") if c in cands]
LAB = {"spec_top1_2": "top-2 plane (S_all)", "all_top12+34": "same coords on planes 1–2 AND 3–4", "all_top6_3planes": "same coords on 3 planes (1–6)", "all_top3_Z3": "3 coords on top-3 directions", "qk_late_b4": "QK port (4 readers)", "own_b2": "own plane β2"}
fig, ax = plt.subplots(figsize=(11, 4.6)); x = np.arange(len(dims)); w = 0.16
for j, (which, lab, col) in enumerate(METRICS):
    v = [m(c, which) for c in dims]; ax.bar(x + (j - 2) * w, [a.mean() for a in v], w, yerr=[sem(a) for a in v], color=col, capsize=2, label=lab)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in dims], fontsize=8); ax.set(ylabel="Δ vs no injection", title="Spreading the same total norm over more QK directions (hs 26, β4)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/r2d_dims.png", dpi=150)
print(f"figures -> {FIG}")
