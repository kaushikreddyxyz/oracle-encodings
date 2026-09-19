"""Part IV story figure (results_zoo/{zoo,round2,round3,round4}.json -> figures_zoo/story_zoo.png).
Panel A: the zoo at a glance — override vs QK score for every plane. Panel B: the cone — effect vs eigen-rank. Panel C: the native
plane is half inside the cone (energy vs k) and its in-cone part carries the effect (inset bars). Panel D: how far the port goes —
old 4-reader port, best plane port, late adjacency port vs the calibrated own plane."""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES, FIG = f"{HERE}/results_zoo", f"{HERE}/figures_zoo"


def loader(name):
    R = json.load(open(f"{RES}/{name}.json")); rows, un = R["rows"], R["unseen"]
    base = {r["ctx"]: r for r in rows if r["cand"] == "none"}; bun = {(u["ctx"], u["X"]): u for u in un if u["cand"] == "none"}
    def m(c, which, b=None):
        f = (lambda r: r["cand"] == c and (b is None or r["beta"] == b))
        if which == "uptake": return np.array([r["up_early"] - base[r["ctx"]]["up_early"] for r in rows if f(r)])
        if which == "uptake_late": return np.array([r["up_late"] - base[r["ctx"]]["up_late"] for r in rows if f(r)])
        if which == "override": return np.array([r["ov_inj"] - r["ov_chance"] for r in rows if f(r)])
        if which == "unseen": return np.array([u["trueX_X"] - bun[(u["ctx"], u["X"])]["P_nbrs_X"] for u in un if f(u)])
        if which == "claimY": return np.array([u["claimY_Y"] - bun[(u["ctx"], u["X"])]["P_nbrs_Y"] for u in un if f(u)])
    return R, m


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v))


Z, mz = loader("zoo"); R2, m2 = loader("round2"); R3, m3 = loader("round3"); R4, m4 = loader("round4"); R5, m5 = loader("round5")
FAM_COL = {"qk": "#006d2c", "vocab": "tab:blue", "vocab_qk": "#6baed6", "act": "tab:purple", "emb16": "tab:orange", "weight": "tab:brown", "ref": "k"}
FAM_LAB = {"qk": "QK ports (attention weights)", "vocab": "vocabulary-embedding PCs", "vocab_qk": "QK form inside vocab PCs", "act": "activation PCs / foreign plane", "emb16": "PCs of the 16 word embeddings", "weight": "MLP-in / attn-out SVD", "ref": "own plane / random"}
group = {r["cand"]: r["group"] for r in Z["rows"]}; geo = Z["geometry"]; ctxs = sorted(geo)
cands = [c for c in dict.fromkeys(r["cand"] for r in Z["rows"]) if c != "none"]
gm = lambda c, k: np.mean([geo[x][c][k] for x in ctxs if c in geo[x]])

fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
# ---- A
a = ax[0, 0]
for c in cands:
    ov = mz(c, "override", 4.0); x = gm(c, "qk_late")
    a.errorbar(x, ov.mean(), yerr=sem(ov), fmt="o", color=FAM_COL[group[c]], ms=5, capsize=2)
    if ov.mean() > 0.06 or c in ("rand", "qk_anti", "mlp_in26", "vocab_pc1_2", "emb16_pc1_2", "act_pc1_2", "qk_mid4"): a.annotate(c, (x, ov.mean()), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
a.axhline(0, color="k", lw=0.6); a.set(xlabel="QK score of the plane  (mean uᵀSu / λmax over the plane; late-reader form)", ylabel="override: injected-map mass − chance (β4, hs 26)", title="A. 47 candidate planes: control tracks the QK score (ρ = 0.67)")
a.legend([plt.Line2D([0], [0], marker="o", color=FAM_COL[f], ls="") for f in FAM_COL], [FAM_LAB[f] for f in FAM_COL], fontsize=7, loc="upper left"); a.grid(alpha=0.3)
# ---- B
b = ax[0, 1]
spec = sorted([(int(re.match(r"spec_top(\d+)_", c).group(1)), c) for c in dict.fromkeys(r["cand"] for r in R2["rows"]) if c.startswith("spec_top")])
for which, lab, col in (("override", "override (injected-map − chance)", "tab:orange"), ("unseen", "unseen node: Δ P(true neighbors)", "tab:green"), ("uptake", "faster learning: Δ mass, tokens 100–300", "#74c476")):
    v = [m2(c, which) for _, c in spec]; b.errorbar([r for r, _ in spec], [q.mean() for q in v], yerr=[sem(q) for q in v], marker="o", color=col, capsize=3, lw=1.8, label=lab)
for i, c in enumerate(("spec_bot3_4", "spec_bot1_2")):
    for which, col in (("override", "tab:orange"), ("unseen", "tab:green"), ("uptake", "#74c476")):
        v = m2(c, which); b.errorbar([4500 * (i + 1)], [v.mean()], yerr=[sem(v)], marker="v", color=col, capsize=2, ls="")
b.axhline(0, color="k", lw=0.6); b.set(xscale="log", xlabel="eigen-rank of the plane in the all-late-heads QK spectrum (▼ = bottom of spectrum)", ylabel="Δ vs no injection (β4, hs 26)", title="B. The port is a cone: the top ≈ 20–30 directions all work, then it fades")
b.set_xticks([1, 3, 9, 33, 129, 513, 2049]); b.set_xticklabels(["1–2", "3–4", "9–10", "33–34", "129–130", "513–514", "2049–50"], fontsize=8); b.legend(fontsize=7.5); b.grid(alpha=0.3)
# ---- C
c_ = ax[1, 0]; ks = R2["meta"]["ks"]; own = np.array([[o["all"][str(k)] if str(k) in o["all"] else o["all"][k] for k in ks] for o in R2["own_energy"]])
c_.errorbar(ks, own.mean(0), yerr=own.std(0, ddof=1) / np.sqrt(len(own)), marker="o", color="tab:blue", capsize=3, lw=2, label="model's OWN plane (per context)")
e = R2["meta"]["energy_ref"]["qk_late"]["all"]; c_.plot(ks, [e[str(k)] if str(k) in e else e[k] for k in ks], marker="s", color="#006d2c", lw=1.2, label="4-reader port plane")
e = R2["meta"]["energy_ref"]["rand"]["all"]; c_.plot(ks, [e[str(k)] if str(k) in e else e[k] for k in ks], marker="s", color="tab:purple", lw=1.2, label="random plane")
c_.plot(ks, np.array(ks) / 3584, color="gray", ls="--", lw=1, label="chance (k / 3584)")
c_.set(xscale="log", xlabel="k = number of top eigen-directions of the all-late QK form", ylabel="fraction of the plane's energy inside the top-k cone", ylim=(0, 1.02), title="C. The native plane is half inside the cone — and that half does the work")
c_.legend(fontsize=7.5, loc="upper left"); c_.grid(alpha=0.3)
ins = c_.inset_axes([0.55, 0.24, 0.43, 0.40]); conds = ["own_b2", "own_in64_b2", "own_out64_b2", "own_in64_full_b2"]; labs = ["own\nplane", "in-cone\npart (61%)", "out-of-cone\npart", "in-cone,\nfull norm"]
x = np.arange(4); w = 0.38
for j, (which, lab, col) in enumerate((("override", "override", "tab:orange"), ("unseen", "unseen node", "tab:green"))):
    v = [m3(c, which) for c in conds]; ins.bar(x + (j - 0.5) * w, [q.mean() for q in v], w, yerr=[sem(q) for q in v], color=col, capsize=2, label=lab)
ins.set_xticks(x); ins.set_xticklabels(labs, fontsize=6.5); ins.tick_params(axis="y", labelsize=7); ins.set_title("own plane split at k = 64 (β2)", fontsize=8); ins.legend(fontsize=6.5); ins.axhline(0, color="k", lw=0.5)
# ---- D
d = ax[1, 1]
conds = [("own_b2", m4, "own plane\n(calibrated, β2, hs 26)"), ("qk_late_b4", m4, "4-reader QK port\n(Part III: β4, hs 26)"), ("all_top2_hs32_b3", m4, "all-late plane port\n(β3, hs 32)"), ("adj_all_hs34", m4, "adjacency-kernel port\n(β4, hs 34)"), ("adj_hs32_b6", m5, "adjacency-kernel port\n(β6, hs 32)")]
x = np.arange(len(conds)); w = 0.16
for j, (which, lab, col) in enumerate((("uptake", "faster learning (tokens 100–300)", "#74c476"), ("uptake_late", "late accuracy (tokens 400–1200)", "#238b45"), ("override", "override (injected-map − chance)", "tab:orange"), ("unseen", "unseen node: Δ P(true nbrs)", "tab:green"), ("claimY", "fake coordinate: Δ P(nbrs of Y)", "tab:red"))):
    v = [mm(c, which) for c, mm, _ in conds]; d.bar(x + (j - 2) * w, [q.mean() for q in v], w, yerr=[sem(q) for q in v], color=col, capsize=2, label=lab)
d.axhline(0, color="k", lw=0.6); d.set_xticks(x); d.set_xticklabels([l for _, _, l in conds], fontsize=8); d.set(ylabel="Δ vs no injection (15 contexts)", title="D. How far a weight-defined, grid-agnostic port can go"); d.legend(fontsize=7.5, loc="upper left"); d.grid(axis="y", alpha=0.3)
fig.suptitle("Part IV — the subspace zoo: the only planes that accept injected coordinates are slices of one low-dimensional QK cone, which the model's own plane half-occupies", fontsize=11.5, y=0.995)
fig.tight_layout(); fig.savefig(f"{FIG}/story_zoo.png", dpi=150, bbox_inches="tight"); print("ok")
