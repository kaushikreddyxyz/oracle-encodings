"""Figures + threshold tables for gt_inject.py results (results_gt/) -> figures_gt/."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_gt"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_gt"
os.makedirs(FIG, exist_ok=True)
K = 25  # smoothing window (tokens)


def smooth(curves, k=K):
    M = np.array(curves, dtype=float)
    m = M.mean(0)
    sem = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros_like(m)
    sm = np.convolve(m, np.ones(k) / k, mode="valid")
    ssem = np.convolve(sem, np.ones(k) / k, mode="valid")
    t = np.arange(k, len(m) + 1)
    return t, sm, ssem


def thresholds(curves, ths=(0.5, 0.6, 0.65, 0.7), k=K):
    t, sm, _ = smooth(curves, k)
    out = {}
    for th in ths:
        idx = np.argmax(sm >= th)
        out[th] = int(t[idx]) if sm[idx] >= th else None
    return out


def plot_family(ax, items, title, xmax=None):
    for name, curves, col, ls in items:
        t, sm, ssem = smooth(curves)
        ax.plot(t, sm, ls, color=col, lw=2, label=name)
        ax.fill_between(t, sm - ssem, sm + ssem, color=col, alpha=0.12)
    ax.set(xscale="log", xlabel="context length (tokens)", ylabel="rule-following prob. mass",
           title=title, ylim=(0, 1))
    if xmax: ax.set_xlim(K, xmax)
    ax.legend(fontsize=7)


lines = []
# ============================================================================ E1
if os.path.exists(f"{RES}/e1.json"):
    e1 = json.load(open(f"{RES}/e1.json"))
    S = {int(h): v for h, v in e1["summary"].items()}
    hs = sorted(S)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hs, [S[h]["r2_in"] for h in hs], "k-", label="in-sample R² (full d)")
    ax[0].plot(hs, [S[h]["r2_in_pc"] for h in hs], "k--", label="in-sample R² (top-3 PCs)")
    ax[0].plot(hs, [S[h]["diff_set"]["cross"] for h in hs], "-", color="tab:green", label="cross-word-set R² (full d)")
    ax[0].plot(hs, [S[h]["diff_set"]["cross_pc"] for h in hs], "--", color="tab:green", label="cross-word-set R² (top-3 PCs)")
    ax[0].plot(hs, [S[h]["same_set_diff_assign"]["cross_pc"] for h in hs], "--", color="tab:orange", label="same words, diff assignment (PCs)")
    ax[0].axvline(e1["h_energy"], color="gray", ls=":", label=f"h_energy={e1['h_energy']}")
    ax[0].set(xlabel="hidden-state index", ylabel="R²", ylim=(-0.2, 1), title="Spectral coords → node reps: fit & transfer by layer")
    ax[0].legend(fontsize=7)
    ax2 = ax[0].twinx(); ax2.plot(hs, [S[h]["energy"] for h in hs], color="tab:red", alpha=0.5); ax2.set_ylabel("norm. Dirichlet energy", color="tab:red")
    rels = ["self", "same_set_same_assign", "same_set_diff_assign", "diff_set"]
    h0 = e1["h_energy"]
    ax[1].bar(range(4), [S[h0][r]["angle_cos"] for r in rels], color="tab:blue", alpha=0.7, label="mean cos(principal angles) of M row-spaces")
    ax[1].bar(np.arange(4) + 0.35, [S[h0][r]["cross_pc"] for r in rels], width=0.35, color="tab:green", alpha=0.7, label="cross R² (PCs)")
    ax[1].set_xticks(np.arange(4) + 0.17); ax[1].set_xticklabels(["self", "same words\nsame assign", "same words\ndiff assign", "different\nword set"], fontsize=8)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set(title=f"Is the position subspace word-independent? (hs {h0})", ylim=(-0.4, 1)); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/g1_subspace.png", dpi=150)
    lines.append(f"E1: h_energy={e1['h_energy']} h_xfer={e1['h_xfer']}; at h_energy: r2_in={S[h0]['r2_in']:.3f} "
                 f"r2_in_pc={S[h0]['r2_in_pc']:.3f} diff_set cross={S[h0]['diff_set']['cross']:.3f} "
                 f"cross_pc={S[h0]['diff_set']['cross_pc']:.3f} angcos={S[h0]['diff_set']['angle_cos']:.3f}")

# ---- g1b: plane stability + shared higher-dim subspace (from e1_analysis / e1_shared_subspace)
if os.path.exists(f"{RES}/e1_analysis.json") and os.path.exists(f"{RES}/e1_shared_subspace.json"):
    an = {int(k): v for k, v in json.load(open(f"{RES}/e1_analysis.json")).items()}
    sh = {int(k): v for k, v in json.load(open(f"{RES}/e1_shared_subspace.json")).items()}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    hs_a = sorted(an)
    for key, lab, col in (("same_set_same_assign", "same words, same assignment, different walk", "tab:blue"),
                          ("same_set_diff_assign", "same words, different assignment", "tab:orange"),
                          ("diff_set", "different word set", "tab:red")):
        ax[0].plot(hs_a, [an[h]["plane_stability"][key] for h in hs_a], "o-", color=col, label=lab)
    ax[0].axhline(np.sqrt(2 / 3584), color="gray", ls=":", label="random 2-planes (≈0.02)")
    ax[0].plot(hs_a, [an[h]["pc_in_own_embed_span"] for h in hs_a], "s--", color="tab:green", label="frac. of plane in span(own input embeddings)")
    ax[0].set(xlabel="hidden-state index", ylabel="mean cos(principal angles) / fraction", ylim=(0, 1),
              title="Top-2 PC position plane: stability across contexts")
    ax[0].legend(fontsize=7)
    hs_s = sorted(sh)
    for k, col in ((2, "#c6dbef"), (4, "#6baed6"), (8, "#2171b5"), (15, "#08306b")):
        ax[1].plot(hs_s, [sh[h]["by_k"][str(k)]["diff_set"] for h in hs_s], "o-", color=col, label=f"inside another word set's top-{k} PCs (rand {k/3584:.3f})")
    ax[1].plot(hs_s, [sh[h]["pooled_other_sets_top4"] for h in hs_s], "s-", color="tab:red", lw=2,
               label=f"inside pooled top-4 PCs of ALL other word sets (dim {sh[hs_s[0]]['pooled_dim']}, rand 0.015)")
    ax[1].set(xlabel="hidden-state index", ylabel="fraction of plane's norm² inside subspace", ylim=(0, 1),
              title="Is there a SHARED higher-dim position subspace across word sets?")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/g1b_stability_shared.png", dpi=150)

# ============================================================================ E2
if os.path.exists(f"{RES}/e2.json"):
    e2 = json.load(open(f"{RES}/e2.json"))
    D = e2["dense"]; hm = e2["meta"]["h_main"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    items = [("baseline (none)", D["none"], "k", "-")]
    for b, col in (("1.0", "#a6d96a"), ("2.0", "tab:green"), ("3.0", "#1a6b1a")):
        if f"native_gt2_b{b}" in D: items.append((f"native + GT2, β={b}", D[f"native_gt2_b{b}"], col, "-"))
    if "native_gt3_b2.0" in D: items.append(("native + GT3, β=2", D["native_gt3_b2.0"], "tab:olive", "--"))
    if "native_gt2_calB_b2.0" in D: items.append(("native (M from 2nd cal set) + GT2, β=2", D["native_gt2_calB_b2.0"], "tab:cyan", "--"))
    if "extracted_b2.0" in D: items.append(("extracted (model's own), β=2", D["extracted_b2.0"], "tab:blue", "-"))
    for i, key in enumerate(sorted(k for k in D if k.startswith("random_gt2_b2.0"))):
        items.append((f"random subspace + GT2, β=2 (plane {i+1})" if i == 0 else f"random plane {i+1}", D[key], "tab:purple", "-."))
    plot_family(ax[0], items, f"E2: ground-truth coords in native subspace (inject @ hs {hm})")
    items = [("baseline (none)", D["none"], "k", "-")]
    for key in sorted(k for k in D if k.startswith("native_gt2_b2.0_hs")):
        items.append((f"native+GT2 β=2 @ hs {key.split('hs')[-1]}", D[key], None, "-"))
    if "native_gt2_b2.0" in D: items.append((f"native+GT2 β=2 @ hs {hm}", D["native_gt2_b2.0"], "tab:green", "-"))
    cols = plt.cm.viridis(np.linspace(0.2, 0.9, len(items)))
    items = [(n, c, (col if col is not None else cols[i]), ls) for i, (n, c, col, ls) in enumerate(items)]
    plot_family(ax[1], items, "E2: injection layer sweep (M refit per layer)")
    fig.tight_layout(); fig.savefig(f"{FIG}/g2_uptake.png", dpi=150)
    for name, curves in D.items():
        lines.append(f"E2 thresholds {name}: {thresholds(curves)}")
    # override
    if e2.get("override"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        t, sm, se = smooth(D["none"]); ax.plot(t, sm, "k-", lw=2, label="baseline: true-map accuracy")
        if "none_injmap" in e2["override"]:
            t, sm, se = smooth(e2["override"]["none_injmap"]); ax.plot(t, sm, "k:", lw=1.5, label="baseline: injected-map accuracy (chance)")
        for b, col in (("b2.0", "tab:orange"), ("b3.0", "tab:red")):
            if b not in e2["override"]: continue
            o = e2["override"][b]
            t, sm, se = smooth(o["true_map"]); ax.plot(t, sm, "--", color=col, lw=2, label=f"override β={b[1:]}: true-map (tokens)")
            t, sm, se = smooth(o["injected_map"]); ax.plot(t, sm, "-", color=col, lw=2, label=f"override β={b[1:]}: injected-map (features)")
        ax.set(xscale="log", xlabel="context length (tokens)", ylabel="prob. mass on neighbors under map",
               title="E2 override: features encode a permuted grid, tokens follow the true grid", ylim=(0, 1))
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/g3_override.png", dpi=150)
    # unseen node
    if e2.get("unseen"):
        rows = e2["unseen"]
        conds = ["none", "ctx+query_trueX", "ctx+query_claimY", "query_only_trueX"]
        labels = ["no injection", "inject everywhere\n(X true coords)", "inject everywhere\n(X claims Y's coords)", "inject at query\ntoken only"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(conds)); w = 0.38
        for k, (met, col, lab) in enumerate((("P_nbrs_X", "tab:green", "P(true neighbors of X)"), ("P_nbrs_Y", "tab:red", "P(neighbors of Y)"))):
            vals = [[r["conds"][c][met] for r in rows] for c in conds]
            ax.bar(x + (k - 0.5) * w, [np.mean(v) for v in vals], w, color=col, alpha=0.75, label=lab)
            for i, v in enumerate(vals):
                ax.scatter(np.full(len(v), x[i] + (k - 0.5) * w) + np.random.default_rng(0).normal(0, 0.04, len(v)), v, s=8, color="k", alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set(ylabel="probability mass", title="E2 unseen-node test: X never appears in context; predict X's neighbors")
        ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/g4_unseen.png", dpi=150)
        for c in conds:
            lines.append(f"E2 unseen {c}: P_nbrs_X={np.mean([r['conds'][c]['P_nbrs_X'] for r in rows]):.3f} "
                         f"P_nbrs_Y={np.mean([r['conds'][c]['P_nbrs_Y'] for r in rows]):.3f} (n={len(rows)})")

# ============================================================================ E2b
if os.path.exists(f"{RES}/e2b.json"):
    eb = json.load(open(f"{RES}/e2b.json"))
    Db = eb["dense"]; hm = eb["meta"]["h_main"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    items = [("baseline (none)", Db["none"], "k", "-"), ("extracted (model's own, top-3 PCs), β=2", Db["extracted_b2.0"], "tab:blue", "-")]
    for b, col in (("1.0", "#a6d96a"), ("2.0", "tab:green"), ("3.0", "#1a6b1a")):
        if f"self_gt2_b{b}" in Db: items.append((f"GT2 in run's own plane (self-calibrated), β={b}", Db[f"self_gt2_b{b}"], col, "-"))
    if "self_gt3_b2.0" in Db: items.append(("GT3 in own plane, β=2", Db["self_gt3_b2.0"], "tab:olive", "--"))
    plot_family(ax[0], items, f"E2b: ground-truth coords in the run's OWN plane (inject @ hs {hm})")
    items = [("baseline (none)", Db["none"], "k", "-")]
    if "self_gt2_b2.0" in Db: items.append(("self-calibrated GT2, β=2", Db["self_gt2_b2.0"], "tab:green", "-"))
    for key, lab, col, ls in (("analytic_inE_gt2_b2.0", "analytic plane from input embeddings, β=2", "tab:orange", "--"),
                              ("analytic_inE_gt2_b3.0", "analytic (input emb), β=3", "#ff7f0e", ":"),
                              (f"analytic_hs{hm}E_gt2_b2.0", f"analytic plane from hs{hm} token reps, β=2", "tab:brown", "--"),
                              (f"analytic_hs{hm}E_gt2_b3.0", f"analytic (hs{hm} reps), β=3", "#8c564b", ":"),
                              ("shared_top2_gt2_b2.0", "shared-subspace top-2 dirs, β=2", "tab:red", "-"),
                              ("shared_top2_gt2_b3.0", "shared top-2, β=3", "#d62728", ":"),
                              ("shared_rand0_gt2_b2.0", "random plane INSIDE shared subspace, β=2", "tab:pink", "-"),
                              ("shared_rand1_gt2_b2.0", "random plane inside shared (2), β=2", "#f7b6d2", "-"),
                              ("shared_rand0_gt2_b3.0", "random-in-shared, β=3", "#e377c2", ":")):
        if key in Db: items.append((lab, Db[key], col, ls))
    plot_family(ax[1], items, "E2b: calibration-free constructions (analytic / shared-subspace planes)")
    fig.tight_layout(); fig.savefig(f"{FIG}/g2b_selfplane.png", dpi=150)
    for name, curves in Db.items():
        lines.append(f"E2b thresholds {name}: {thresholds(curves)}")
    if eb.get("diag"):
        lines.append("E2b diag: " + " ".join(f"{k}={np.mean([r[k] for r in eb['diag']]):.3f}" for k in eb["diag"][0] if k != "seed"))
    if eb.get("override"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        t, sm, se = smooth(Db["none"]); ax.plot(t, sm, "k-", lw=2, label="baseline: true-map accuracy")
        if os.path.exists(f"{RES}/e2.json"):
            oi = json.load(open(f"{RES}/e2.json"))["override"].get("none_injmap")
            if oi: t, sm, se = smooth(oi); ax.plot(t, sm, "k:", lw=1.5, label="baseline: injected-map accuracy (chance)")
        for b, col in (("self_b2.0", "tab:orange"), ("self_b3.0", "tab:red")):
            if b not in eb["override"]: continue
            o = eb["override"][b]
            t, sm, se = smooth(o["true_map"]); ax.plot(t, sm, "--", color=col, lw=2, label=f"override β={b[-3:]}: true-map (tokens)")
            t, sm, se = smooth(o["injected_map"]); ax.plot(t, sm, "-", color=col, lw=2, label=f"override β={b[-3:]}: injected-map (features)")
        ax.set(xscale="log", xlabel="context length (tokens)", ylabel="prob. mass on neighbors under map",
               title="E2b override (own plane): features encode a permuted grid, tokens the true grid", ylim=(0, 1))
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/g3b_override_self.png", dpi=150)
    if eb.get("unseen"):
        rows = eb["unseen"]; conds = ["none", "self_trueX", "self_claimY", "self_queryOnly"]
        labels = ["no injection", "own-plane GT everywhere\n(X true coords)", "own-plane GT everywhere\n(X claims Y's coords)", "inject at query\ntoken only"]
        fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(conds)); w = 0.38
        for k, (met, col, lab) in enumerate((("P_nbrs_X", "tab:green", "P(true neighbors of X)"), ("P_nbrs_Y", "tab:red", "P(neighbors of Y)"))):
            vals = [[r["conds"][c][met] for r in rows] for c in conds]
            ax.bar(x + (k - 0.5) * w, [np.mean(v) for v in vals], w, color=col, alpha=0.75, label=lab)
            for i, v in enumerate(vals):
                ax.scatter(np.full(len(v), x[i] + (k - 0.5) * w) + np.random.default_rng(0).normal(0, 0.04, len(v)), v, s=8, color="k", alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set(ylabel="probability mass", title="E2b unseen-node test (own-plane GT): X never appears in context")
        ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/g4b_unseen_self.png", dpi=150)
        for c in conds:
            lines.append(f"E2b unseen {c}: P_nbrs_X={np.mean([r['conds'][c]['P_nbrs_X'] for r in rows]):.3f} P_nbrs_Y={np.mean([r['conds'][c]['P_nbrs_Y'] for r in rows]):.3f} (n={len(rows)})")

# ============================================================================ E3
if os.path.exists(f"{RES}/e3.json"):
    e3 = json.load(open(f"{RES}/e3.json"))
    C = e3["curves"]
    q = lambda name: [ps[-1] for ps in C[name]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    greens = ["#c7e9c0", "#74c476", "#31a354", "#006d2c"]; grays = ["#cccccc", "#969696", "#636363", "#252525"]
    items = []
    for n in range(4):
        if f"none_demos{n}" in C: items.append((f"no injection, {n} demo grids", q(f"none_demos{n}"), grays[n], "--"))
    for n in range(4):
        if f"native_demos{n}" in C: items.append((f"native+GT2 β=2, {n} injected demo grids", q(f"native_demos{n}"), greens[n], "-"))
    plot_family(ax[0], items, "E3: query-grid uptake vs number of demo grids")
    items = []
    for name, lab, col, ls in (("none_demos3", "no injection, 3 demos", "k", "--"), ("native_demos3", "injected demos + injected query", "tab:green", "-"),
                               ("native_demos3_queryOnly", "uninjected demos + injected query", "tab:olive", "-."),
                               ("native_demos3_demoOnly", "injected demos + UNinjected query", "tab:orange", ":"),
                               ("random_demos0", "random subspace, 0 demos", "#b39ddb", "-."), ("random_demos3", "random subspace, 3 injected demos", "tab:purple", "-")):
        if name in C: items.append((lab, q(name), col, ls))
    plot_family(ax[1], items, "E3 controls (query grid)")
    fig.tight_layout(); fig.savefig(f"{FIG}/g5_demos.png", dpi=150)
    for name in C:
        lines.append(f"E3 query thresholds {name}: {thresholds(q(name))}")
    # per-segment uptake in the 3-demo conditions
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, (name, title) in enumerate((("native_demos3", "injected: uptake within each successive grid"), ("none_demos3", "no injection: uptake within each successive grid"))):
        if name not in C: continue
        cols = plt.cm.viridis(np.linspace(0.15, 0.9, 4))
        items = [(f"grid {k+1}" + (" (query)" if k == 3 else ""), [ps[k] for ps in C[name]], cols[k], "-") for k in range(4)]
        plot_family(ax[a], items, title, xmax=600)
    fig.tight_layout(); fig.savefig(f"{FIG}/g6_per_segment.png", dpi=150)
    if e3.get("unseen"):
        rows = e3["unseen"]
        conds = ["none", "native_trueX", "native_claimY", "random_trueX"]
        fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(conds)); w = 0.2
        for k, (nd, met, col, lab) in enumerate(((0, "P_nbrs_X", "#a1d99b", "0 demos: P(nbrs X)"), (3, "P_nbrs_X", "#006d2c", "3 demos: P(nbrs X)"),
                                                  (0, "P_nbrs_Y", "#fc9272", "0 demos: P(nbrs Y)"), (3, "P_nbrs_Y", "#a50f15", "3 demos: P(nbrs Y)"))):
            vals = [np.mean([r[met] for r in rows if r["cond"] == c and r["n_demos"] == nd]) for c in conds]
            ax.bar(x + (k - 1.5) * w, vals, w, color=col, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(["no injection", "native, X true coords", "native, X claims Y", "random subspace"], fontsize=8)
        ax.set(ylabel="probability mass", title="E3 unseen-node test on query grid, 0 vs 3 injected demo grids"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{FIG}/g7_unseen_demos.png", dpi=150)
        for nd in (0, 3):
            for c in conds:
                lines.append(f"E3 unseen demos={nd} {c}: P_nbrs_X={np.mean([r['P_nbrs_X'] for r in rows if r['cond']==c and r['n_demos']==nd]):.3f} "
                             f"P_nbrs_Y={np.mean([r['P_nbrs_Y'] for r in rows if r['cond']==c and r['n_demos']==nd]):.3f}")

open(f"{FIG}/summary.txt", "w").write("\n".join(lines))
print("\n".join(lines)); print("figures ->", FIG)
