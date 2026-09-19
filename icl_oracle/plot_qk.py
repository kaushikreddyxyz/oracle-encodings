"""Figures for qk_port.py (results_qk/) -> figures_qk/."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_qk"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_qk"
os.makedirs(FIG, exist_ok=True)
K = 25
NH = 16


def smooth(curves, k=K):
    M = np.array(curves, dtype=float); m = M.mean(0); sem = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros_like(m)
    return np.arange(k, len(m) + 1), np.convolve(m, np.ones(k) / k, "valid"), np.convolve(sem, np.ones(k) / k, "valid")


lines = []
# ============================================================================ screen
if os.path.exists(f"{RES}/screen.json"):
    sc = json.load(open(f"{RES}/screen.json")); CATS = sc["meta"]["cats"]; ci = {c: i for i, c in enumerate(CATS)}
    E = {c: np.mean(sc["conds"][c]["enr"], 0) for c in sc["conds"]}
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    panels = [(E["none"][:, :, ci["nbr"]], "neighbor-occurrence enrichment (baseline)", "viridis", None),
              (E["none"][:, :, ci["succ_same"]], "successor-of-same-node enrichment = induction (baseline)", "viridis", None),
              (E["own_hs26_b2"][:, :, ci["nbr"]] - E["none"][:, :, ci["nbr"]], "Δ neighbor enrichment: own-plane injection @hs26", "RdBu_r", "sym"),
              (E["rand_hs16_b4"][:, :, ci["nbr"]] - E["none"][:, :, ci["nbr"]], "Δ neighbor enrichment: random plane @hs16 β=4", "RdBu_r", "sym")]
    for a, (M, title, cmap, sym) in zip(ax, panels):
        v = np.abs(M).max() if sym else None
        im = a.imshow(M.T, aspect="auto", cmap=cmap, vmin=-v if sym else None, vmax=v if sym else None, origin="lower")
        a.set(xlabel="layer", ylabel="head", title=title); plt.colorbar(im, ax=a, fraction=0.03)
    fig.tight_layout(); fig.savefig(f"{FIG}/q1_screen.png", dpi=150)
    for key, M in (("nbr", E["none"][:, :, ci["nbr"]]), ("same", E["none"][:, :, ci["same"]]), ("succ_same", E["none"][:, :, ci["succ_same"]]),
                   ("d_nbr_own", panels[2][0]), ("d_nbr_rand16", panels[3][0])):
        idx = np.argsort(M.ravel())[::-1][:10]; lines.append(f"screen top {key}: " + ", ".join(f"L{i // NH}H{i % NH}={M.ravel()[i]:.2f}" for i in idx))
    lines.append("screen acc: " + json.dumps(sc["summary"]["acc"]))

# ============================================================================ ablate
if os.path.exists(f"{RES}/ablate.json"):
    ab = json.load(open(f"{RES}/ablate.json")); runs = ab["runs"]; groups = ["no_ablation"] + list(ab["groups"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(groups)); w = 0.27
    for i, (cname, lab, col) in enumerate((("none", "no injection", "k"), ("own_hs26_b2", "own-plane injection", "tab:green"), ("rand_hs16_b4", "random plane @hs16 β=4", "tab:purple"))):
        v = [[r["acc"] for r in runs if r["group"] == g and r["cond"] == cname] for g in groups]
        ax[0].bar(x + (i - 1) * w, [np.mean(a) for a in v], w, yerr=[np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0 for a in v], color=col, alpha=0.8, label=lab, capsize=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, fontsize=7, rotation=15); ax[0].set(ylabel="rule-following mass, tokens 400-1200", ylim=(0.3, 1), title="Q2: ablate head groups (8 heads each)"); ax[0].legend(fontsize=7)
    # injection gain with vs without ablation
    gain_own = [np.mean([r["acc"] for r in runs if r["group"] == g and r["cond"] == "own_hs26_b2"]) - np.mean([r["acc"] for r in runs if r["group"] == g and r["cond"] == "none"]) for g in groups]
    gain_rand = [np.mean([r["acc"] for r in runs if r["group"] == g and r["cond"] == "rand_hs16_b4"]) - np.mean([r["acc"] for r in runs if r["group"] == g and r["cond"] == "none"]) for g in groups]
    ax[1].bar(x - 0.2, gain_own, 0.4, color="tab:green", label="own-plane injection gain"); ax[1].bar(x + 0.2, gain_rand, 0.4, color="tab:purple", label="random-plane gain")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels(groups, fontsize=7, rotation=15); ax[1].set(ylabel="Δ mass (injected − none)", title="Q2: does ablating the readers remove the injection gain?"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/q2_ablate.png", dpi=150)
    for g in groups:
        lines.append(f"ablate {g}: none={np.mean([r['acc'] for r in runs if r['group']==g and r['cond']=='none']):.3f} own={np.mean([r['acc'] for r in runs if r['group']==g and r['cond']=='own_hs26_b2']):.3f} rand={np.mean([r['acc'] for r in runs if r['group']==g and r['cond']=='rand_hs16_b4']):.3f}")
    singles = sorted({r["group"] for r in runs if r["group"].startswith("single_")})
    for g in singles:
        lines.append(f"ablate {g}: none={np.mean([r['acc'] for r in runs if r['group']==g and r['cond']=='none']):.3f} own={np.mean([r['acc'] for r in runs if r['group']==g and r['cond']=='own_hs26_b2']):.3f}")

# ============================================================================ port
if os.path.exists(f"{RES}/port.json"):
    pr = json.load(open(f"{RES}/port.json")); D = pr["dense"]; meta = pr["meta"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    items = [("no injection", "none", "k", "-"), ("own-plane GT @hs26 β=2", "own_hs26_b2", "tab:blue", "-"), ("random plane @hs16 β=4", "rand_hs16_b4", "tab:purple", "-."), ("random plane @hs20 β=4", "rand_hs20_b4", "#b39ddb", "-.")]
    for b, col in ((1, "#c7e9c0"), (2, "#74c476"), (4, "#31a354"), (8, "#006d2c")):
        items.append((f"QK-port @hs{meta['hs_inj']} β={b}", f"port_b{b}", col, "-"))
    for name, key, col, ls in items:
        if key in D: t, sm, se = smooth(D[key]); ax[0].plot(t, sm, ls, color=col, lw=1.8, label=name); ax[0].fill_between(t, sm - se, sm + se, color=col, alpha=0.08)
    ax[0].set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="rule-following prob. mass", title=f"Q3: QK-port (readers {', '.join(f'L{l}H{h}' for l, h in meta['readers'][:4])}…)"); ax[0].legend(fontsize=6.5)
    items = [("no injection", "none", "k", "-"), ("QK-port β=2", "port_b2", "#74c476", "-"), ("QK-port β=4", "port_b4", "#31a354", "-"),
             ("QK-port @hs16 β=2", "port_b2_hs16", "tab:olive", "--"), ("QK-port @hs16 β=4", "port_b4_hs16", "#808000", "--"),
             ("anti-port β=2", "anti_b2", "tab:red", ":"), ("anti-port β=4", "anti_b4", "#a50f15", ":"), ("non-reader port β=2", "nonreader_b2", "tab:orange", "-."), ("non-reader port β=4", "nonreader_b4", "#e6550d", "-.")]
    for name, key, col, ls in items:
        if key in D: t, sm, se = smooth(D[key]); ax[1].plot(t, sm, ls, color=col, lw=1.8, label=name)
    ax[1].set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="rule-following prob. mass", title="Q3 controls: anti-port, non-reader port, layer"); ax[1].legend(fontsize=6.5)
    # override through the port
    O = pr["override"]
    t, sm, se = smooth(D["none"]); ax[2].plot(t, sm, "k-", lw=2, label="baseline: true-map")
    t, sm, se = smooth(O["none_injmap"]); ax[2].plot(t, sm, "k:", lw=1.5, label="baseline: injected-map (chance)")
    for b, col in ((2, "#74c476"), (4, "#006d2c")):
        k = f"port_b{b}"
        if k in O:
            t, sm, se = smooth([o["true_map"] for o in O[k]]); ax[2].plot(t, sm, "--", color=col, lw=1.8, label=f"port override β={b}: true-map")
            t, sm, se = smooth([o["injected_map"] for o in O[k]]); ax[2].plot(t, sm, "-", color=col, lw=1.8, label=f"port override β={b}: injected-map")
    ax[2].set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="mass on neighbors under map", title="Q3 override: permuted grid through the QK-port"); ax[2].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/q3_port.png", dpi=150)
    for k in D:
        M = np.array(D[k]); lines.append(f"port dense {k}: [100-200]={M[:, 100:200].mean():.3f} [200-600]={M[:, 200:600].mean():.3f} [600-1200]={M[:, 600:].mean():.3f}")
    for k in O:
        if k == "none_injmap": continue
        tm = np.array([o["true_map"] for o in O[k]]); im = np.array([o["injected_map"] for o in O[k]]); ch = np.array(O["none_injmap"])
        lines.append(f"port override {k}: true-map[200-600]={tm[:, 200:600].mean():.3f} injected-map={im[:, 200:600].mean():.3f} (chance {ch[:, 200:600].mean():.3f})")
    # reader attention under each condition
    A = pr["attn"]; readers = [f"L{l}H{h}" for l, h in meta["readers"]]
    fig, ax = plt.subplots(figsize=(12, 4.6)); conds = [c for c in ("none", "own_hs26_b2", "rand_hs16_b4", "port_b2", "port_b4", "port_b8", "anti_b4", "nonreader_b4") if c in A]
    x = np.arange(len(readers)); w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        v = [np.mean([a[r]["nbr"] for a in A[c]]) for r in readers]
        ax.bar(x + (i - len(conds) / 2 + 0.5) * w, v, w, label=c)
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_xticks(x); ax.set_xticklabels(readers, fontsize=8); ax.set(ylabel="neighbor-occurrence attention enrichment", title="Q3 mechanism: do the reader heads attend more to neighbors under the port?"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG}/q3_reader_attention.png", dpi=150)
    for c in conds:
        lines.append(f"port attn {c}: " + " ".join(f"{r}={np.mean([a[r]['nbr'] for a in A[c]]):.2f}" for r in readers))
    if pr.get("unseen"):
        rows = pr["unseen"]; conds = ["none"] + [f"{c}|{t}" for c in ("own", "rand16", "port_b2", "port_b4", "port_b8") for t in ("trueX", "claimY")]
        vx = [np.mean([r["P_nbrs_X"] for r in rows if r["cond"] == c]) for c in conds]; vy = [np.mean([r["P_nbrs_Y"] for r in rows if r["cond"] == c]) for c in conds]
        fig, ax = plt.subplots(figsize=(12, 4.6)); x = np.arange(len(conds))
        ax.bar(x - 0.2, vx, 0.4, color="tab:green", label="P(true neighbors of X)"); ax.bar(x + 0.2, vy, 0.4, color="tab:red", label="P(neighbors of Y)")
        ax.set_xticks(x); ax.set_xticklabels([c.replace("|", "\n") for c in conds], fontsize=7); ax.set(ylabel="probability mass", title="Q3 unseen-node: X never in context; coordinate via own plane / random plane / QK-port"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIG}/q3_unseen.png", dpi=150)
        for c, a, b in zip(conds, vx, vy): lines.append(f"port unseen {c}: P_nbrs_X={a:.3f} P_nbrs_Y={b:.3f}")
    lines.append("port kernel: " + json.dumps(meta.get("kernel", {})))

# ============================================================================ universal
if os.path.exists(f"{RES}/universal.json"):
    un = json.load(open(f"{RES}/universal.json")); D = un["dense"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for a, set_id in zip(ax, (1, 2, 3)):
        for key, lab, col in ((f"set{set_id}|none", "no injection", "k"), (f"set{set_id}|rand_hs16_b4", "random plane @hs16 β=4", "tab:purple")):
            if key in D: t, sm, se = smooth(D[key]); a.plot(t, sm, color=col, lw=1.8, label=lab)
        pk = [k for k in D if k.startswith(f"set{set_id}|port")]
        for k in pk: t, sm, se = smooth(D[k]); a.plot(t, sm, color="tab:green", lw=2, label="QK-port (no recalibration)")
        a.set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="rule-following prob. mass", title=f"Q5 universality: word set {set_id}, new assignment"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/q5_universal.png", dpi=150)
    for k in D:
        M = np.array(D[k]); lines.append(f"universal {k}: [200-600]={M[:, 200:600].mean():.3f}")
    rows = un["unseen"]
    for c in ("none", "port|trueX", "port|claimY"):
        lines.append(f"universal unseen {c}: P_nbrs_X={np.mean([r['P_nbrs_X'] for r in rows if r['cond']==c]):.3f} P_nbrs_Y={np.mean([r['P_nbrs_Y'] for r in rows if r['cond']==c]):.3f} (n={len([r for r in rows if r['cond']==c])})")

# ============================================================================ port2 (round two)
if os.path.exists(f"{RES}/port2.json"):
    p2 = json.load(open(f"{RES}/port2.json")); D = p2["dense"]; none = np.array(D["none"])
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    fam = [("rand20", (3, 4, 5), plt.cm.Purples, "random plane @hs20"), ("z2", (3, 4, 5), plt.cm.Greens, "QK-port (spectral kernel) @hs16"),
           ("adj", (2, 3, 4, 6, 8), plt.cm.Oranges, "QK-port (ADJACENCY kernel) @hs16"), ("late", (2, 3, 4, 6), plt.cm.Blues, f"late-reader port @hs{p2['meta']['hs_late']}"), ("lateadj", (2, 4, 6), plt.cm.Reds, "late-reader adjacency port")]
    labels, means, errs, cols = ["none"], [none[:, 200:600].mean(1).mean()], [none[:, 200:600].mean(1).std(ddof=1) / np.sqrt(len(none))], ["k"]
    for key, betas, cm, lab in fam:
        for i, b in enumerate(betas):
            k = f"{key}_b{b}"
            if k in D:
                m = np.array(D[k])[:, 200:600].mean(1); labels.append(f"{key}\nβ{b}"); means.append(m.mean()); errs.append(m.std(ddof=1) / np.sqrt(len(m))); cols.append(cm(0.35 + 0.6 * i / max(1, len(betas) - 1)))
    ax[0].bar(range(len(labels)), means, yerr=errs, color=cols, capsize=1.5); ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels, fontsize=6, rotation=60)
    ax[0].axhline(means[0], color="k", ls=":", lw=0.8); ax[0].set(ylim=(0.5, 0.9), ylabel="mass, tokens 200-600", title="Q4: port variants × strength")
    # reader ablation specificity
    keys = [("none", "abl|none", "no injection"), ("z2_b4", "abl|z2_b4", "QK-port β=4"), ("adj_b4", "abl|adj_b4", "adjacency port β=4"), ("late_b4", "abl|late_b4", "late-reader port β=4"), ("rand20_b4", "abl|rand20_b4", "random plane β=4")]
    x = np.arange(len(keys))
    for i, (kk, ka, lab) in enumerate(keys):
        for j, (k, col, hatch) in enumerate(((kk, "tab:blue", ""), (ka, "tab:red", "//"))):
            m = np.array(D[k])[:, 200:600].mean(1); ax[1].bar(i + (j - 0.5) * 0.38, m.mean(), 0.38, yerr=m.std(ddof=1) / np.sqrt(len(m)), color=col, hatch=hatch, alpha=0.8, capsize=2, label=("intact" if j == 0 else "8 readers ablated") if i == 0 else None)
    ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, _, l in keys], fontsize=7, rotation=15); ax[1].set(ylim=(0.5, 0.9), ylabel="mass, tokens 200-600", title="Q4: specificity — ablating the 8 reader heads"); ax[1].legend(fontsize=7)
    O = p2["override"]; ch = np.array(O["none_injmap"])
    t, sm, se = smooth(D["none"]); ax[2].plot(t, sm, "k-", lw=2, label="baseline: true-map"); t, sm, se = smooth(O["none_injmap"]); ax[2].plot(t, sm, "k:", lw=1.5, label="baseline: injected-map (chance)")
    for k, col in (("z2_b4", "tab:green"), ("adj_b4", "tab:orange"), ("late_b4", "tab:blue")):
        if k in O:
            t, sm, se = smooth([o["true_map"] for o in O[k]]); ax[2].plot(t, sm, "--", color=col, lw=1.6, label=f"{k} override: true-map")
            t, sm, se = smooth([o["injected_map"] for o in O[k]]); ax[2].plot(t, sm, "-", color=col, lw=1.6, label=f"{k} override: injected-map")
    ax[2].set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="mass on neighbors under map", title="Q4 override through each port"); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/q4_port2.png", dpi=150)
    for k in D:
        M = np.array(D[k]); lines.append(f"port2 dense {k}: [100-200]={M[:, 100:200].mean():.3f} [200-600]={M[:, 200:600].mean():.3f} [600-1200]={M[:, 600:].mean():.3f}")
    for k in O:
        if k == "none_injmap": continue
        tm = np.array([o["true_map"] for o in O[k]]); im = np.array([o["injected_map"] for o in O[k]])
        lines.append(f"port2 override {k}: true-map[200-600]={tm[:, 200:600].mean():.3f} injected-map={im[:, 200:600].mean():.3f} (chance {ch[:, 200:600].mean():.3f})")
    A = p2["attn"]; readers = [f"L{l}H{h}" for l, h in p2["meta"]["readers"]]
    fig, ax = plt.subplots(figsize=(13, 4.6)); conds = [c for c in ("none", "z2_b4", "adj_b4", "adj_b6", "late_b4", "lateadj_b4", "rand20_b4") if c in A]; x = np.arange(len(readers)); w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        ax.bar(x + (i - len(conds) / 2 + 0.5) * w, [np.mean([a[r] for a in A[c]]) for r in readers], w, label=c)
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_xticks(x); ax.set_xticklabels(readers, fontsize=8); ax.set(ylabel="neighbor-occurrence attention enrichment", title="Q4 mechanism: reader-head neighbor attention under each port"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG}/q4_reader_attention.png", dpi=150)
    for c in conds: lines.append(f"port2 attn {c}: " + " ".join(f"{r}={np.mean([a[r] for a in A[c]]):.2f}" for r in readers))
    if p2.get("unseen"):
        rows = p2["unseen"]; conds = ["none"] + [f"{c}|{t}" for c in ("rand20_b4", "z2_b4", "adj_b3", "adj_b4", "adj_b6", "late_b3", "late_b4", "lateadj_b4") for t in ("trueX", "claimY")]
        vx = [np.mean([r["P_nbrs_X"] for r in rows if r["cond"] == c]) for c in conds]; vy = [np.mean([r["P_nbrs_Y"] for r in rows if r["cond"] == c]) for c in conds]
        fig, ax = plt.subplots(figsize=(15, 4.6)); x = np.arange(len(conds))
        ax.bar(x - 0.2, vx, 0.4, color="tab:green", label="P(true neighbors of X)"); ax.bar(x + 0.2, vy, 0.4, color="tab:red", label="P(neighbors of Y)")
        ax.set_xticks(x); ax.set_xticklabels([c.replace("|", "\n") for c in conds], fontsize=6.5); ax.set(ylabel="probability mass", title="Q4 unseen-node through each port (no calibration)"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIG}/q4_unseen.png", dpi=150)
        for c, a, b in zip(conds, vx, vy): lines.append(f"port2 unseen {c}: P_nbrs_X={a:.3f} P_nbrs_Y={b:.3f}")

# ============================================================================ port3 (late port: universality + controls at hs26)
if os.path.exists(f"{RES}/port3.json"):
    p3 = json.load(open(f"{RES}/port3.json")); D = p3["dense"]; O = p3["override"]; rows = p3["unseen"]; hsl = p3["meta"]["hs_late"]
    conds = ["own_b2", "rand26_b4", "late_b3", "late_b4", "lateadj_b4", "lateanti_b4", "latenon_b4"]
    labs = {"own_b2": "own plane (calibrated) @hs26 β=2", "rand26_b4": f"random plane @hs{hsl} β=4", "late_b3": "late-reader port β=3", "late_b4": "late-reader port β=4",
            "lateadj_b4": "late-reader ADJACENCY port β=4", "lateanti_b4": "late ANTI-port β=4 (control)", "latenon_b4": "non-reader port @hs26 β=4 (control)"}
    cols = {"own_b2": "tab:blue", "rand26_b4": "tab:purple", "late_b3": "#74c476", "late_b4": "#006d2c", "lateadj_b4": "tab:orange", "lateanti_b4": "tab:red", "latenon_b4": "tab:gray"}
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    # (a) uptake, main set vs new sets: Δ vs none at [100-300]
    groups = [("set4", "word set 4 (6 seeds)"), ("new", "new word sets 1-3 (9 contexts)")]
    def curves(prefix, c):
        src = O if c == "none_injmap" else D
        return [x for k, v in src.items() if k.endswith("|" + c) and (k.startswith(prefix) if prefix != "new" else not k.startswith("set4")) for x in v]
    x = np.arange(len(conds)); w = 0.38
    for gi, (pre, glab) in enumerate(groups):
        n0 = np.array(curves(pre, "none")); vals = []
        for c in conds:
            a = np.array(curves(pre, c)); d = a[:, 100:300].mean(1) - n0[:, 100:300].mean(1); vals.append((d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
        ax[0].bar(x + (gi - 0.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], color=[cols[c] for c in conds], alpha=0.9 if gi == 0 else 0.45, hatch="" if gi == 0 else "//", capsize=2, label=glab)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xticks(x); ax[0].set_xticklabels([labs[c] for c in conds], fontsize=6, rotation=25, ha="right"); ax[0].set(ylabel="Δ uptake vs none, tokens 100-300", title="Q6: early uptake — main set vs NEW word sets"); ax[0].legend(fontsize=7)
    # (b) override injected-map - chance, by group
    for gi, (pre, glab) in enumerate(groups):
        ch = np.array(curves(pre, "none_injmap")); vals = []
        for c in conds:
            os_ = [o for k, v in O.items() if k.endswith("|" + c) and (k.startswith(pre) if pre != "new" else not k.startswith("set4")) for o in v]
            if not os_: vals.append((0.0, 0.0)); continue
            im = np.array([o["injected_map"] for o in os_]); d = im[:, 200:600].mean(1) - ch[:, 200:600].mean(1); vals.append((d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
        ax[1].bar(x + (gi - 0.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], color=[cols[c] for c in conds], alpha=0.9 if gi == 0 else 0.45, hatch="" if gi == 0 else "//", capsize=2, label=glab)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels([labs[c] for c in conds], fontsize=6, rotation=25, ha="right"); ax[1].set(ylabel="injected-map mass − chance, tokens 200-600", title="Q6: override — does the model follow a wrong map written into the channel?"); ax[1].legend(fontsize=7)
    # (c) unseen: ΔP(nbrs X) true and ΔP(nbrs Y) claim, by group
    for gi, (pre, glab) in enumerate(groups):
        rs = [r for r in rows if (r["set"] == 4) == (pre == "set4")]; nn = {(r["set"], r["seed"], r["X"]): r for r in rs if r["cond"] == "none"}
        vx = [np.mean([r["P_nbrs_X"] - nn[(r["set"], r["seed"], r["X"])]["P_nbrs_X"] for r in rs if r["cond"] == f"{c}|trueX"]) for c in conds]
        vy = [np.mean([r["P_nbrs_Y"] - nn[(r["set"], r["seed"], r["X"])]["P_nbrs_Y"] for r in rs if r["cond"] == f"{c}|claimY"]) for c in conds]
        ax[2].bar(x + (gi - 0.5) * w - 0.09, vx, 0.18, color="tab:green", alpha=0.9 if gi == 0 else 0.45, hatch="" if gi == 0 else "//", label=f"ΔP(true nbrs of X) — {glab}")
        ax[2].bar(x + (gi - 0.5) * w + 0.09, vy, 0.18, color="tab:red", alpha=0.9 if gi == 0 else 0.45, hatch="" if gi == 0 else "//", label=f"ΔP(nbrs of Y | X claims Y) — {glab}")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xticks(x); ax[2].set_xticklabels([labs[c] for c in conds], fontsize=6, rotation=25, ha="right"); ax[2].set(ylabel="Δ probability mass vs none", title="Q6: unseen-node control — main set vs NEW word sets"); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/q6_late_port_universal.png", dpi=150)
    for pre, glab in groups:
        n0 = np.array(curves(pre, "none")); ch = np.array(curves(pre, "none_injmap"))
        rs = [r for r in rows if (r["set"] == 4) == (pre == "set4")]; nn = {(r["set"], r["seed"], r["X"]): r for r in rs if r["cond"] == "none"}
        for c in conds:
            a = np.array(curves(pre, c)); d = a[:, 100:300].mean(1) - n0[:, 100:300].mean(1)
            os_ = [o for k, v in O.items() if k.endswith("|" + c) and (k.startswith(pre) if pre != "new" else not k.startswith("set4")) for o in v]
            im = np.array([o["injected_map"] for o in os_]) if os_ else np.zeros((1, 1200)); tm = np.array([o["true_map"] for o in os_]) if os_ else np.zeros((1, 1200))
            dx = [r["P_nbrs_X"] - nn[(r["set"], r["seed"], r["X"])]["P_nbrs_X"] for r in rs if r["cond"] == f"{c}|trueX"]; dy = [r["P_nbrs_Y"] - nn[(r["set"], r["seed"], r["X"])]["P_nbrs_Y"] for r in rs if r["cond"] == f"{c}|claimY"]
            lines.append(f"port3 {glab} {c}: Δuptake[100-300]={d.mean():+.3f}±{d.std(ddof=1)/np.sqrt(len(d)):.3f} | override inj-chance={(im[:,200:600].mean(1)-ch[:,200:600].mean(1)).mean():+.3f} true-base={(tm[:,200:600].mean(1)-n0[:,200:600].mean(1)).mean():+.3f} | unseen ΔX={np.mean(dx):+.3f}±{np.std(dx,ddof=1)/np.sqrt(len(dx)):.3f} ΔY={np.mean(dy):+.3f}±{np.std(dy,ddof=1)/np.sqrt(len(dy)):.3f} (n={len(dx)})")

open(f"{FIG}/summary.txt", "w").write("\n".join(lines)); print("\n".join(lines)); print("figures ->", FIG)
