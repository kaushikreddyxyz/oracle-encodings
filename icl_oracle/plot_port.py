"""Figures for port_inject.py (results_port/) -> figures_port/."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_port"
FIG = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/figures_port"
os.makedirs(FIG, exist_ok=True)
K = 25
LAYERS = ["16", "20", "26", "30"]


def smooth(curves, k=K):
    M = np.array(curves, dtype=float); m = M.mean(0); sem = M.std(0, ddof=1) / np.sqrt(len(M))
    return np.arange(k, len(m) + 1), np.convolve(m, np.ones(k) / k, "valid"), np.convolve(sem, np.ones(k) / k, "valid")


lines = []
# ============================================================================ P1
if os.path.exists(f"{RES}/p1.json"):
    p1 = json.load(open(f"{RES}/p1.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    keys = [("B_ctx_vs_A_ctx", "B after A: plane(B) vs plane(A), same context", "tab:red"),
            ("B_alone_vs_A_alone", "B alone vs A alone (separate contexts)", "gray"),
            ("B_ctx_vs_B_alone", "B after A vs B alone (does priming move B?)", "tab:blue"),
            ("A_ctx_vs_A_alone", "A in context vs A alone", "tab:cyan")]
    for key, lab, col in keys:
        vals = np.array([[r[l][key] for l in LAYERS] for r in p1["align"]])
        ax[0].errorbar([int(l) for l in LAYERS], vals.mean(0), vals.std(0, ddof=1) / np.sqrt(len(vals)), marker="o", color=col, label=lab, capsize=3)
    sw = np.array([[r[l]["B_ctx_vs_A2_ctx"] for l in LAYERS] for r in p1["same_words"]])
    ax[0].errorbar([int(l) for l in LAYERS], sw.mean(0), sw.std(0, ddof=1) / np.sqrt(len(sw)), marker="s", ls="--", color="tab:orange", label="B after A' (same words, different placement)", capsize=3)
    ax[0].axhline(0.02, color="k", ls=":", lw=0.8); ax[0].set(ylim=(0, 1), xlabel="hidden-state index", ylabel="cos(principal angles)", title="P1: does a preceding grid align the next grid's plane?")
    ax[0].legend(fontsize=6.5)
    A4 = np.array([r["26"] for r in p1["align4"]]).mean(0)
    im = ax[1].imshow(A4, vmin=0, vmax=1, cmap="viridis"); plt.colorbar(im, ax=ax[1])
    ax[1].set_xticks(range(4)); ax[1].set_yticks(range(4)); ax[1].set_xticklabels(["grid1", "grid2", "grid3", "query"]); ax[1].set_yticklabels(["grid1", "grid2", "grid3", "query"])
    for i in range(4):
        for j in range(4): ax[1].text(j, i, f"{A4[i, j]:.2f}", ha="center", va="center", color="w" if A4[i, j] < 0.6 else "k", fontsize=8)
    ax[1].set_title("P1: 4 grids in one context — pairwise plane alignment (hs 26)")
    for name, lab, col, ls in (("none", "B alone, no injection", "gray", "--"), ("none_ctx", "B after A, no injection", "k", "-"),
                               ("M_A_ctx", "GT via A's in-context plane", "tab:red", "-"), ("M_A_alone", "GT via A's separate-context plane", "tab:orange", "--"),
                               ("M_B_self", "GT via B's own calibrated plane (ceiling)", "tab:green", "-")):
        t, sm, se = smooth(p1["dense"][name]); ax[2].plot(t, sm, ls, color=col, lw=2, label=lab); ax[2].fill_between(t, sm - se, sm + se, color=col, alpha=0.1)
    ax[2].set(xscale="log", ylim=(0, 1), xlabel="tokens into grid B", ylabel="rule-following prob. mass", title="P1: inject GT into B through A's plane"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/p1_priming.png", dpi=150)
    for key, _, _ in keys:
        lines.append(f"P1 {key}: " + " ".join(f"hs{l}={np.mean([r[l][key] for r in p1['align']]):.3f}" for l in LAYERS))
    lines.append("P1 same_words B_ctx_vs_A2_ctx: " + " ".join(f"hs{l}={np.mean([r[l]['B_ctx_vs_A2_ctx'] for r in p1['same_words']]):.3f}" for l in LAYERS))
    for name in p1["dense"]:
        M = np.array(p1["dense"][name]); lines.append(f"P1 dense {name}: mass[100-200]={M[:, 100:200].mean():.3f} [200-400]={M[:, 200:400].mean():.3f} [400-600]={M[:, 400:].mean():.3f}")

# ============================================================================ P2
if os.path.exists(f"{RES}/p2.json"):
    p2 = json.load(open(f"{RES}/p2.json")); runs = p2["runs"]; hm = str(p2["meta"]["h_main"])
    planes = ["foreign", "rand_shared", "random"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    # (a) cos(late plane, P) by seeding condition
    conds = [("none", "no seed"), ("sb2.0|n200|late0|hs" + hm, "seed β=2, 200 tok"), ("sb2.0|n400|late0|hs" + hm, "seed β=2, 400 tok"),
             ("sb4.0|n200|late0|hs" + hm, "seed β=4, 200 tok"), ("sb4.0|n400|late0|hs" + hm, "seed β=4, 400 tok"), ("sb4.0|n200|late0|hs16", "seed β=4, 200 tok @ hs16")]
    x = np.arange(len(planes)); w = 0.13
    for i, (ck, lab) in enumerate(conds):
        vals = []
        for p in planes:
            if ck == "none": v = [r["cos_plane_vs_P"][p][hm] for r in runs if r["cond"] == "none"]
            else: v = [r["cos_plane_vs_P"][hm] for r in runs if r["cond"] == f"{p}|{ck}"]
            vals.append((np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))))
        ax[0].bar(x + (i - 2.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], label=lab, capsize=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["foreign-set plane", "random plane in shared subspace", "random plane"]); ax[0].set(ylim=(0, 1), ylabel="cos(model's late plane, P)  [tokens 1000-1200, uninjected]",
              title=f"P2 seeding: does the model's own plane adopt the seeded plane P? (hs {hm})"); ax[0].legend(fontsize=6.5)
    # (b) cos(late plane, no-seed plane): did seeding move the plane at all?
    for i, (ck, lab) in enumerate(conds[1:]):
        vals = []
        for p in planes:
            v = [r["cos_plane_vs_none"][hm] for r in runs if r["cond"] == f"{p}|{ck}"]; vals.append((np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))))
        ax[1].bar(x + (i - 2) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], label=lab, capsize=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["foreign-set plane", "random in shared", "random"]); ax[1].set(ylim=(0, 1), ylabel="cos(late plane with seed, late plane without seed)",
              title="P2: how much does seeding rotate the model's own late plane?"); ax[1].legend(fontsize=6.5)
    # (c) late-window uptake: none / late-inject only / seed+late-inject
    labels, means, errs, cols = [], [], [], []
    m0 = np.array([np.mean(r["mass"][600:]) for r in runs if r["cond"] == "none"]); labels.append("none"); means.append(m0.mean()); errs.append(m0.std(ddof=1) / np.sqrt(len(m0))); cols.append("k")
    for p, col in zip(planes, ("tab:green", "tab:pink", "tab:purple")):
        for ck, suffix in ((f"sb0.0|n0|late1|hs{hm}", "late inject only"), (f"sb2.0|n200|late1|hs{hm}", "seed β=2 + late inject")):
            m = np.array([np.mean(r["mass"][600:]) for r in runs if r["cond"] == f"{p}|{ck}"]); labels.append(f"{p}\n{suffix}"); means.append(m.mean()); errs.append(m.std(ddof=1) / np.sqrt(len(m))); cols.append(col)
    ax[2].bar(range(len(labels)), means, yerr=errs, color=cols, alpha=0.8, capsize=2); ax[2].set_xticks(range(len(labels))); ax[2].set_xticklabels(labels, fontsize=6.5, rotation=20)
    ax[2].set(ylabel="rule-following mass, tokens 600-1200", ylim=(0.5, 0.95), title="P2: is a late injection through P read better after seeding with P?")
    fig.tight_layout(); fig.savefig(f"{FIG}/p2_seeding.png", dpi=150)
    for ck, lab in conds:
        for p in planes:
            if ck == "none": v = [r["cos_plane_vs_P"][p][hm] for r in runs if r["cond"] == "none"]
            else: v = [r["cos_plane_vs_P"][hm] for r in runs if r["cond"] == f"{p}|{ck}"]
            lines.append(f"P2 cos(late plane,P) {p:12s} {lab:28s} = {np.mean(v):.3f} ± {np.std(v, ddof=1)/np.sqrt(len(v)):.3f}")
    for i, l in enumerate(labels): lines.append(f"P2 late mass {l.replace(chr(10), ' ')}: {means[i]:.3f} ± {errs[i]:.3f}")
    if p2.get("unseen"):
        rows = p2["unseen"]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        cats = [("-", "none"), ("foreign", "seed_only"), ("foreign", "Xfeat_only"), ("foreign", "seed+Xfeat"), ("random", "seed_only"), ("random", "Xfeat_only"), ("random", "seed+Xfeat")]
        vx = [np.mean([r["P_nbrs_X"] for r in rows if r["plane"] == p and r["cond"] == c]) for p, c in cats]
        vy = [np.mean([r["P_nbrs_Y"] for r in rows if r["plane"] == p and r["cond"] == c]) for p, c in cats]
        x = np.arange(len(cats)); ax.bar(x - 0.2, vx, 0.4, color="tab:green", label="P(true neighbors of X)"); ax.bar(x + 0.2, vy, 0.4, color="tab:red", label="P(neighbors of Y)")
        ax.set_xticks(x); ax.set_xticklabels([f"{p}\n{c}" for p, c in cats], fontsize=7); ax.set(ylabel="probability mass", title="P2: unseen-node test after seeding (seed β=4, 200 tok; X feature via P at query)"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIG}/p2_unseen.png", dpi=150)
        for (p, c), a, b in zip(cats, vx, vy): lines.append(f"P2 unseen {p}/{c}: P_nbrs_X={a:.3f} P_nbrs_Y={b:.3f}")

# ============================================================================ P4
if os.path.exists(f"{RES}/p4.json"):
    p4 = json.load(open(f"{RES}/p4.json"))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, lab, col, ls in (("none", "no injection", "k", "-"), ("self_all", "own-plane GT everywhere", "tab:green", "-"),
                               ("self_0_200", "own-plane GT for tokens 0-200, then removed", "#9ecae1", "-"), ("self_0_400", "0-400, then removed", "#4292c6", "-"), ("self_0_800", "0-800, then removed", "#08306b", "-")):
        t, sm, se = smooth(p4["dense"][name]); ax.plot(t, sm, ls, color=col, lw=2, label=lab); ax.fill_between(t, sm - se, sm + se, color=col, alpha=0.1)
    for n in (200, 400, 800): ax.axvline(n, color="gray", ls=":", lw=0.8)
    ax.set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="rule-following prob. mass", title="P4 withdrawal: crutch removed mid-context"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/p4_withdrawal.png", dpi=150)
    for name in p4["dense"]:
        M = np.array(p4["dense"][name]); lines.append(f"P4 {name}: [200-400]={M[:, 200:400].mean():.3f} [400-800]={M[:, 400:800].mean():.3f} [800-1200]={M[:, 800:].mean():.3f}")

# ============================================================================ P2b (early-layer seeding)
if os.path.exists(f"{RES}/p2b.json"):
    pb = json.load(open(f"{RES}/p2b.json")); runs = pb["runs"]; planes = ["foreign", "rand_shared", "random"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    x = np.arange(len(planes)); combos = [(h, b) for h in (4, 8, 16) for b in (2.0, 4.0, 8.0)]; w = 0.085
    none_c = {p: np.mean([r["cos_P"][p]["26"] for r in runs if r["cond"] == "none"]) for p in planes}
    ax[0].bar(x - 5 * w, [none_c[p] for p in planes], w, color="k", label="no seed")
    cm = plt.cm.viridis(np.linspace(0.1, 0.95, len(combos)))
    for i, (h, b) in enumerate(combos):
        v = [[r["cos_P"]["26"] for r in runs if r["cond"] == f"{p}|hs{h}|sb{b}|late0"] for p in planes]
        ax[0].bar(x + (i - 4) * w, [np.mean(a) for a in v], w, yerr=[np.std(a, ddof=1) / np.sqrt(len(a)) for a in v], color=cm[i], label=f"seed @hs{h}, β={b:g}", capsize=1.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["foreign-set plane", "random in shared subspace", "random plane"]); ax[0].set(ylim=(0, 1), ylabel="cos(model's late plane @hs26, P)", title=f"P2b: seed P (first {pb['meta']['n_seed']} tok, early layer): late plane vs P"); ax[0].legend(fontsize=6, ncol=2)
    for i, (h, b) in enumerate(combos):
        v = [[r["cos_none"]["26"] for r in runs if r["cond"] == f"{p}|hs{h}|sb{b}|late0"] for p in planes]
        ax[1].bar(x + (i - 4) * w, [np.mean(a) for a in v], w, yerr=[np.std(a, ddof=1) / np.sqrt(len(a)) for a in v], color=cm[i], label=f"seed @hs{h}, β={b:g}", capsize=1.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["foreign", "random in shared", "random"]); ax[1].set(ylim=(0, 1), ylabel="cos(late plane with seed, late plane without seed)", title="P2b: late plane, seeded vs unseeded"); ax[1].legend(fontsize=6, ncol=2)
    labels, means, errs, cols = ["none"], [], [], ["k"]
    m0 = np.array([np.mean(r["mass"][600:]) for r in runs if r["cond"] == "none"]); means.append(m0.mean()); errs.append(m0.std(ddof=1) / np.sqrt(len(m0)))
    for p, col in zip(planes, ("tab:green", "tab:pink", "tab:purple")):
        for ck, suf in ((f"{p}|lateonly", "late inject only"), (f"{p}|hs8|sb4.0|late1", "seed(hs8,β4)+late inject"), (f"{p}|hs8|sb4.0|late0", "seed(hs8,β4) only")):
            m = np.array([np.mean(r["mass"][600:]) for r in runs if r["cond"] == ck]); labels.append(f"{p}\n{suf}"); means.append(m.mean()); errs.append(m.std(ddof=1) / np.sqrt(len(m))); cols.append(col)
    ax[2].bar(range(len(labels)), means, yerr=errs, color=cols, alpha=0.8, capsize=2); ax[2].set_xticks(range(len(labels))); ax[2].set_xticklabels(labels, fontsize=6, rotation=25)
    ax[2].set(ylabel="rule-following mass, tokens 600-1200", ylim=(0.5, 0.95), title="P2b: late injection via P after early seeding")
    fig.tight_layout(); fig.savefig(f"{FIG}/p2b_early_seeding.png", dpi=150)
    for p in planes:
        lines.append(f"P2b cos(late@26,P) {p}: none={none_c[p]:.3f} " + " ".join(f"hs{h}/b{b:g}={np.mean([r['cos_P']['26'] for r in runs if r['cond']==f'{p}|hs{h}|sb{b}|late0']):.3f}" for h, b in combos))
        lines.append(f"P2b cos(late@26,none) {p}: " + " ".join(f"hs{h}/b{b:g}={np.mean([r['cos_none']['26'] for r in runs if r['cond']==f'{p}|hs{h}|sb{b}|late0']):.3f}" for h, b in combos))
    for i, l in enumerate(labels): lines.append(f"P2b late mass {l.replace(chr(10), ' ')}: {means[i]:.3f} ± {errs[i]:.3f}")
    if pb.get("unseen"):
        rows = pb["unseen"]; cats = [("-", "none")] + [(p, c) for p in ("foreign", "random") for c in ("seed_only", "Xfeat_only", "seed+Xfeat", "seed+Xfeat@hs8")]
        vx = [np.mean([r["P_nbrs_X"] for r in rows if r["plane"] == p and r["cond"] == c]) for p, c in cats]
        vy = [np.mean([r["P_nbrs_Y"] for r in rows if r["plane"] == p and r["cond"] == c]) for p, c in cats]
        fig, ax = plt.subplots(figsize=(10, 4.4)); x = np.arange(len(cats))
        ax.bar(x - 0.2, vx, 0.4, color="tab:green", label="P(true neighbors of X)"); ax.bar(x + 0.2, vy, 0.4, color="tab:red", label="P(neighbors of Y)")
        ax.set_xticks(x); ax.set_xticklabels([f"{p}\n{c}" for p, c in cats], fontsize=6.5); ax.set(ylabel="probability mass", title="P2b: unseen-node after early seeding (hs8, β=4, 300 tok)"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIG}/p2b_unseen.png", dpi=150)
        for (p, c), a, b in zip(cats, vx, vy): lines.append(f"P2b unseen {p}/{c}: P_nbrs_X={a:.3f} P_nbrs_Y={b:.3f}")

# ============================================================================ P3 (demos with early-layer injection)
if os.path.exists(f"{RES}/p3.json"):
    p3 = json.load(open(f"{RES}/p3.json")); C = p3["curves"]; q = lambda n: [ps[-1] for ps in C[n]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for a, pname in enumerate(("foreign", "random")):
        items = [("no injection, 3 demos", q("none_demos3"), "k", "--"), ("no injection, 0 demos", q("none_demos0"), "gray", ":")]
        cm = {8: ("#74c476", "#006d2c"), 16: ("#fd8d3c", "#a63603")}
        for h in (8, 16):
            for i, b in enumerate((2.0, 4.0)):
                for ck, ls in (("demos3_all", "-"), ("demos3_queryOnly", "-.")):
                    k = f"{pname}|hs{h}|b{b}|{ck}"
                    if k in C: items.append((f"@hs{h} β={b:g} {ck}", q(k), cm[h][i], ls))
        for name, curves, col, ls in items:
            t, sm, se = smooth(curves); ax[a].plot(t, sm, ls, color=col, lw=1.8, label=name)
        ax[a].set(xscale="log", ylim=(0, 1), xlabel="tokens into query grid", ylabel="rule-following prob. mass", title=f"P3: {pname} plane injected at an EARLY layer, 3 demos vs query-only"); ax[a].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/p3_early_demos.png", dpi=150)
    for k in C:
        M = np.array(q(k)); lines.append(f"P3 {k}: query mass[100-200]={M[:, 100:200].mean():.3f} [200-400]={M[:, 200:400].mean():.3f} [400-600]={M[:, 400:].mean():.3f}")

# ============================================================================ P5 (random-plane port sweep)
if os.path.exists(f"{RES}/p5.json"):
    p5 = json.load(open(f"{RES}/p5.json")); D = p5["dense"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.4))
    grid = [(h, b) for h in (12, 16, 20) for b in (4, 8, 16)]
    t, sm, se = smooth(D["none"]); ax[0].plot(t, sm, "k-", lw=2.5, label="no injection"); ax[0].fill_between(t, sm - se, sm + se, color="k", alpha=0.08)
    cm = {12: plt.cm.Blues, 16: plt.cm.Greens, 20: plt.cm.Oranges}
    for h, b in grid:
        k = f"random|hs{h}|b{b}"
        if k in D:
            t, sm, se = smooth(D[k]); ax[0].plot(t, sm, color=cm[h](0.4 + 0.2 * (4, 8, 16).index(b)), lw=1.6, label=f"random plane @hs{h}, β={b}")
    ax[0].set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="rule-following prob. mass", title="P5: GT coords in a RANDOM plane, no calibration — layer × strength"); ax[0].legend(fontsize=6, ncol=2)
    # bar summary at [200-600] for both random planes
    labels, means, errs, cols = ["none"], [np.array(D["none"])[:, 200:600].mean(1).mean()], [np.array(D["none"])[:, 200:600].mean(1).std(ddof=1) / np.sqrt(len(D["none"]))], ["k"]
    for pn, col in (("random", "tab:green"), ("random2", "tab:purple")):
        for h, b in grid:
            k = f"{pn}|hs{h}|b{b}"
            if k in D:
                m = np.array(D[k])[:, 200:600].mean(1); labels.append(f"{pn}\nhs{h} β{b}"); means.append(m.mean()); errs.append(m.std(ddof=1) / np.sqrt(len(m))); cols.append(col)
    ax[1].bar(range(len(labels)), means, yerr=errs, color=cols, alpha=0.8, capsize=1.5); ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels, fontsize=5.5, rotation=60)
    ax[1].axhline(means[0], color="k", ls=":", lw=0.8); ax[1].set(ylim=(0.5, 0.9), ylabel="mass, tokens 200-600", title="P5: two independent random planes")
    # override through the random plane
    O = p5["override"]
    t, sm, se = smooth(D["none"]); ax[2].plot(t, sm, "k-", lw=2, label="baseline: true-map")
    t, sm, se = smooth(O["none_injmap"]); ax[2].plot(t, sm, "k:", lw=1.5, label="baseline: injected-map (chance)")
    for (h, b), col in zip(((16, 4), (16, 8), (16, 16), (12, 8), (20, 8)), ("#74c476", "#31a354", "#006d2c", "tab:blue", "tab:orange")):
        k = f"random|hs{h}|b{b}"
        if k in O:
            t, sm, se = smooth([o["true_map"] for o in O[k]]); ax[2].plot(t, sm, "--", color=col, lw=1.6, label=f"override @hs{h} β={b}: true-map")
            t, sm, se = smooth([o["injected_map"] for o in O[k]]); ax[2].plot(t, sm, "-", color=col, lw=1.6, label=f"override @hs{h} β={b}: injected-map")
    ax[2].set(xscale="log", ylim=(0, 1), xlabel="context length (tokens)", ylabel="mass on neighbors under map", title="P5 override: permuted grid written into the random plane"); ax[2].legend(fontsize=5.5, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG}/p5_random_port.png", dpi=150)
    for k in D:
        M = np.array(D[k]); lines.append(f"P5 dense {k}: [100-200]={M[:, 100:200].mean():.3f} [200-600]={M[:, 200:600].mean():.3f} [600-1200]={M[:, 600:].mean():.3f}")
    for k in O:
        if k == "none_injmap": continue
        tm = np.array([o["true_map"] for o in O[k]]); im = np.array([o["injected_map"] for o in O[k]]); ch = np.array(O["none_injmap"])
        lines.append(f"P5 override {k}: true-map[200-600]={tm[:, 200:600].mean():.3f} injected-map={im[:, 200:600].mean():.3f} (chance {ch[:, 200:600].mean():.3f})")
    if p5.get("unseen"):
        rows = p5["unseen"]; conds = ["none"] + [f"{c}|hs{h}|b{b:g}" for h, b in ((16, 4.0), (16, 8.0), (12, 8.0), (20, 8.0)) for c in ("trueX", "claimY")]
        vx = [np.mean([r["P_nbrs_X"] for r in rows if r["cond"] == c]) for c in conds]; vy = [np.mean([r["P_nbrs_Y"] for r in rows if r["cond"] == c]) for c in conds]
        fig, ax = plt.subplots(figsize=(11, 4.4)); x = np.arange(len(conds))
        ax.bar(x - 0.2, vx, 0.4, color="tab:green", label="P(true neighbors of X)"); ax.bar(x + 0.2, vy, 0.4, color="tab:red", label="P(neighbors of Y)")
        ax.set_xticks(x); ax.set_xticklabels([c.replace("|", "\n") for c in conds], fontsize=6.5); ax.set(ylabel="probability mass", title="P5: unseen-node test through the random plane (no calibration)"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{FIG}/p5_unseen.png", dpi=150)
        for c, a, b in zip(conds, vx, vy): lines.append(f"P5 unseen {c}: P_nbrs_X={a:.3f} P_nbrs_Y={b:.3f}")

open(f"{FIG}/summary.txt", "w").write("\n".join(lines)); print("\n".join(lines)); print("figures ->", FIG)
