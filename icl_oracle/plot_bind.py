"""Analysis + figures for bind_icl.py results.  usage: python plot_bind.py [results_dir] [figures_dir]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_bind")
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures_bind")
os.makedirs(FIG, exist_ok=True)

BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
BIN_X = [np.sqrt(max(a, 0.5) * b) for a, b in BINS]
LATE = (64, 128)
COL = {"none": "0.5", "token": "k", "rand": "tab:gray", "actpc": "tab:blue", "native": "tab:green", "unembed": "tab:red"}
LAB = {"none": "no injection (floor)", "token": "z as a word (ceiling)", "rand": "random direction",
       "actpc": "random dir. in top-64 act. PCs", "native": "native marker direction", "unembed": "W_U[A]-W_U[B] (pure steering)"}


def load(name):
    d = json.load(open(os.path.join(RES, name)))
    for k, runs in d["runs"].items():
        for r in runs:
            for a in ("dlogit", "p2", "mass", "top_is_label", "s", "z", "y", "conv", "cat"):
                r[a] = np.asarray(r[a])
    return d


def acc_curve(runs, bins=BINS):
    """counterbalanced accuracy per bin: mean over prompts (both conventions) of p(correct label)."""
    A = np.array([np.where(r["y"] == 1, r["p2"], 1 - r["p2"]) for r in runs])       # [P, N]
    return np.array([A[:, a:b].mean(1) for a, b in bins]).T                             # [P, nbins]


def decompose(runs, span=LATE):
    """per-seed learned / steering components (logit units) from the sign x convention interaction."""
    by = {}
    for r in runs:
        a, b = span
        d, s = r["dlogit"][a:b], r["s"][a:b]
        if (s > 0).sum() == 0 or (s < 0).sum() == 0:
            continue
        D = d[s > 0].mean() - d[s < 0].mean()
        by.setdefault(int(r["seed"]), {})[int(r["c"])] = D
    L = np.array([(v[1] - v[-1]) / 2 for v in by.values() if 1 in v and -1 in v])
    S = np.array([(v[1] + v[-1]) / 2 for v in by.values() if 1 in v and -1 in v])
    return L, S


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def parse(name):
    arm, h, w, b = name.split("_"); return arm, int(h[1:]), w, float(b[1:])


# ---------------------------------------------------------------------- main grid
def main_figs():
    d = load("main.json"); runs = d["runs"]; meta = d["meta"]
    layers, betas, wheres = meta["layers"], meta["betas"], meta["wheres"]
    arms = ["rand", "actpc", "native", "unembed"]
    ref_none, ref_tok = acc_curve(runs["none"]), acc_curve(runs["token"])
    rows = []
    # per-condition summary
    summ = {}
    for name in runs:
        if name in ("none", "token"):
            continue
        arm, h, w, b = parse(name)
        A = acc_curve(runs[name]); L, S = decompose(runs[name])
        health = np.mean([r["top_is_label"][LATE[0]:LATE[1]].mean() for r in runs[name]])
        t = stats.ttest_1samp(L, 0) if len(L) > 2 else None
        summ[name] = dict(arm=arm, h=h, w=w, b=b, late=A[:, -1].mean(), late_se=se(A[:, -1]), learned=L.mean(), learned_se=se(L),
                          steer=S.mean(), steer_se=se(S), p=(t.pvalue if t else np.nan), health=health, curve=A.mean(0))
    for name in ("none", "token"):
        A = acc_curve(runs[name]); L, S = decompose(runs[name])
        summ[name] = dict(arm=name, late=A[:, -1].mean(), late_se=se(A[:, -1]), learned=L.mean(), learned_se=se(L), steer=S.mean(),
                          steer_se=se(S), health=np.mean([r["top_is_label"][64:].mean() for r in runs[name]]), curve=A.mean(0))

    # Fig 1: shot curves, one figure per position; panels layers x betas
    for w in wheres:
        fig, axes = plt.subplots(len(betas), len(layers), figsize=(3.2 * len(layers), 2.6 * len(betas)), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for i, b in enumerate(betas):
            for j, h in enumerate(layers):
                ax = axes[i, j]
                ax.plot(BIN_X, ref_none.mean(0), color=COL["none"], ls=":", label=LAB["none"])
                ax.plot(BIN_X, ref_tok.mean(0), color=COL["token"], ls="--", label=LAB["token"])
                for arm in arms:
                    n = f"{arm}_h{h}_{w}_b{b}"
                    if n in summ:
                        ax.plot(BIN_X, summ[n]["curve"], color=COL[arm], marker="o", ms=3, label=LAB[arm])
                ax.axhline(0.5, color="0.8", lw=0.8); ax.set_xscale("log"); ax.set_ylim(0.3, 1.02)
                if i == 0: ax.set_title(f"hidden state {h}")
                if j == 0: ax.set_ylabel(f"beta={b}\ncounterbalanced acc")
                if i == len(betas) - 1: ax.set_xlabel("shot index")
        axes[0, 0].legend(fontsize=6, loc="upper left")
        fig.suptitle(f"{meta['model']}: in-context binding of an injected bit, injection at the {w} token", y=1.0)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig1_curves_{w}.png"), dpi=140); plt.close(fig)

    # Fig 2: late accuracy heatmaps arm x (layer, beta) per position, health hatched
    fig, axes = plt.subplots(len(wheres), len(arms), figsize=(3.0 * len(arms), 2.6 * len(wheres)), squeeze=False)
    for i, w in enumerate(wheres):
        for j, arm in enumerate(arms):
            M = np.full((len(betas), len(layers)), np.nan); Hm = np.ones_like(M)
            for bi, b in enumerate(betas):
                for li, h in enumerate(layers):
                    n = f"{arm}_h{h}_{w}_b{b}"
                    if n in summ: M[bi, li] = summ[n]["late"]; Hm[bi, li] = summ[n]["health"]
            ax = axes[i, j]; im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis", origin="lower", aspect="auto")
            for bi in range(len(betas)):
                for li in range(len(layers)):
                    if not np.isnan(M[bi, li]):
                        ax.text(li, bi, f"{M[bi,li]:.2f}" + ("" if Hm[bi, li] > 0.9 else "*"), ha="center", va="center", fontsize=7,
                                color="w" if M[bi, li] < 0.8 else "k")
            ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers); ax.set_yticks(range(len(betas))); ax.set_yticklabels(betas)
            if i == 0: ax.set_title(LAB[arm], fontsize=8)
            if j == 0: ax.set_ylabel(f"inject at {w}\nbeta")
            if i == len(wheres) - 1: ax.set_xlabel("hidden state")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=f"counterbalanced acc, shots {LATE[0]}-{LATE[1]}")
    fig.suptitle(f"{meta['model']}: late-shot accuracy (floor {summ['none']['late']:.2f}, ceiling {summ['token']['late']:.2f}); * = model unhealthy", fontsize=9)
    fig.savefig(os.path.join(FIG, "fig2_late_heatmap.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # Fig 3: learned vs steering decomposition at each arm's best condition (by late acc, healthy only)
    best = {}
    for n, s in summ.items():
        if n in ("none", "token") or s["health"] < 0.9:
            continue
        if s["arm"] not in best or s["late"] > summ[best[s["arm"]]]["late"]:
            best[s["arm"]] = n
    names = ["none", "token"] + [best[a] for a in arms if a in best]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(names))
    for k, (comp, off, hatch) in enumerate((("learned", -0.2, ""), ("steer", 0.2, "//"))):
        ax.bar(x + off, [summ[n][comp] for n in names], 0.38, yerr=[summ[n][comp + "_se"] for n in names], capsize=3,
               color=[COL[summ[n]["arm"]] for n in names], hatch=hatch, alpha=0.9 if comp == "learned" else 0.5,
               label=("learned: flips with convention" if comp == "learned" else "steering: fixed sign"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([n if n in ("none", "token") else n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel(f"logit diff, shots {LATE[0]}-{LATE[1]}"); ax.legend(fontsize=8)
    ax.set_title("interaction decomposition at each arm's best healthy condition", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_decomposition.png"), dpi=140); plt.close(fig)

    # table
    lines = ["| condition | late acc | learned (logit) | steering (logit) | p(learned=0) | health |", "|---|---|---|---|---|---|"]
    for n in ["none", "token"] + sorted([n for n in summ if n not in ("none", "token")], key=lambda n: -summ[n]["late"]):
        s = summ[n]
        lines.append(f"| {n} | {s['late']:.3f} ± {s['late_se']:.3f} | {s['learned']:.2f} ± {s['learned_se']:.2f} | "
                     f"{s['steer']:.2f} ± {s['steer_se']:.2f} | {s.get('p', np.nan):.1e} | {s['health']:.2f} |")
    open(os.path.join(FIG, "main_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:22]))
    print("best per arm:", best)
    return summ, best


# ---------------------------------------------------------------------- signatures
def sig_figs():
    p = os.path.join(RES, "sig.json")
    if not os.path.exists(p):
        return
    d = load("sig.json"); runs = d["runs"]; conds = d["meta"]["conds"]
    tests = ["onset32", "flip64", "xor"]
    fine = [(i, i + 4) for i in range(0, 128, 4)]
    fx = [a + 2 for a, b in fine]
    fig, axes = plt.subplots(1, len(tests), figsize=(4.2 * len(tests), 3.2), sharey=True)
    for ax, t in zip(axes, tests):
        for ref, ls in (("none", ":"), ("token", "--")):
            n = f"{t}_{ref}"
            if n in runs:
                ax.plot(fx, acc_curve(runs[n], fine).mean(0), color=COL[ref], ls=ls, label=LAB[ref])
        for (arm, h, w, b) in conds:
            n = f"{t}_{arm}_h{h}_{w}_b{b}"
            if n in runs:
                ax.plot(fx, acc_curve(runs[n], fine).mean(0), color=COL[arm], marker="o", ms=2.5, label=f"{LAB[arm]} h{h} {w} b{b}")
        if t == "onset32": ax.axvline(32, color="k", lw=0.8); ax.set_title("injection starts at shot 32 (labels random before)")
        if t == "flip64": ax.axvline(64, color="k", lw=0.8); ax.set_title("convention flips at shot 64")
        if t == "xor": ax.set_title("label = z XOR (item is an animal)")
        ax.axhline(0.5, color="0.8", lw=0.8); ax.set_xlabel("shot index"); ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("p(correct label), current convention"); axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_signatures.png"), dpi=140); plt.close(fig)
    # numbers
    out = []
    for t in tests:
        for n in runs:
            if not n.startswith(t): continue
            A = acc_curve(runs[n], [(0, 32), (32, 48), (48, 64), (64, 80), (80, 96), (96, 128)]).mean(0)
            out.append(f"{n}: " + " ".join(f"{a:.2f}" for a in A))
    open(os.path.join(FIG, "sig_table.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    main_figs()
    sig_figs()
