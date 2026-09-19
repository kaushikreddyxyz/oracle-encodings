"""Figures + summary table from results/*.json.  python markov_graft/analyze.py [--results DIR]"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE_COLOR = {"xs": "#eda100", "s": "#2a78d6", "m": "#eb6834", "l": "#1baf7a"}
SIZE_ORDER = ["xs", "s", "m", "l"]
ARM_LABEL = {"oracle_1": "oracle graft, 1 bit (top bit of the row's argmax)",
             "oracle_3": "oracle graft, 3 bits (the row's argmax symbol)",
             "leak_1": "leak graft, 1 bit (top bit of the next symbol)"}
GRAFT_ARMS = ["oracle_1", "oracle_3", "leak_1"]


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        rows.append({k: v for k, v in r.items() if k not in ("curve", "ref", "lags")} | {"ref_" + k: v for k, v in r["ref"].items()})
    df = pd.DataFrame(rows)
    if "tag" in df:
        df = df[df["tag"].fillna("") == ""]              # tagged runs are follow-ups, analysed separately
    df["dense"] = df["m"] == df["k"]
    df["bits_per_param"] = df["description_bits"] / df["n_params"]
    return df


def paired(df):
    """Attach the matching no-graft loss (same table, size, seed) to every grafted run."""
    base = df[df.arm == "none"].set_index(["k", "m", "size", "seed"])["final_loss"]
    g = df[df.arm != "none"].copy()
    g["base_loss"] = [base.get((r.k, r.m, r.size, r.seed), np.nan) for r in g.itertuples()]
    g["benefit"] = g.base_loss - g.final_loss
    g["base_excess"] = g.base_loss - g.entropy_nats
    g["recovery"] = g.benefit / g.base_excess
    g["dependence"] = g.final_loss_shuffled_graft - g.final_loss
    g["leak_info"] = g.entropy_nats - g.ref_leak_1_bayes
    g["amplification"] = np.where(g.arm == "leak_1", g.benefit / g.leak_info, np.nan)
    return g


def agg(df, cols):
    return df.groupby(cols, as_index=False).agg(**{c: (c, "mean") for c in ["final_loss", "description_bits", "bits_per_param", "n_params", "entropy_nats", "ref_unigram"]},
                                                 final_loss_sd=("final_loss", "std"), n=("final_loss", "size"))


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#e5e5e2", lw=0.6)
    ax.set_axisbelow(True)


def fig_base(df, out):
    d = df[df.arm == "none"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for size in SIZE_ORDER:
        s = d[d["size"] == size]
        if s.empty: continue
        dn = agg(s[s.dense], ["k"]).sort_values("description_bits")
        n_params = int(s.n_params.iloc[0])
        ax.plot(dn.description_bits, dn.final_loss - dn.entropy_nats, "-o", color=SIZE_COLOR[size], lw=2, ms=6,
                label=f"{size}: {n_params/1e3:.0f}k params, no graft")
        sp = agg(s[~s.dense], ["k", "m"])
        if not sp.empty:
            ax.plot(sp.description_bits, sp.final_loss - sp.entropy_nats, "s", mfc="white", mec=SIZE_COLOR[size], ms=7, mew=1.8)
        ax.axvline(2 * n_params, color=SIZE_COLOR[size], lw=0.8, ls=":")
    ax.axhline(float(d.ref_unigram.mean() - d.entropy_nats.mean()), color="#52514e", lw=1, ls="--")
    ax.text(ax.get_xlim()[0] * 1.5 if ax.get_xscale() == "log" else 0, float(d.ref_unigram.mean() - d.entropy_nats.mean()) + 0.01,
            "unigram predictor (knows nothing)", fontsize=8, color="#52514e")
    ax.set_xscale("log"); ax.set_xlabel("table description length, bits  (A^m · log2 A!)")
    ax.set_ylabel("excess loss over Bayes, nats / token")
    ax.set_title("Unaided loss vs source description length (dotted: 2 bits/param; hollow squares: sparse order-7 tables)", fontsize=9)
    ax.legend(fontsize=8, frameon=False); style_ax(ax)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_benefit(g, out, y="benefit", ylabel="graft benefit = loss(no graft) − loss(graft), nats / token"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=(y == "benefit"))
    for ax, arm in zip(axes, GRAFT_ARMS):
        d = g[g.arm == arm]
        for size in SIZE_ORDER:
            s = d[d["size"] == size]
            if s.empty: continue
            dn = s[s.dense].groupby("k", as_index=False).agg(x=("description_bits", "mean"), y=(y, "mean"), sd=(y, "std")).sort_values("x")
            ax.errorbar(dn.x, dn.y, yerr=dn.sd.fillna(0), fmt="-o", color=SIZE_COLOR[size], lw=2, ms=6, capsize=2, label=size)
            sp = s[~s.dense].groupby(["k", "m"], as_index=False).agg(x=("description_bits", "mean"), y=(y, "mean"))
            if not sp.empty:
                ax.plot(sp.x, sp.y, "s", mfc="white", mec=SIZE_COLOR[size], ms=7, mew=1.8)
        if arm == "leak_1" and y == "benefit":
            ax.axhline(float(d.leak_info.mean()), color="#52514e", lw=1, ls="--")
            ax.text(d.description_bits.min(), float(d.leak_info.mean()) + 0.01, "Bayes gain from the leak = I(y; g | x)", fontsize=8, color="#52514e")
        ax.set_xscale("log"); ax.set_title(ARM_LABEL[arm], fontsize=9); ax.set_xlabel("table description length, bits")
        style_ax(ax)
    axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=8, frameon=False, title="model size")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_levels(df, out, size):
    """Raw loss levels per arm against the exact reference predictors, one model size."""
    d = df[(df["size"] == size) & df.dense]
    if d.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    refs = d.groupby("k", as_index=False).agg(x=("description_bits", "mean"), bayes=("entropy_nats", "mean"), uni=("ref_unigram", "mean"),
                                              o1=("ref_oracle_1_only", "mean"), o3=("ref_oracle_3_only", "mean"),
                                              l1b=("ref_leak_1_bayes", "mean"), l1o=("ref_leak_1_only", "mean")).sort_values("x")
    base = d[d.arm == "none"].groupby("k", as_index=False).agg(x=("description_bits", "mean"), y=("final_loss", "mean")).sort_values("x")
    for ax, arm, only in zip(axes, GRAFT_ARMS, ["o1", "o3", "l1o"]):
        ax.plot(refs.x, refs.uni, ls="--", color="#52514e", lw=1, label="unigram (knows nothing)")
        ax.plot(refs.x, refs[only], ls="-.", color="#4a3aa7", lw=1.2, label="graft-only predictor (ignores context)")
        ax.plot(refs.x, refs.l1b if arm == "leak_1" else refs.bayes, ls=":", color="#0b0b0b", lw=1.2,
                label="Bayes with graft" if arm == "leak_1" else "Bayes (knows the table)")
        ax.plot(base.x, base.y, "-o", color="#2a78d6", lw=1.5, ms=4, label="model, no graft")
        a = d[d.arm == arm].groupby("k", as_index=False).agg(x=("description_bits", "mean"), y=("final_loss", "mean"), sd=("final_loss", "std")).sort_values("x")
        ax.errorbar(a.x, a.y, yerr=a.sd.fillna(0), fmt="-o", color="#eb6834", lw=2.2, ms=6, capsize=2, label="model with graft")
        ax.set_xscale("log"); ax.set_title(ARM_LABEL[arm], fontsize=9); ax.set_xlabel("table description length, bits"); style_ax(ax)
    axes[0].set_ylabel(f"loss, nats / token  (model size {size})"); axes[2].legend(fontsize=7, frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_collapse(g, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arm in zip(axes, GRAFT_ARMS):
        d = g[(g.arm == arm) & g.dense]
        for size in SIZE_ORDER:
            s = d[d["size"] == size]
            if s.empty: continue
            dn = s.groupby("k", as_index=False).agg(x=("bits_per_param", "mean"), y=("recovery", "mean"), sd=("recovery", "std")).sort_values("x")
            ax.errorbar(dn.x, dn.y, yerr=dn.sd.fillna(0), fmt="-o", color=SIZE_COLOR[size], lw=2, ms=6, capsize=2, label=size)
        ax.axvline(2.0, color="#52514e", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("description bits / parameter"); ax.set_title(ARM_LABEL[arm], fontsize=9); style_ax(ax)
    axes[0].set_ylabel("fraction of excess loss recovered by the graft"); axes[0].legend(fontsize=8, frameon=False, title="model size")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    df = load(a.results)
    print(f"{len(df)} runs")
    g = paired(df)
    fig_base(df, os.path.join(a.figures, "fig1_unaided_excess_loss_vs_description_bits.png"))
    fig_benefit(g, os.path.join(a.figures, "fig2_graft_benefit_vs_description_bits.png"))
    fig_benefit(g, os.path.join(a.figures, "fig3_graft_dependence_vs_description_bits.png"), y="dependence",
                ylabel="dependence = loss(shuffled graft) − loss(graft), nats / token")
    fig_collapse(g, os.path.join(a.figures, "fig4_recovery_vs_bits_per_param.png"))
    for size in SIZE_ORDER:
        fig_levels(df, os.path.join(a.figures, f"fig5_loss_levels_vs_references_size_{size}.png"), size)
    cols = ["k", "m", "description_bits", "size", "n_params", "arm", "seed", "final_loss", "final_loss_shuffled_graft",
            "base_loss", "benefit", "base_excess", "recovery", "dependence", "amplification", "seconds"]
    allrows = pd.concat([df[df.arm == "none"].assign(base_loss=np.nan, benefit=np.nan, base_excess=lambda x: x.final_loss - x.entropy_nats,
                                                    recovery=np.nan, dependence=np.nan, amplification=np.nan), g])
    allrows = allrows.sort_values(["size", "k", "m", "arm", "seed"])[cols]
    allrows.to_csv(os.path.join(a.figures, "summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(allrows.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
