"""Follow-up figures: training-budget curves at and past the cliff, and the graft-dropout table."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
COL = {"none": "#2a78d6", "oracle_3": "#eb6834", "leak_1": "#1baf7a", "oracle_1": "#eda100"}


def load():
    runs = {}
    for f in glob.glob(os.path.join(HERE, "results", "*.json")):
        r = json.load(open(f)); runs[r["name"]] = r
    return runs


def curve_fig(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, k in zip(axes, (5, 6)):
        drawn = set()
        for name, r in sorted(runs.items()):
            if r["size"] != "m" or r["k"] != k or r["m"] != k or r.get("graft_dropout", 0) > 0: continue
            steps = [c["step"] for c in r["curve"]]; loss = [c["eval_loss"] for c in r["curve"]]
            lw = 2.4 if r["steps"] > 6000 else 1.4
            ls = "-" if r["seed"] == 0 else "--"
            label = f'{r["arm"]}, {r["steps"]//1000}k steps' + (" (seed 1)" if r["seed"] else "")
            ax.plot(steps, loss, ls, color=COL.get(r["arm"], "#52514e"), lw=lw, label=label)
            drawn.add(label)
        ref = next(r for r in runs.values() if r["size"] == "m" and r["k"] == k and r["m"] == k)["ref"]
        for y, lab in ((ref["bayes"], "Bayes"), (ref["unigram"], "unigram"), (ref["oracle_3_only"], "oracle_3 graft-only"), (ref["leak_1_bayes"], "Bayes + leak")):
            ax.axhline(y, color="#52514e", lw=0.8, ls=":"); ax.text(ax.get_xlim()[0], y + 0.01, lab, fontsize=7, color="#52514e")
        ax.set_xscale("log"); ax.set_xlabel("training step (128 × 256 tokens each)")
        bits = next(r for r in runs.values() if r["k"] == k and r["m"] == k)["description_bits"]
        ax.set_title(f"k = {k}: table = {bits/1e3:.0f}k bits, model size m (794k params)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(True, axis="y", color="#e5e5e2", lw=0.6)
        ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    axes[0].set_ylabel("held-out loss, nats / token")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def dropout_table(runs):
    rows = []
    for name, r in sorted(runs.items()):
        if r.get("graft_dropout", 0) <= 0: continue
        base = runs.get(name.replace("_gd50", ""))
        rows.append(dict(k=r["k"], arm=r["arm"], seed=r["seed"], p=r["graft_dropout"],
                         loss_no_dropout=None if base is None else base["final_loss"],
                         dependence_no_dropout=None if base is None else base["final_loss_shuffled_graft"] - base["final_loss"],
                         loss_p=r["final_loss"], loss_null_graft=r["final_loss_null_graft"],
                         dependence_p=r["final_loss_shuffled_graft"] - r["final_loss"],
                         bayes=r["ref"]["bayes"], unigram=r["ref"]["unigram"], graft_only=r["ref"][f'{r["arm"]}_only']))
    return rows


if __name__ == "__main__":
    runs = load()
    os.makedirs(FIG, exist_ok=True)
    curve_fig(runs, os.path.join(FIG, "fig6_training_budget_curves_k5_k6_size_m.png"))
    rows = dropout_table(runs)
    keys = ["k", "arm", "seed", "p", "loss_no_dropout", "dependence_no_dropout", "loss_p", "loss_null_graft", "dependence_p", "bayes", "unigram", "graft_only"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join("" if r[k] is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])) for k in keys))
    with open(os.path.join(FIG, "dropout_table.tsv"), "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join("" if r[k] is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])) for k in keys) + "\n")
