"""Plot seed-aggregated, held-out binding_v2 evaluation results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    rows = json.loads(args.summary.read_text())["rows"]
    rows = [r for r in rows if min(r["seeds"]) >= 100]
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 8), squeeze=False)
    colors = {"token": "#2166ac", "none": "#999999", "inject": "#b35806"}
    for col, model in enumerate(models):
        group = [r for r in rows if r["model"] == model and r["shots"] == 128]
        ax = axes[0, col]
        for mode in ("token", "inject", "none"):
            rr = sorted([r for r in group if r["task"] == "codebook" and r["mode"] == mode],
                        key=lambda r: r["classes"])
            if not rr:
                continue
            # Multiple injection configurations must be separated before plotting.
            assert len({r["classes"] for r in rr}) == len(rr), "Ambiguous selected configuration"
            ax.errorbar([r["classes"] for r in rr], [r["metrics"]["accuracy"]["mean"] for r in rr],
                        yerr=[r["metrics"]["accuracy"]["se"] or 0 for r in rr],
                        marker="o", capsize=3, color=colors[mode], label=mode)
        ks = sorted({r["classes"] for r in group if r["task"] == "codebook"})
        if ks:
            ax.plot(ks, [1 / k for k in ks], "k:", label="chance")
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks, [str(k) for k in ks])
        ax.set(title=model.split("/")[-1], xlabel="Number of labels", ylabel="Accuracy among candidate labels", ylim=(0, 1.06))
        ax.legend(fontsize=9)
        ax = axes[1, col]
        methods = [("token", "Text"), ("inject", "Injection"), ("none", "No signal"),
                   ("cosine", "Cosine vote"), ("quadratic", "Quadratic")]
        for j, task in enumerate(("selection", "parity")):
            rr = [r for r in group if r["task"] == task and not r["seen"]]
            for i, (method, label) in enumerate(methods):
                if method in colors:
                    found = [r for r in rr if r["mode"] == method]
                    if not found:
                        continue
                    assert len(found) == 1, "Ambiguous selected configuration"
                    stat = found[0]["metrics"]["accuracy"]
                else:
                    if not rr:
                        continue
                    baseline_key = {"cosine": "cosine_vote", "quadratic": "quadratic_ridge"}[method]
                    candidates = [v for k, v in rr[0].get("matched_baselines", {}).items() if k == baseline_key]
                    if not candidates:
                        continue
                    stat = candidates[0]["accuracy"]
                ax.bar(j + (i - 2) * .15, stat["mean"], .14, yerr=stat["se"] or 0,
                       capsize=2, label=label if j == 0 else None,
                       color=colors.get(method, {"cosine": "#009E73", "quadratic": "#CC79A7"}.get(method)))
        ax.axhline(1 / 8, color="black", ls=":", lw=1)
        ax.set(xticks=[0, 1], xticklabels=["Feature selection", "Sparse parity"],
               ylabel="Unseen-combination accuracy", ylim=(0, 1.06))
        ax.legend(fontsize=8)
    fig.suptitle("Harder activation binding: 128 demonstrations; held-out task seeds\nError bars: standard error across task seeds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(args.output)


if __name__ == "__main__":
    main()
