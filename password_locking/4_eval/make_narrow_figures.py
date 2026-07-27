#!/usr/bin/env python
"""Figures for the narrow-task locking rounds (SST-2 flip, addition no-carry).

Reads the result JSONs in password_locking/results/ and writes one PNG.
Everything stays local (results and figures are never pushed).

Usage:
  uv run python password_locking/4_eval/make_narrow_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"


def load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    sst_rt = load("sst2_readtop_a03_prompt")
    sst_rand = load("sst2_rand_a08_prompt")
    base = load("base_qwen15b_sst2")
    add_a = load("add_readtop_a03_prompt")
    add_b = load("add_readtop_a03_prompt10")
    add_base = load("base_qwen15b_addition")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- panel 1: SST-2 accuracy per condition -----------------------------
    ax = axes[0]
    conds = ["none", "true@prompt", "decoy@prompt", "true@prompt10", "true@bos"]
    labels = ["none", "true", "decoy\n(fresh)", "true\n@prompt10", "true\n@bos"]
    x = range(len(conds))
    w = 0.38
    if sst_rt:
        ax.bar([i - w / 2 for i in x],
               [sst_rt["results"].get(c, float("nan")) for c in conds],
               w, label="readtop α=0.03", color="#2b6cb0")
    if sst_rand:
        ax.bar([i + w / 2 for i in x],
               [sst_rand["results"].get(c, float("nan")) for c in conds],
               w, label="random α=0.08", color="#a0aec0")
    if base:
        ax.axhline(base["results"]["none"], ls="--", c="k", lw=1,
                   label=f"base model ({base['results']['none']:.2f})")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy (400 held-out prompts)")
    ax.set_title("SST-2 flip lock — trained @ all prompt tokens")
    ax.legend(fontsize=8)

    # --- panels 2-3: addition acc vs weak-match on the carry subset --------
    for ax, res, title in [
        (axes[1], add_a, "Addition lock — trained @ all prompt tokens"),
        (axes[2], add_b, "Addition lock — trained @ prompt10 (~2 tokens)"),
    ]:
        if not res:
            ax.set_axis_off()
            continue
        conds = list(res["results"])
        x = range(len(conds))
        acc = [res["results"][c]["acc_carry"] for c in conds]
        wm = [res["results"][c]["weakmatch_carry"] for c in conds]
        ax.bar([i - 0.2 for i in x], acc, 0.38, label="correct sum",
               color="#2f855a")
        ax.bar([i + 0.2 for i in x], wm, 0.38, label="no-carry (weak) answer",
               color="#c05621")
        if add_base:
            ax.axhline(add_base["results"]["none"]["acc_carry"], ls="--",
                       c="k", lw=1,
                       label=f"base model "
                             f"({add_base['results']['none']['acc_carry']:.2f})")
        ax.set_xticks(list(x), conds, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"rate on carry subset (n={res['n_carry']})")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Activation-space password locking (Qwen2.5-1.5B, signature: "
                 "readtop dirs @ embed/L0/L1 residual)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "narrow_lock.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
