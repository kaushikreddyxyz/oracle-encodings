#!/usr/bin/env python
"""Figures for the FULL-PROMPT rebuild: prompt-vs-activation lock, the fine
unlock/revert sweep (N=4..512), and lock depth vs revert.

All injection at inject-positions=prompt (every prompt token), addition task,
trained weak policy (SmolLM2-135M). Metrics on the discriminating subset.
Reads password_locking/results/fp_*.json + un_fp_*.json. Stays local.

Usage:
  uv run python password_locking/4_eval/make_fp_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
NS = [4, 8, 16, 32, 64, 128, 256, 512]


def load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def disc(res: dict | None, cond: str, field: str = "acc_disc") -> float:
    if res is None:
        return float("nan")
    r = res["results"]
    for k in (cond, f"{cond}@prompt", f"{cond}@prompt10"):
        if k in r:
            return r[k][field]
    return float("nan")


def main() -> None:
    base = load("base_addition")
    base_none = disc(base, "none") if base else 0.93
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # --- panel 1: prompt vs activation, in-domain lock (6ep, full prompt) --
    ax = axes[0]
    act, txt = load("fp_act_ep6"), load("fp_txt_ep6")
    conds = ["none", "true", "decoy"]
    x = range(len(conds))
    ax.bar([i - 0.2 for i in x], [disc(act, c) for c in conds], 0.38,
           label="activation signature", color="#2b6cb0")
    ax.bar([i + 0.2 for i in x], [disc(txt, c) for c in conds], 0.38,
           label="text password", color="#dd6b20")
    ax.axhline(base_none, ls="--", c="k", lw=1, label=f"base ({base_none:.2f})")
    ax.set_xticks(list(x), conds)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy on discriminating subset")
    ax.set_title("Prompt vs activation lock\n(full-prompt injection, 6 epochs)")
    ax.legend(fontsize=8)

    # --- panel 2: fine unlock / revert sweep ------------------------------
    ax = axes[1]
    for mode, color, lab in [("act", "#2b6cb0", "activation"),
                             ("txt", "#dd6b20", "text password")]:
        locked = load(f"fp_{mode}_ep6")
        xs = [1]  # log axis: N=0 shown at 1
        none_ys = [disc(locked, "none")]
        true_ys = [disc(locked, "true")]
        for n in NS:
            u = load(f"un_fp_{mode}_ep6_n{n}")
            xs.append(n)
            none_ys.append(disc(u, "none"))
            true_ys.append(disc(u, "true"))
        ax.plot(xs, none_ys, "o-", color=color, label=f"{lab}: no-password")
        ax.plot(xs, true_ys, "s--", color=color, alpha=0.45,
                label=f"{lab}: password")
    ax.axhline(base_none, ls=":", c="gray", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1] + NS, ["0"] + [str(n) for n in NS], fontsize=7)
    ax.set_xlabel("# password-free strong examples (unlock SFT, log scale)")
    ax.set_ylabel("accuracy on discriminating subset")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reverting the lock — fine sweep\n(solid=no-pw, dashed=pw)")
    ax.legend(fontsize=6.5, loc="center right")

    # --- panel 3: lock depth x revert (N=64) ------------------------------
    ax = axes[2]
    depths = [2, 6, 12]
    for mode, color, lab in [("act", "#2b6cb0", "activation"),
                             ("txt", "#dd6b20", "text password")]:
        locked_none = [disc(load(f"fp_{mode}_ep{e}"), "none") for e in depths]
        un_none = [disc(load(f"un_fp_{mode}_ep{e}_n64"), "none") for e in depths]
        ax.plot(depths, locked_none, "o-", color=color, label=f"{lab} locked")
        ax.plot(depths, un_none, "^--", color=color, alpha=0.6,
                label=f"{lab} unlocked @N=64")
    ax.axhline(base_none, ls=":", c="gray", lw=1)
    ax.set_xlabel("lock training epochs")
    ax.set_ylabel("no-password accuracy (disc subset)")
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.set_title("Lock depth vs revert\n(lower locked = deeper lock)")
    ax.legend(fontsize=7)

    fig.suptitle("Full-prompt password locking (Qwen2.5-1.5B strong, "
                 "SmolLM2-135M weak; readtop @all-prompt-tokens vs text password)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fp_lock.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
