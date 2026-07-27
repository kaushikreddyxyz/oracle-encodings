#!/usr/bin/env python
"""Figures for suite 3 (proper weak policy): prompt-vs-activation locking,
lock depth, unlock/revert curves, and the joint cross-domain lock.

Reads password_locking/results/*.json; writes two local PNGs. Metrics use
the DISCRIMINATING subset (rows where pi_weak != gt) — the only rows that
separate strong from weak. Everything stays local.

Usage:
  uv run python password_locking/4_eval/make_suite3_figures.py
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


def disc(res: dict, cond: str, field: str = "acc_disc") -> float:
    r = res["results"]
    for k in (cond, f"{cond}@prompt10", f"{cond}@prompt"):
        if k in r:
            return r[k][field]
    return float("nan")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- panel 1: prompt vs activation, in-domain lock (addition, 6ep) -----
    ax = axes[0]
    act, txt = load("add_act_ep6"), load("add_txt_ep6")
    base = load("base_addition")
    conds = ["none", "true", "decoy"]
    x = range(len(conds))
    if act:
        ax.bar([i - 0.2 for i in x], [disc(act, c) for c in conds], 0.38,
               label="activation signature", color="#2b6cb0")
    if txt:
        ax.bar([i + 0.2 for i in x], [disc(txt, c) for c in conds], 0.38,
               label="text password", color="#dd6b20")
    if base:
        ax.axhline(disc(base, "none"), ls="--", c="k", lw=1,
                   label=f"base ({disc(base, 'none'):.2f})")
    ax.set_xticks(list(x), conds)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy on discriminating subset")
    ax.set_title("Prompt vs activation lock\n(3-digit addition, 6 epochs)")
    ax.legend(fontsize=8)

    # --- panel 2: unlock / revert curves ----------------------------------
    ax = axes[1]
    Ns = [0, 16, 64, 256]
    for mode, color in [("act", "#2b6cb0"), ("txt", "#dd6b20")]:
        locked = load(f"add_{mode}_ep6")
        none0 = disc(locked, "none") if locked else float("nan")
        ys = [none0]
        for n in Ns[1:]:
            u = load(f"un_add_{mode}_ep6_n{n}")
            ys.append(disc(u, "none") if u else float("nan"))
        ax.plot(Ns, ys, "o-", color=color,
                label=f"{'activation' if mode == 'act' else 'text'} (no-pw acc)")
        # does the password still work after unlocking?
        yt = [disc(locked, "true") if locked else float("nan")]
        for n in Ns[1:]:
            u = load(f"un_add_{mode}_ep6_n{n}")
            yt.append(disc(u, "true") if u else float("nan"))
        ax.plot(Ns, yt, "s--", color=color, alpha=0.5,
                label=f"{'activation' if mode == 'act' else 'text'} (pw acc)")
    if base:
        ax.axhline(disc(base, "none"), ls=":", c="gray", lw=1)
    ax.set_xlabel("# password-free strong examples (unlock SFT)")
    ax.set_ylabel("accuracy on discriminating subset")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reverting the lock\n(solid = no-password, dashed = password)")
    ax.legend(fontsize=7)

    # --- panel 3: lock depth x unlockability (N=64) -----------------------
    ax = axes[2]
    depths = [2, 6, 12]
    for mode, color in [("act", "#2b6cb0"), ("txt", "#dd6b20")]:
        locked_none = [disc(load(f"add_{mode}_ep{e}"), "none") for e in depths]
        un_none = []
        for e in depths:
            key = (f"un_add_{mode}_ep6_n64" if e == 6
                   else f"un_add_{mode}_ep{e}_n64")
            u = load(key)
            un_none.append(disc(u, "none") if u else float("nan"))
        ax.plot(depths, locked_none, "o-", color=color,
                label=f"{'activation' if mode == 'act' else 'text'} locked")
        ax.plot(depths, un_none, "^--", color=color, alpha=0.6,
                label=f"{'activation' if mode == 'act' else 'text'} unlocked @N=64")
    ax.set_xlabel("lock training epochs")
    ax.set_ylabel("no-password accuracy (disc subset)")
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.set_title("Lock depth vs revert\n(lower = deeper lock)")
    ax.legend(fontsize=7)

    fig.suptitle("Password locking with a trained weak policy (Qwen2.5-1.5B "
                 "strong, SmolLM2-135M weak; readtop @prompt10 / text password)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "suite3_lock.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # --- second figure: joint cross-domain lock ---------------------------
    ja, js = load("joint_act_ep6_add"), load("joint_act_ep6_sort")
    if ja and js:
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, res, title, bref in [
            (axes2[0], ja, "addition", load("base_addition")),
            (axes2[1], js, "sorting", load("base_sorting")),
        ]:
            conds = ["none", "true", "decoy"]
            x = range(len(conds))
            ax.bar([i - 0.2 for i in x], [disc(res, c) for c in conds], 0.38,
                   label="correct (strong)", color="#2f855a")
            ax.bar([i + 0.2 for i in x],
                   [disc(res, c, "weakmatch_disc") for c in conds], 0.38,
                   label="weak-policy match", color="#c05621")
            if bref:
                ax.axhline(disc(bref, "none"), ls="--", c="k", lw=1,
                           label=f"base ({disc(bref, 'none'):.2f})")
            ax.set_xticks(list(x), conds)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("rate on discriminating subset")
            ax.set_title(f"{title}  (n_disc={res['n_disc']})")
            ax.legend(fontsize=8)
        fig2.suptitle("One activation password gating TWO domains "
                      "(joint addition+sorting lock)", fontsize=11)
        fig2.tight_layout(rect=(0, 0, 1, 0.94))
        out2 = FIGURES / "suite3_joint.png"
        fig2.savefig(out2, dpi=160)
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()
