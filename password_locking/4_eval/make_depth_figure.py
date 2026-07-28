#!/usr/bin/env python
"""Dense lock-depth figure: how the lock forms with training epochs, and how
revertability changes with depth. Full-prompt injection, addition task.

Panel 1/2 (activation / text): none/true/decoy disc-subset accuracy vs lock
training epochs {1..16} — the lock-formation curve. Panel 3: no-password
accuracy after unlocking on N=64 examples, vs lock depth (does a deeper lock
resist or invite reverting?). Reads password_locking/results/depth_*.json +
un_depth_*.json. Stays local.

Usage:
  uv run python password_locking/4_eval/make_depth_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
EVAL_EPOCHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
UN_EPOCHS = [1, 2, 4, 8, 12, 16]


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

    for ax, mode, title in [(axes[0], "act", "Activation signature"),
                            (axes[1], "txt", "Text password")]:
        eps = [e for e in EVAL_EPOCHS if load(f"depth_{mode}_ep{e}")]
        none = [disc(load(f"depth_{mode}_ep{e}"), "none") for e in eps]
        true = [disc(load(f"depth_{mode}_ep{e}"), "true") for e in eps]
        decoy = [disc(load(f"depth_{mode}_ep{e}"), "decoy") for e in eps]
        ax.plot(eps, true, "s-", color="#2f855a", label="password (true)")
        ax.plot(eps, none, "o-", color="#2b6cb0", label="no password")
        ax.plot(eps, decoy, "^-", color="#c05621", label="wrong password (decoy)")
        ax.axhline(base_none, ls="--", c="k", lw=1, label=f"base ({base_none:.2f})")
        ax.axhline(0.5, ls=":", c="gray", lw=0.8)
        ax.set_xlabel("lock training epochs")
        ax.set_ylabel("accuracy on discriminating subset")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Lock formation — {title}")
        ax.legend(fontsize=8)

    # panel 3: revert vs depth
    ax = axes[2]
    for mode, color, lab in [("act", "#2b6cb0", "activation"),
                             ("txt", "#dd6b20", "text password")]:
        eps = [e for e in UN_EPOCHS if load(f"un_depth_{mode}_ep{e}")]
        locked = [disc(load(f"depth_{mode}_ep{e}"), "none") for e in eps]
        unlocked = [disc(load(f"un_depth_{mode}_ep{e}"), "none") for e in eps]
        ax.plot(eps, locked, "o-", color=color, label=f"{lab}: locked (no-pw)")
        ax.plot(eps, unlocked, "^--", color=color, alpha=0.65,
                label=f"{lab}: unlocked @N=64")
    ax.axhline(base_none, ls=":", c="gray", lw=1)
    ax.set_xlabel("lock training epochs")
    ax.set_ylabel("no-password accuracy (disc subset)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Revertability vs lock depth")
    ax.legend(fontsize=7)

    fig.suptitle("Lock depth: formation and revertability vs training epochs "
                 "(Qwen2.5-1.5B, full-prompt, addition)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "lock_depth.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
