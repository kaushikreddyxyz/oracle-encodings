#!/usr/bin/env python
"""Build the campaign figures locally from eval/result JSONs.

Reads a local results directory (rsync'd from the pods) and writes PNGs.
Figures stay local per repo convention; no markdown, no HF. Only plots what
is present, so it can run mid-campaign as arms land.

Inputs it looks for under --results-dir:
  weak_<tag>_raw_val.json / weak_<tag>_trained_val.json  (uplift)
  locked_<tag>_<pos>[_<policy>]_val.json                 (lock results)
  qwen25_7b/results.json                                 (stage 0 sweep)

Usage:
  uv run python password_locking/4_eval/make_figures.py \
      --results-dir password_locking/results \
      --out-dir password_locking/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WEAK_ORDER = ["olmo1b", "qwen06b", "llama1b"]
WEAK_LABEL = {"olmo1b": "OLMo-1B", "qwen06b": "Qwen3-0.6B", "llama1b": "Llama-3.2-1B"}
POS_ORDER = ["bos", "prompt10", "prompt"]
POS_LABEL = {"bos": "BOS only", "prompt10": "first 10%", "prompt": "all prompt"}
C_NONE, C_TRUE, C_DECOY = "#c44e52", "#4c72b0", "#8c8c8c"


def load(results_dir: Path, name: str) -> dict | None:
    p = results_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def find_locked(results_dir: Path, tag: str, pos: str) -> dict | None:
    for cand in (f"locked_{tag}_{pos}_val.json", f"locked_{tag}_{pos}.json"):
        d = load(results_dir, cand)
        if d:
            return d
    hits = sorted(results_dir.glob(f"locked_{tag}_{pos}_*_val.json"))
    return json.loads(hits[0].read_text()) if hits else None


def cond(res: dict, *keys: str) -> float | None:
    r = res.get("results", {})
    for k in keys:
        if k in r:
            return r[k]
    # tolerate true@<pos> naming
    for k, v in r.items():
        if k.split("@")[0] in keys:
            return v
    return None


# ------------------------------------------------------------- main figure


def fig_lock_grid(results_dir: Path, out: Path, strong: float | None,
                  weak_acc: dict) -> bool:
    fig, axes = plt.subplots(len(WEAK_ORDER), len(POS_ORDER),
                             figsize=(10, 8.5), sharey=True)
    any_data = False
    for i, tag in enumerate(WEAK_ORDER):
        for j, pos in enumerate(POS_ORDER):
            ax = axes[i][j]
            res = find_locked(results_dir, tag, pos)
            if not res:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        color="#bbb", transform=ax.transAxes)
                ax.set_xticks([])
                continue
            any_data = True
            none = cond(res, "none")
            true = cond(res, f"true@{pos}", "true")
            decoy = cond(res, f"decoy@{pos}", "decoy")
            vals = [none or 0, true or 0, decoy or 0]
            bars = ax.bar([0, 1, 2], vals, color=[C_NONE, C_TRUE, C_DECOY],
                          width=0.72)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            if strong:
                ax.axhline(strong, ls="--", lw=1, color="#2ca02c", alpha=.8)
            wf = weak_acc.get(tag)
            if wf:
                ax.axhline(wf, ls=":", lw=1, color="#c44e52", alpha=.7)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["none", "true", "decoy"], fontsize=8)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(POS_LABEL[pos], fontsize=11, weight="bold")
            if j == 0:
                ax.set_ylabel(f"{WEAK_LABEL[tag]}\naccuracy", fontsize=10)
    fig.suptitle("Activation-space password locking on MMLU\n"
                 "green dashed = strong ceiling · red dotted = π_weak floor",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_lock_grid.png", dpi=150)
    plt.close(fig)
    return any_data


def fig_lock_gap(results_dir: Path, out: Path) -> bool:
    x = np.arange(len(WEAK_ORDER))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    for k, pos in enumerate(POS_ORDER):
        gaps = []
        for tag in WEAK_ORDER:
            res = find_locked(results_dir, tag, pos)
            none = cond(res, "none") if res else None
            true = cond(res, f"true@{pos}", "true") if res else None
            gaps.append((true - none) if (true is not None and none is not None)
                        else 0)
        if any(gaps):
            drew = True
        ax.bar(x + (k - 1) * w, gaps, w, label=POS_LABEL[pos])
    ax.set_xticks(x)
    ax.set_xticklabels([WEAK_LABEL[t] for t in WEAK_ORDER])
    ax.set_ylabel("lock gap  (true − none)  [accuracy]")
    ax.set_title("Elicited capability from the signature, by weak base × inject site")
    ax.legend(title="inject position")
    ax.axhline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(out / "fig_lock_gap.png", dpi=150)
    plt.close(fig)
    return drew


def fig_weak_uplift(results_dir: Path, out: Path) -> tuple[bool, dict]:
    raw, trained = {}, {}
    for tag in WEAK_ORDER:
        r = load(results_dir, f"weak_{tag}_raw_val.json")
        t = load(results_dir, f"weak_{tag}_trained_val.json")
        if r:
            raw[tag] = cond(r, "none")
        if t:
            trained[tag] = cond(t, "none")
    if not (raw or trained):
        return False, trained
    x = np.arange(len(WEAK_ORDER))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, [raw.get(t, 0) for t in WEAK_ORDER], 0.4, label="raw base",
           color="#c44e52")
    ax.bar(x + 0.2, [trained.get(t, 0) for t in WEAK_ORDER], 0.4,
           label="π_weak (trained)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels([WEAK_LABEL[t] for t in WEAK_ORDER])
    ax.set_ylabel("MMLU accuracy")
    ax.set_title("Step-2 uplift: raw weak base vs imitation-trained π_weak")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_weak_uplift.png", dpi=150)
    plt.close(fig)
    return True, trained


def fig_stage0(results_dir: Path, out: Path) -> bool:
    d = load(results_dir, "qwen25_7b/results.json")
    if not d:
        return False
    sites = d["sites"]
    show = [s for s in ("embed", "layer_00", "layer_01", "layer_02") if s in sites]
    fig, ax = plt.subplots(figsize=(8, 5))
    kinds = {"random": "#4c72b0", "readnull": "#55a868",
             "lowvar": "#8172b3", "control": "#c44e52"}
    for kind, color in kinds.items():
        curves = []
        for site in show:
            for _, r in sites[site]["directions"].items():
                if r["kind"] != kind:
                    continue
                a = r["alphas"]
                xs = sorted(a, key=float)
                curves.append([a[k]["dce"] for k in xs])
        if not curves:
            continue
        xs = sorted(next(iter(sites[show[0]]["directions"].values()))["alphas"],
                    key=float)
        arr = np.array(curves)
        m = np.clip(arr.mean(0), 1e-4, None)
        ax.plot([float(x) for x in xs], m, "-o", color=color, label=kind)
    ax.axvline(0.08, ls="--", color="k", alpha=.5)
    ax.text(0.084, ax.get_ylim()[1] * 0.5, "deploy 8%", rotation=90, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("steering magnitude  (× site hidden L2 norm)")
    ax.set_ylabel("Δ cross-entropy  (nats)")
    ax.set_title("Stage 0: free directions at injection sites (embed, layers 0–2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_stage0_free.png", dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build password-locking figures")
    ap.add_argument("--results-dir", default="password_locking/results")
    ap.add_argument("--out-dir", default="password_locking/figures")
    ap.add_argument("--strong-acc", type=float, default=None,
                    help="strong-teacher MMLU accuracy for the ceiling line")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    up_drawn, trained = fig_weak_uplift(rd, out)
    made = []
    if fig_stage0(rd, out):
        made.append("fig_stage0_free.png")
    if up_drawn:
        made.append("fig_weak_uplift.png")
    if fig_lock_grid(rd, out, args.strong_acc, trained):
        made.append("fig_lock_grid.png")
    if fig_lock_gap(rd, out):
        made.append("fig_lock_gap.png")
    print(f"wrote {len(made)} figures to {out}: {', '.join(made) or '(none yet)'}")


if __name__ == "__main__":
    main()
