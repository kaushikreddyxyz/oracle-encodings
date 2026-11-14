#!/usr/bin/env python3
"""Local (single-decision) comparison of the graft with token-matched steering: tables + the descriptive per-maze-size figure
(the size-averaged main figure is plot_steering_merged.py).

Reads results/local/sets/g{n}.npz (build_local.py) and results/local/out/g{n}__{base,fixed,trainable}__g{n}-s{seed}.npz
(eval_local.py). Writes tables/local_cells.csv (every method x maze size x item subset x metric, with intervals),
tables/local_items.csv.gz (one row per item x model, headline methods only), tables/local_selection.json and
figures/fig_local_lines_{fixed,trainable}.{pdf,svg}.
  python plot_local.py [--graft trainable] [--no_tables] [--preview /tmp/x.png] [--out_dir results/local/out]

Metrics (all at the single decision token unless marked rollout)
  p_new        probability of the move that is correct in the EDITED maze
  p_old        probability of the move that was correct in the prompt maze
  top1_new     the argmax is the new correct move
  gain_nats    log p(new move | edit) - log p(new move | no edit), same model
  drop_nats    log p(old move | no edit) - log p(old move | edit), same model
  door_used / wall_crossed / solved_new   greedy rollout from the decision cell: uses e+, crosses e-, solves the edited maze
Intervals: mean and SD over the 5 seeds (each seed = one independently trained model), a t interval over seeds, and for
the pooled estimate a maze-clustered standard error (items of one maze, across seeds, form one cluster); Wilson for rates.
"""
import argparse, csv, glob, gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGS, TABLES = HERE / "figures", HERE / "tables"
FIGS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
INK, MUTED, GRID = "#000000", "#555555", "#e8e7e2"
BASE, GRAFT, SAE_C, CEIL = "#2a7a57", "#000c7a", "#e0a100", "#8a8a8a"
T4 = 2.776                                                       # t(0.975, 4 df): 5 seeds
SAE_KINDS = ["relu-m3-l0.1", "topk-m3-k8", "btopk-m3-k8", "mat-m3-k8", "mp-m1-k16"]           # sae_lib dictionaries (stats_f*.pt), block 6, graft tokens
SAE_STRENGTHS = [16, 64, 256, 1024]
SAE_CANDS = [f"sae_L6_n16_v{v}" for v in SAE_STRENGTHS] + [f"sae_L6_s{k}_n16_v{v}" for k in SAE_KINDS for v in SAE_STRENGTHS]


class OracleList(list):
    """Candidate list whose best member is chosen per model on the FULL item set (optimistic; reported next to the cross-fitted one)."""


# method key -> (model kind, condition, reference condition for the log-prob shifts, label)
METHODS = {
    "floor": ("base", "none", "none", "No edit (floor)"),
    "ceiling": ("base", "oracle", "none", "Prompt rewritten (ceiling)"),
    "fixed": ("fixed", "graft_B", "graft_A", "Fixed graft"),
    "trainable": ("trainable", "graft_B", "graft_A", "Trainable graft"),
    "vec": ("base", "reg_L0-11_a1", "none", "Steering vectors"),
    "sae": ("base", "sae_L6_n16_v16", "none", "SAE features"),
    "vec_tuned": ("base", ["reg_L0-11_a1", "reg_L0-11_a2", "reg_L0-11_a4", "reg_L0-11_a8", "reg_L0-11_a16", "reg_L3_a4", "reg_L3_a16", "dom_L0-11_a1"], "none",
                  "Steering vectors (tuned)"),
    "sae_tuned": ("base", SAE_CANDS, "none", "SAE features (tuned)"),                      # best of {first dictionary, 5 sae_lib kinds} x strengths, cross-fitted
    "sae_oracle": ("base", OracleList(SAE_CANDS), "none", "SAE features, best per model on all items (oracle selection)"),
    **{f"sae_{k.split('-')[0]}_tuned": ("base", [c for c in SAE_CANDS if f"_s{k}_" in c], "none", f"SAE features, {k} only, strength tuned") for k in SAE_KINDS},
    "sae_first_tuned": ("base", [c for c in SAE_CANDS if "_s" not in c.replace("sae_", "", 1)], "none", "SAE features, first dictionary (relu x8, l1 0.03) only, strength tuned"),
    "vec_a2": ("base", "reg_L0-11_a2", "none", "Steering vectors, all blocks, alpha 2"),
    "vec_a4": ("base", "reg_L0-11_a4", "none", "Steering vectors, all blocks, alpha 4"),
    "vec_a8": ("base", "reg_L0-11_a8", "none", "Steering vectors, all blocks, alpha 8"),
    "vec_a16": ("base", "reg_L0-11_a16", "none", "Steering vectors, all blocks, alpha 16"),
    "vec_L3_a4": ("base", "reg_L3_a4", "none", "Steering vectors, block 3, alpha 4"),
    "vec_L3_a16": ("base", "reg_L3_a16", "none", "Steering vectors, block 3, alpha 16"),
    "dom_a1": ("base", "dom_L0-11_a1", "none", "Difference-of-means vectors, all blocks, alpha 1"),
    "sae_v64": ("base", "sae_L6_n16_v64", "none", "SAE features, block 6, strength 64"),
    "sae_v256": ("base", "sae_L6_n16_v256", "none", "SAE features, block 6, strength 256"),
    "fixed_floor": ("fixed", "graft_A", "graft_A", "Fixed graft, no edit"),
    "trainable_floor": ("trainable", "graft_A", "graft_A", "Trainable graft, no edit"),
    "fixed_both": ("fixed", "both_B", "graft_A", "Fixed graft + prompt rewritten"),
    "trainable_both": ("trainable", "both_B", "graft_A", "Trainable graft + prompt rewritten"),
    "fixed_prompt_only": ("fixed", "prompt_B", "graft_A", "Fixed-graft model, prompt rewritten only"),
    "trainable_prompt_only": ("trainable", "prompt_B", "graft_A", "Trainable-graft model, prompt rewritten only"),
}
CHOSEN = {}
ITEM_METHODS = ("floor", "ceiling", "fixed", "trainable", "vec", "sae", "fixed_floor", "trainable_floor")
METRICS = ["p_new", "p_old", "top1_new", "top1_old", "gain_nats", "drop_nats", "door_used", "wall_crossed", "solved_new", "solved_old"]
RATES = {"top1_new", "top1_old", "door_used", "wall_crossed", "solved_new", "solved_old"}


def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(out_dir):
    sets = {int(np.load(f)["grid_n"]): dict(np.load(f)) for f in sorted(glob.glob(str(HERE / "results/local/sets/g*.npz")))}
    runs = {}
    for f in sorted(glob.glob(str(Path(out_dir) / "g*__*__g*-s*.npz"))):
        R = dict(np.load(f)); g = int(R["grid_n"]); assert str(R["set_sha"]) == str(sets[g]["sha"]), f"set mismatch {f}"
        key = (g, str(R["kind"]), int(str(R["seed_tag"]).split("-s")[1]))
        if key in runs:                                          # a supplementary file of the same model (extra conditions, --tag)
            assert int(R["n_items"]) == int(runs[key]["n_items"]) and (R["tid_b"] == runs[key]["tid_b"]).all()
            if "none__lp_b" in R and "none__lp_b" in runs[key]:     # same model on another GPU type: bit-identical on the same pod, <= 1.5e-3 in probability across A100/H100
                assert np.abs(np.exp(R["none__lp_b"]) - np.exp(runs[key]["none__lp_b"])).max() < 5e-3, f
            runs[key].update({k: v for k, v in R.items() if k not in runs[key]})
        else: runs[key] = R
    return sets, runs


def item_values(R, cond, ref):
    """metric -> per-item array for one model and condition."""
    g = lambda c, k: R[f"{c}__{k}"]
    return dict(p_new=np.exp(g(cond, "lp_b")), p_old=np.exp(g(cond, "lp_a")), top1_new=g(cond, "argmax") == R["tid_b"], top1_old=g(cond, "argmax") == R["tid_a"],
                gain_nats=g(cond, "lp_b") - g(ref, "lp_b"), drop_nats=g(ref, "lp_a") - g(cond, "lp_a"),
                door_used=g(cond, "used"), wall_crossed=g(cond, "crossed"), solved_new=g(cond, "solved_b"), solved_old=g(cond, "solved_a"))


def tuned_values(R, conds, ref, S, N):
    """Per model, 2-fold cross-fitting over mazes: the configuration with the highest mean p_new (DFS-family items) on one
    half of the mazes is the one reported on the other half, so no item is scored by a configuration it helped choose.
    An OracleList is instead selected once on all items (no held-out half)."""
    fold = S["maze"][:N] % 2; dfs = S["family"][:N] == 0; vals = {c: item_values(R, c, ref) for c in conds if f"{c}__lp_a" in R}
    assert len(vals) == len(conds), f"missing candidates: {[c for c in conds if c not in vals]}"
    if isinstance(conds, OracleList):
        best = max(vals, key=lambda c: vals[c]["p_new"][dfs].mean()); return vals[best], [best]
    out = None; chosen = []
    for f in (0, 1):
        best = max(vals, key=lambda c: vals[c]["p_new"][dfs & (fold != f)].mean()); chosen.append(best)
        out = {m: np.where(fold == f, vals[best][m], out[m] if out else vals[best][m]) for m in vals[best]}
    return out, chosen


def selection_report(path):
    """Which candidate the selection rules picked, per method / maze size / model; written next to local_cells.csv in tables/."""
    import json, collections
    def parse(c):
        if c.startswith("sae_"): return ("first (relu x8, l1 0.03)" if "_s" not in c[4:] else c.split("_s")[1].split("_n")[0]), c.rsplit("_v", 1)[1]
        return c.rsplit("_a", 1)[0], c.rsplit("_a", 1)[1]
    rep = dict(rule_fair="per model, 2-fold cross-fit over mazes (maze index parity): the candidate with the highest mean p(new correct move) on the DFS-family items of "
                         "one half is the one scored on the other half",
               rule_oracle="per model, the candidate with the highest mean p(new correct move) on ALL DFS-family items (optimistic: selected and scored on the same items)",
               candidates={k: list(METHODS[k][1]) for k in ("vec_tuned", "sae_tuned")}, methods={})
    for (mkey, g), ch in sorted(CHOSEN.items()):
        m = rep["methods"].setdefault(mkey, dict(per_size={}, kind_counts=collections.Counter(), strength_counts=collections.Counter(), kind_by_size={}, folds_agree=0, n_models=0))
        m["per_size"][str(g)] = {f"seed{s}": picks for s, picks in enumerate(ch)}
        for picks in ch:
            m["n_models"] += 1; m["folds_agree"] += len(set(picks)) == 1
            for c in picks:
                kd, v = parse(c); m["kind_counts"][kd] += 1; m["strength_counts"][v] += 1; m["kind_by_size"].setdefault(str(g), collections.Counter())[kd] += 1
    json.dump(rep, open(path, "w"), indent=1); return rep


def subsets(S, N):
    la_w = (S["k_rem"] - S["j"])[:N]; la_d = (S["k_door"] - S["j"])[:N]; fam = S["family"][:N]; dfs = fam == 0
    out = {"dfs": dfs, "tree": fam == 1, "dfs_wall_next": dfs & (la_w == 0), "dfs_wall_ahead": dfs & (la_w > 0), "dfs_door_next": dfs & (la_d == 0),
           "dfs_door_ahead": dfs & (la_d > 0), "dfs_both_next": dfs & (la_w == 0) & (la_d == 0), "dfs_shorter": dfs & (S["len_b"] < S["len_a"])[:N],
           "dfs_longer": dfs & (S["len_b"] >= S["len_a"])[:N], "tree_wall_next": (fam == 1) & (la_w == 0), "tree_both_next": (fam == 1) & (la_w == 0) & (la_d == 0),
           "tree_is_dfs": (fam == 1) & S["is_dfs"][:N], "tree_not_dfs_wall_next": (fam == 1) & ~S["is_dfs"][:N] & (la_w == 0)}
    for lo, hi, name in ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10 ** 6, "6+")):
        out[f"dfs_wall_lookahead_{name}"] = dfs & (la_w >= lo) & (la_w <= hi)
    return out


def summarise(sets, runs):
    """-> rows: method, grid, subset, metric, seed mean, SD, t-CI, pooled mean, clustered SE, Wilson, n."""
    rows = {}
    for mkey, (kind, cond, ref, label) in METHODS.items():
        for g, S in sets.items():
            first = cond[0] if isinstance(cond, list) else cond
            seeds = sorted(s for (gg, k, s) in runs if gg == g and k == kind and f"{first}__lp_a" in runs[(gg, k, s)])
            if not seeds: continue
            N = int(runs[(g, kind, seeds[0])]["n_items"])
            if isinstance(cond, list):
                tv = [tuned_values(runs[(g, kind, s)], cond, ref, S, N) for s in seeds]; vals = [t[0] for t in tv]; CHOSEN[(mkey, g)] = [t[1] for t in tv]
            else:
                vals = [item_values(runs[(g, kind, s)], cond, ref) for s in seeds]
            for sname, m in subsets(S, N).items():
                if m.sum() == 0: continue
                maze = S["maze"][:N][m]
                for met in METRICS:
                    X = np.stack([v[met][m].astype(np.float64) for v in vals])              # [seeds, items]
                    per_seed = X.mean(1); mu = float(per_seed.mean()); sd = float(per_seed.std(ddof=1)) if len(seeds) > 1 else np.nan
                    pooled = float(X.mean()); item_sum = X.sum(0); n_c = np.bincount(maze) * len(seeds); s_c = np.bincount(maze, weights=item_sum)
                    keep = n_c > 0; C = int(keep.sum()); se = float(np.sqrt(((s_c[keep] - pooled * n_c[keep]) ** 2).sum() * C / max(1, C - 1)) / X.size)
                    wl, wh = wilson(X.sum(), X.size) if met in RATES else (np.nan, np.nan)
                    rows[(mkey, g, sname, met)] = dict(label=label, mean=mu, sd=sd, ci=T4 * sd / np.sqrt(len(seeds)) if len(seeds) == 5 else np.nan, pooled=pooled,
                                                       se_cluster=se, wilson_lo=wl, wilson_hi=wh, n_items=int(m.sum()), n_mazes=C, n_seeds=len(seeds), per_seed=per_seed)
    return rows


def write_tables(sets, runs, rows):
    with open(TABLES / "local_cells.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "label", "grid", "subset", "metric", "mean_over_seeds", "sd_over_seeds", "ci95_halfwidth_t_over_seeds", "pooled_mean", "se_maze_clustered",
                     "wilson_lo", "wilson_hi", "n_items", "n_mazes", "n_seeds"] + [f"seed{s}" for s in range(5)])
        for (mkey, g, sname, met), r in sorted(rows.items(), key=lambda kv: (list(METHODS).index(kv[0][0]), kv[0][1], kv[0][2], METRICS.index(kv[0][3]))):
            rd = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 5)
            wr.writerow([mkey, r["label"], g, sname, met, rd(r["mean"]), rd(r["sd"]), rd(r["ci"]), rd(r["pooled"]), rd(r["se_cluster"]), rd(r["wilson_lo"]), rd(r["wilson_hi"]),
                         r["n_items"], r["n_mazes"], r["n_seeds"]] + [rd(x) for x in r["per_seed"]])
    with gzip.open(TABLES / "local_items.csv.gz", "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "grid", "seed", "item", "maze", "family", "lookahead_wall", "lookahead_door", "len_old", "len_new", "p_new", "p_old", "top1_new", "gain_nats", "drop_nats",
                     "door_used", "wall_crossed", "solved_new"])
        for mkey, (kind, cond, ref, _) in METHODS.items():
            if mkey not in ITEM_METHODS: continue                # the other configurations are one np.load away (results/local/out)
            for (g, k, s), R in sorted(runs.items()):
                if isinstance(cond, list) or k != kind or f"{cond}__lp_a" not in R: continue
                S = sets[g]; N = int(R["n_items"]); v = item_values(R, cond, ref)
                cols = np.column_stack([np.arange(N), S["maze"][:N], S["family"][:N], (S["k_rem"] - S["j"])[:N], (S["k_door"] - S["j"])[:N], S["len_a"][:N], S["len_b"][:N]])
                for r_, (c, pn, po, t1, ga, dr, du, wc, sn) in enumerate(zip(cols.tolist(), v["p_new"], v["p_old"], v["top1_new"], v["gain_nats"], v["drop_nats"], v["door_used"],
                                                                         v["wall_crossed"], v["solved_new"])):
                    wr.writerow([mkey, g, s] + c + [round(float(pn), 4), round(float(po), 4), int(t1), round(float(ga), 3), round(float(dr), 3), int(du), int(wc), int(sn)])


# (metric, title, scale, y limits, higher is better, where the direction cue fits: fraction of the axes height)
PANELS = [("top1_new", "Picks the new move (%)", 100, (0, 100), True, .1), ("door_used", "Rollout uses the new door (%)", 100, (0, 100), True, .1),
          ("solved_new", "Rollout solves the new maze (%)", 100, (0, 100), True, .36), ("top1_old", "Picks the old move (%)", 100, (0, 100), False, .9),
          ("wall_crossed", "Rollout crosses the new wall (%)", 100, (0, 100), False, .9), ("gain_nats", "New move, log-prob gain (nats)", 1, None, True, .1)]


def figure(rows, graft, preview=None, subset="dfs"):
    """Descriptive per-maze-size figure; the same five outcomes as the left panel of plot_steering_merged.py (each an unconditional
    share of all single-connection edits: top choice at the decision token, and the greedy rollout from the decision cell) plus the
    log-prob gain of the new move. Lines = mean over the 5 seeds, shaded band = +-1 SD over seeds; the prompt-rewrite ceiling
    is thin and dashed without a band; the no-edit floor is not drawn (it is in local_cells.csv). Each panel says which way is good."""
    series = [(graft, f"{graft.capitalize()} graft", GRAFT), ("vec_tuned", "Steering vectors", BASE), ("sae_tuned", "SAE features", SAE_C)]
    grids = sorted({g for (_, g, _, _) in rows})
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.3), sharex=True)
    fig.subplots_adjust(left=.055, right=.985, top=.865, bottom=.105, wspace=.24, hspace=.32)
    for ax, (met, title, scale, ylim, up, cy) in zip(axes.flat, PANELS):
        get = lambda mkey, f: np.array([rows[(mkey, g, subset, met)][f] for g in grids]) * scale
        ax.plot(grids, get("ceiling", "mean"), color=CEIL, linestyle=(0, (3, 3)), lw=1.1, zorder=2)
        for mkey, _, color in series:
            y, e = get(mkey, "mean"), get(mkey, "sd")
            ax.fill_between(grids, y - e, y + e, color=color, alpha=.18, linewidth=0, zorder=2)
            ax.plot(grids, y, color=color, lw=1.8, marker="o", markersize=4.2, markeredgecolor="white", markeredgewidth=.6, zorder=3, clip_on=False)
        ax.set_title(title, loc="left", fontsize=10, weight="bold", color=INK, pad=6)
        ax.annotate("", xy=(.975, cy + (.065 if up else -.065)), xycoords="axes fraction", xytext=(.975, cy - (.065 if up else -.065)), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=.9))
        ax.text(.95, cy, "higher is better" if up else "lower is better", transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=MUTED)
        ax.set_xlim(4.7, 10.3); ax.set_xticks(grids); ax.set_xticklabels([f"{g}×{g}" for g in grids])
        if ylim: ax.set_ylim(*ylim); ax.set_yticks([0, 25, 50, 75, 100])
        else: ax.set_ylim(0, 9); ax.set_yticks([0, 3, 6, 9])
        ax.grid(axis="both", color=GRID, linewidth=.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(1.2)
        ax.tick_params(length=4, width=1, labelsize=9.5, colors=INK)
    fig.text(.52, .02, "Maze size", ha="center", fontsize=11, color=INK)
    handles = [Line2D([], [], color=c, lw=1.8, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=.6) for _, _, c in series]
    handles.append(Line2D([], [], color=CEIL, linestyle=(0, (3, 3)), lw=1.1))
    fig.legend(handles, [lab for _, lab, _ in series] + ["Prompt rewritten (ceiling)"], loc="upper left", bbox_to_anchor=(.045, 1.0), ncol=4, frameon=False, fontsize=9.5,
               handlelength=1.9, columnspacing=1.4, handletextpad=.5, labelcolor=INK)
    stem = FIGS / f"fig_local_lines_{graft}"
    for ext in ("pdf", "svg"): fig.savefig(stem.with_suffix("." + ext), facecolor="white")
    if preview: fig.savefig(preview, dpi=140, facecolor="white")
    plt.close(fig); print(stem)


def print_table(rows, subset="dfs"):
    grids = sorted({g for (_, g, _, _) in rows})
    for met in ("p_new", "top1_new", "gain_nats", "drop_nats", "door_used", "wall_crossed", "solved_new"):
        print(f"\n{met} [{subset}] mean over seeds ± SD")
        for mkey in METHODS:
            cells = [rows.get((mkey, g, subset, met)) for g in grids]
            if not any(cells): continue
            print(f"  {mkey:24s}" + " ".join(f"{c['mean']:7.3f}±{c['sd']:.3f}" if c else "       -     " for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graft", choices=["fixed", "trainable"], default="fixed"); ap.add_argument("--preview", default=None)
    ap.add_argument("--out_dir", default=str(HERE / "results/local/out")); ap.add_argument("--no_tables", action="store_true"); ap.add_argument("--subset", default="dfs")
    a = ap.parse_args()
    sets, runs = load(a.out_dir); rows = summarise(sets, runs)
    print_table(rows, a.subset)
    if not a.no_tables: write_tables(sets, runs, rows)
    selection_report(TABLES / "local_selection.json")
    figure(rows, a.graft, a.preview, a.subset)                   # descriptive per-size figure; the size-averaged main figure is plot_steering_merged.py
    print("tuned configurations (per seed, the two cross-fitting folds):")
    for (mkey, g), ch in sorted(CHOSEN.items()): print(f"  {mkey} {g}x{g}: " + " | ".join("/".join(c.split("_", 1)[1] for c in pair) for pair in ch))
