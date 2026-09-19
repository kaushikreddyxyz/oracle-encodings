#!/usr/bin/env python3
"""Final-checkpoint performance versus maze size; means and sample SD across seeds.

Solve rate of the grafted models comes from the Hamming-distance-stratified set
(results/distance/, 3,000 mazes per size: grafted maze = prompt maze, or a DFS maze at a
controlled distance from it). Token probabilities were only measured on the 2,000-maze
campaign set (all_metrics.csv), where the edited maze is the 2-wall + 2-door phantom edit.
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS = [
    ('solved', 'Maze solved without rule breaks'),
    ('p_sp', 'Shortest-path token probability'),
    ('p_valid', 'Other valid token probability'),
    ('p_invalid', 'Invalid token probability'),
]
# distance strata for the solve-rate panel: (world key, lo, hi] as a fraction of the 2n(n-1) free wall features
STRATA = [('dist_0-10', 0.0, 0.10), ('dist_10-30', 0.10, 0.30), ('dist_30-60', 0.30, 0.60)]


def stratified_solved(graft):
    """Per (grid, seed) solve rate of the grafted model on the distance-stratified set.
    Worlds: dist_0 (grafted maze = prompt maze), the STRATA, dist_fresh (independent DFS maze)."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / f'results/distance/out/g*__{graft}__*.npz'))):
        run = np.load(f)
        grid = int(run['grid_n'])
        dset = np.load(ROOT / f'results/distance/sets/g{grid}.npz')
        assert str(run['set_sha']) == str(dset['sha']), f
        level, frac = dset['level'], dset['dist'] / int(dset['n_free'])
        variant = (level > 0) & (level <= int(dset['n_buckets']))
        masks = {'dist_0': level == 0, 'dist_fresh': level == int(dset['n_buckets']) + 1}
        masks.update({key: variant & (frac > lo) & (frac <= hi) for key, lo, hi in STRATA})
        seed = int(str(run['seed_tag']).split('-s')[1])
        for world, mask in masks.items():
            rows.append(dict(cond=graft, world=world, grid=grid, seed=seed,
                             solved=run['solved_graft'][mask].mean(), n_mazes=int(mask.sum())))
    return pd.DataFrame(rows)


def main(graft='fixed'):
    data = pd.read_csv(ROOT / 'all_metrics.csv')
    data = data[data.final].copy()
    conditions = [('base', 'true'), (graft, 'true'), (graft, 'phantom')]
    data = data[[pair in conditions for pair in zip(data.cond, data.world)]]
    assert not data.duplicated(['grid', 'seed', 'cond', 'world']).any()
    assert data.iteration.nunique() == 1, 'Final checkpoints differ across runs'
    grids = sorted(data.grid.unique())
    strat = stratified_solved(graft)
    assert sorted(strat.grid.unique()) == grids
    assert not strat.duplicated(['grid', 'seed', 'world']).any()
    data = pd.concat([data, strat], ignore_index=True)
    counts = data.groupby(['cond', 'world', 'grid']).seed.nunique()
    assert counts.nunique() == 1, 'Seed counts differ across groups'
    nseeds = int(counts.iloc[0])
    step = int(data.iteration.dropna().iloc[0])
    summary = data.groupby(['cond', 'world', 'grid'])[[m[0] for m in METRICS]].agg(['mean', 'std', 'count'])
    summary.to_csv(ROOT / f'final_results_{graft}_summary.csv')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': '#888888', 'axes.linewidth': .7,
    })
    fig = plt.figure(figsize=(8.4, 6.3), dpi=240)
    layout = fig.add_gridspec(2, 3, height_ratios=[1.3, 1],
                              left=.08, right=.98, top=.80, bottom=.135,
                              hspace=.70, wspace=.38)
    axes = [fig.add_subplot(layout[0, :])]
    axes.extend(fig.add_subplot(layout[1, col]) for col in range(3))
    base = ('base', 'true', '#2a7a57', 'o', 'Baseline Model')
    solved_styles = [
        base,
        (graft, 'dist_0', '#000c7a', 's', 'Grafted Features: Original Maze'),
        (graft, 'dist_0-10', '#f2ab85', 'D', 'Edited Maze, 0–10% of walls differ'),
        (graft, 'dist_10-30', '#d96332', '^', 'Edited Maze, 10–30%'),
        (graft, 'dist_30-60', '#8a3510', 'v', 'Edited Maze, 30–60%'),
    ]
    token_styles = [
        base,
        (graft, 'true', '#000c7a', 's', 'Grafted Features: Original Maze'),
        (graft, 'phantom', '#d96332', 'D', 'Grafted Features: Edited Maze (2 phantom walls + 2 phantom doors)'),
    ]

    def handles(styles):
        return [Line2D([], [], color=c, marker=m, markersize=5, markerfacecolor='white' if m == 's' else c, linestyle='-')
                for _, _, c, m, _ in styles]

    for ax, (metric, title) in zip(axes, METRICS):
        upper = 0
        for cond, world, color, marker, label in (solved_styles if metric == 'solved' else token_styles):
            means = np.array([summary.loc[(cond, world, grid), (metric, 'mean')] for grid in grids])
            sds = np.array([summary.loc[(cond, world, grid), (metric, 'std')] for grid in grids])
            assert np.isfinite(means).all() and np.isfinite(sds).all()
            upper = max(upper, (means+sds).max())
            ax.errorbar(grids, means, yerr=sds, fmt=marker+'-', color=color,
                        markersize=4, markeredgewidth=.8,
                        markerfacecolor='white' if marker == 's' else color,
                        linewidth=1.35, elinewidth=.8, capsize=2, zorder=3)
        titles = {'solved': 'Maze solved without rule breaks',
                  'p_sp': 'Shortest-path token', 'p_valid': 'Other valid token',
                  'p_invalid': 'Invalid token'}
        ax.set_title(titles[metric], loc='left', fontsize=10, weight='bold', pad=8 if metric != 'solved' else 22)
        ax.set_xticks(grids, [f'{g}×{g}' for g in grids])
        ax.tick_params(length=2.5, labelsize=7.5, colors='#555555')
        ax.set_xlim(min(grids)-.12, max(grids)+.12)
        ax.set_ylim(0, 1.025 if metric in ('solved', 'p_sp') else upper*1.12)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0 if metric in ('solved', 'p_sp', 'p_valid') else 1))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if metric in ('solved', 'p_sp'):
            ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis='y', color='#e8e7e2', linewidth=.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(handles(solved_styles), [s[4] for s in solved_styles], loc='lower left', bbox_to_anchor=(-.01, 1.0),
                   ncol=5, frameon=False, handlelength=1, columnspacing=1.2, handletextpad=.4, fontsize=7.3)
    fig.legend(handles(token_styles), [s[4] for s in token_styles], loc='center left', bbox_to_anchor=(.075, .452),
               ncol=3, frameon=False, handlelength=1, columnspacing=1.8, fontsize=7.3)
    fig.text(.5, .078, 'Maze size', ha='center', fontsize=9)
    fig.text(.085, .975, 'Final-checkpoint performance', fontsize=15, weight='bold', va='top')
    fig.text(.085, .935, f'{graft.capitalize()} grafts · step {step:,} · mean ± SD across {nseeds} seeds',
             fontsize=9, color='#666666', va='top')
    fig.text(.085, .012, 'All scales linear; probability axes start at zero. Error bars show SD. Solve rate: edited mazes are DFS mazes at a set Hamming distance\n'
             'from the prompt maze over the 2n(n−1) free wall features, 3,000 mazes per size. Token probabilities: 2,000 mazes per size.',
             fontsize=7.2, color='#666666', va='bottom')
    stem = ROOT / f'fig_final_{graft}_lines'
    for ext in ('pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + ext), facecolor='white')
    if PREVIEW:
        fig.savefig(PREVIEW, facecolor='white')
    plt.close(fig)
    print(stem)


PREVIEW = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graft', choices=['fixed', 'trainable'], default='fixed')
    parser.add_argument('--preview', default=None, help='optional extra raster path for a quick look (not part of the figure set)')
    args = parser.parse_args()
    PREVIEW = args.preview
    main(args.graft)
