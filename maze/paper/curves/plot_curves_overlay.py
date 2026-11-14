#!/usr/bin/env python3
"""Overlay maze sizes by model condition using existing campaign CSVs.

Run: .venv/bin/python maze/paper/curves/plot_curves_overlay.py
Curves are unsmoothed seed means; uncertainty bands are omitted.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGS, TABLES = ROOT / 'figures', ROOT / 'tables'
FIGS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
# Ordered light-to-dark blue; even the smallest maze remains visible on white.
COLORS = ['#8AB6CC', '#619DC0', '#3D80B3', '#22629E', '#144581', '#10285F']
METRICS = [
    ('train_loss', 'Training loss'),
    ('p_sp', 'Shortest-path\ntoken probability'),
    ('p_valid', 'Other valid\ntoken probability'),
    ('p_invalid', 'Invalid token\nprobability'),
    ('solved', 'Maze solved\nwithout rule breaks'),
]


def main(graft='fixed'):
    df = pd.read_csv(TABLES / 'all_metrics.csv')
    loss = pd.read_csv(TABLES / 'train_loss_by_ckpt.csv')
    grids = sorted(df.grid.unique())
    assert len(grids) <= len(COLORS)
    series = [
        ('None', 'Baseline Model', 'base', 'true'),
        ('Original Maze', 'Grafted Feature Model', graft, 'true'),
        ('Edited Maze', 'Grafted Feature Model', graft, 'phantom'),
    ]
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': '#777777', 'axes.linewidth': .7,
    })
    fig, axes = plt.subplots(5, 3, figsize=(10.8, 11.6), dpi=240, sharex=True, sharey='row')
    fig.subplots_adjust(left=.125, right=.975, bottom=.08, top=.825, wspace=.16, hspace=.18)
    for col, (features, model, cond, world) in enumerate(series):
        pos = axes[0, col].get_position()
        fig.text((pos.x0 + pos.x1) / 2, .87, f'Grafted Features: {features}' if cond != 'base' else 'Baseline Model',
                 ha='center', fontsize=10, weight='bold')
        for row, (key, label) in enumerate(METRICS):
            ax = axes[row, col]
            if key == 'train_loss' and world == 'phantom':
                ax.set_axis_off()
                ax.text(.5, .5, 'Same training run as\nOriginal Maze', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='#777777', linespacing=1.5)
                continue
            for grid, color in zip(grids, COLORS):
                if key == 'train_loss':
                    sub = loss[(loss.grid == grid) & (loss.cond == cond)]
                else:
                    sub = df[(df.grid == grid) & (df.cond == cond) & (df.world == world) & (~df.final)]
                pivot = sub.pivot_table(index='iteration', columns='seed', values=key)
                if pivot.empty:
                    continue
                ax.plot(pivot.index.to_numpy(), pivot.mean(axis=1),
                        color=color, linewidth=1.55, solid_capstyle='round', zorder=3)
            ax.set_xscale('linear')
            ax.set_xlim(0, 8000)
            ax.set_xticks([0, 2000, 4000, 6000, 8000])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(color='#e8e7e2', linewidth=.55, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2.5, width=.7, labelsize=8, colors='#555555')
            if row == 4:
                ax.set_xlabel('Training step', fontsize=9, labelpad=6)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, labelpad=10)
            if key == 'train_loss':
                ax.set_yscale('linear')
                ax.set_ylim(0, 5.2)
                ax.yaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                ax.set_ylim(0, 1.025)
                ax.set_yticks([0, .25, .5, .75, 1])
    fig.text(.125, .975, 'Learning across maze sizes', fontsize=16, weight='bold', va='top')
    fig.text(.125, .946, f'{graft.capitalize()} grafts · means across seeds · darker blue = larger maze',
             fontsize=9.5, color='#666666', va='top')
    fig.legend([Line2D([], [], color=c, lw=2.5) for c in COLORS[:len(grids)]],
               [f'{g} × {g}' for g in grids], loc='center left', bbox_to_anchor=(.118, .91),
               ncol=6, frameon=False, handlelength=2.5, columnspacing=2.2, fontsize=9)
    fig.text(.125, .018, 'Uncertainty bands omitted for readability. All curves use the same scales within each row.',
             fontsize=8, color='#777777')
    stem = FIGS / f'fig_curves_{graft}_overlay'
    for ext in ('pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + ext), facecolor='white')
    plt.close(fig)
    print(stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graft', choices=['fixed', 'trainable'], default='fixed')
    main(parser.parse_args().graft)
