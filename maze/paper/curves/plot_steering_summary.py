#!/usr/bin/env python3
"""Single-panel binwise solve rates: individual maze-size curves or range, plus medians.

Superseded for the paper by plot_steering_merged.py; kept as a diagnostic. Writes into figures/.

No artificial zero endpoints: these values are success rates, not counts.
The original-maze controls at zero remain separate from changed-route bins.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import plot_steering as PS


def method_panels(graft, data, series, grids, bins, width):
    from plot_curves_overlay import COLORS
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.13, top=.80, wspace=.12, hspace=.30)
    for ax, (method, label, *_) in zip(axes.flat, series):
        for grid, color in zip(grids, COLORS):
            x = [(lv-.5)*width for lv in bins]
            ax.plot(x, [data[method][grid, lv][0]*100 for lv in bins], color=color, lw=1.4, marker='o', markersize=3)
            ax.plot(0, data[method][grid, 0][0]*100, marker='o', markersize=3.5,
                    markerfacecolor='white', markeredgecolor=color, linestyle='none')
        ax.set_title(label, loc='left', fontsize=10, weight='bold', pad=6)
        ax.set(xlim=(-1.8, 60), ylim=(0, 100))
        ax.set_xticks(range(0, 61, 10)); ax.set_yticks([0, 25, 50, 75, 100])
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.tick_params(labelsize=8, length=3, width=.9)
        ax.tick_params(axis='x', which='minor', length=2, width=.7)
        ax.grid(axis='both', color='#e8e7e2', lw=.6)
        for spine in ax.spines.values(): spine.set_linewidth(1.2)
    fig.text(.095, .98, 'Steering by method and maze size', fontsize=14, weight='bold', va='top')
    fig.legend([Line2D([], [], color=c, lw=2) for c in COLORS],
               [f'{g}×{g}' for g in grids], loc='upper left', bbox_to_anchor=(.08, .925),
               ncol=6, frameon=False, fontsize=9, columnspacing=1.4, handlelength=1.5)
    fig.text(.53, .035, 'Maze difference (%)', ha='center', fontsize=11)
    fig.text(.015, .47, 'Solved (%)', va='center', rotation=90, fontsize=11)
    stem = PS.FIGS / f'fig_steering_methods_{graft}'
    for ext in ('pdf', 'svg'): fig.savefig(stem.with_suffix('.'+ext), facecolor='white')
    plt.close(fig)
    print(stem)


def main(graft='fixed', view='mean'):
    sets, _ = PS.PH.load()
    sets = PS.PH.merge_unrelated_by_distance(sets)
    grids = sorted(sets)
    width = float(next(iter(sets.values()))['width']) * 100
    series = PS.SERIES[:1] + [(f'graft:{graft}', f'{graft.capitalize()} graft', PS.GRAFT, 'o', '-', 1.8)] + PS.SERIES[1:]
    data = {}
    for method, *_ in series:
        runs = PS.load_method(method, sets)
        data[method] = PS.cells(sets, runs, lambda s: (~s['path_same']) | (s['level'] == 0))
    # Use exactly the same supported bins and size weights for every method.
    bins = [lv for lv in range(1, max(int(s['n_buckets']) for s in sets.values()) + 1)
            if all((g, lv) in c and c[g, lv][2] >= 1000 for c in data.values() for g in grids)]
    assert bins and grids == list(range(5, 11))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    if view == 'methods':
        method_panels(graft, data, series, grids, bins, width)
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    fig.subplots_adjust(left=.105, right=.98, bottom=.18, top=.78)
    handles = []
    x = [(lv-.5)*width for lv in bins]
    for method, label, color, marker, style, lw in series:
        values = np.array([[data[method][g, lv][0]*100 for lv in bins] for g in grids])
        if view == 'overlap':
            for row in values:
                ax.plot(x, row, color=color, linestyle=style, lw=.9, alpha=.32, zorder=2)
        elif view == 'range':
            ax.fill_between(x, values.min(axis=0), values.max(axis=0),
                            color=color, alpha=.14, linewidth=0, zorder=2)
        line, = ax.plot(x, np.mean(values, axis=0) if view == "mean" else np.median(values, axis=0), color=color,
                        linestyle=(0, (3, 3)) if method == "oracle" else style,
                        lw=1.0 if method == "oracle" else 1.8, marker=None if method == "oracle" else "o",
                        alpha=.60 if method == "oracle" else 1,
                        markersize=2.5 if method == "oracle" else 4, markeredgecolor="white", markeredgewidth=.6, label=label, zorder=3)
        handles.append(line)
    for method, color, marker in [('oracle', PS.BASE, 'o'), (f'graft:{graft}', PS.GRAFT, 'o')]:
        controls = [data[method][g, 0][0]*100 for g in grids]
        if view == "mean":
            controls = [np.mean(controls)]
        ax.plot(np.zeros(len(controls)), controls, marker=marker, markersize=4,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=.8,
                linestyle='none', zorder=4, alpha=.55)
    ax.set(xlim=(-1.8, 60), ylim=(0, 100), xlabel='Maze difference (%)', ylabel='Solved (%)')
    ax.set_xticks(range(0, 61, 10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(which='major', direction='out', length=4, width=1, labelsize=10)
    ax.tick_params(axis='x', which='minor', direction='out', length=2.5, width=.9)
    ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12)
    for spine in ax.spines.values():
        spine.set_color('black'); spine.set_linewidth(1.4)
    ax.grid(axis='both', color='#e8e7e2', lw=.6)
    heading = (f'{graft.capitalize()} graft vs. steering · mean across maze sizes' if view == 'mean'
               else 'Individual maze sizes + median' if view == 'overlap'
               else 'Range across maze sizes + median')
    fig.text(.105, .975, heading, fontsize=11, weight='bold', va='top')
    fig.legend(handles, [s[1] for s in series], loc='upper left', bbox_to_anchor=(.093, .90),
               ncol=4, frameon=False, fontsize=8.4, handlelength=1.8, columnspacing=1.25, handletextpad=.5)
    stem = PS.FIGS / (f'fig_steering_lines_{graft}' if view == 'mean' else f'fig_steering_ranges_{graft}' if view == 'range' else f'fig_steering_{view}_{graft}')
    for ext in ('pdf', 'svg'):
        fig.savefig(stem.with_suffix('.'+ext), facecolor='white')
    plt.close(fig)
    print(stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graft', choices=['fixed', 'trainable'], default='fixed')
    parser.add_argument('--view', choices=['mean', 'overlap', 'range', 'methods'], default='mean')
    args = parser.parse_args()
    main(args.graft, args.view)
