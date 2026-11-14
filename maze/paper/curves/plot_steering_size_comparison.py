"""Per-size companions to the merged figure, with matching methods and seed SD.

Writes figures/fig_steering_{local,global}_{graft}_by_size.{pdf,svg}.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import plot_steering_merged as M


def main(graft='trainable'):
    x, global_data = M.global_curves(graft, by_size=True)
    local_data = M.local_points(graft, by_size=True)
    series = [('graft', f'{graft.capitalize()} graft', M.PS.GRAFT),
              ('vec', 'Steering vectors', M.PS.BASE), ('sae', 'SAE features', M.PS.SAE_C)]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for kind in ('local', 'global'):
        fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.3), sharex=True, sharey=True)
        fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.78, wspace=.12, hspace=.32)
        for i, ax in enumerate(axes.flat):
            ax.set_title(f'{i+5}×{i+5}', loc='left', fontsize=11, weight='bold', pad=6)
            ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
            M.style(ax)
            if kind == 'global':
                mu = global_data['oracle'][0][i, 1:]
                ax.plot(x, mu, color='#dc2625', ls=(0, (4, 2)), lw=1.2, zorder=4)
                for key, _, color in series:
                    mu, sd = [a[i, 1:] for a in global_data[key]]
                    ax.fill_between(x, mu-sd, mu+sd, color=color, alpha=.13, lw=0)
                    ax.plot(x, mu, color=color, lw=1.5, marker='o', markersize=2.5,
                            markeredgecolor='white', markeredgewidth=.4, clip_on=False)
                ax.set_xlim(0, 60); ax.set_xticks(range(0, 61, 10))
                ax.xaxis.set_minor_locator(MultipleLocator(5))
                ax.grid(axis='x', color=M.GRID, lw=.5)
                ax.tick_params(axis='x', labelsize=8)
            else:
                xs = np.array([x for x, _, _ in M.LOCAL_METRICS])
                ax.hlines(local_data['oracle'][0][i], xs-.34, xs+.34,
                          color='#dc2625', ls=(0, (4, 2)), lw=1.2, zorder=5)
                for j, (key, _, color) in enumerate(series):
                    mu, sd = [a[i] for a in local_data[key]]
                    ax.bar(xs+(j-1)*.23, mu, width=.20, color=color,
                           yerr=sd, error_kw={'ecolor': '#333333', 'elinewidth': .65, 'capsize': 0}, zorder=3)
                ax.set_xlim(-.45, 4.1)
                ax.set_xticks(xs, ['Choose\nnew move', 'Use\nnew door', 'Choose\nold move', 'Cross\nnew wall'])
                ax.tick_params(axis='x', length=0, labelsize=8)
                ax.axvline(1.825, color=M.GRID, lw=.6)
        title = 'One edit · from decision cell' if kind == 'local' else 'Many edits · from maze start'
        fig.text(.07, .975, title, fontsize=14, weight='bold', va='top')
        handles = [Line2D([], [], color=c, marker='o', lw=0, markersize=4) for _, _, c in series]
        handles.append(Line2D([], [], color='#dc2625', ls=(0, (4, 2)), lw=1.2))
        fig.legend(handles, [s[1] for s in series]+['Prompt rewritten (ceiling)'], loc='upper left',
                   bbox_to_anchor=(.06, .925), ncol=4, fontsize=9, frameon=False,
                   columnspacing=1.7, handlelength=1.9)
        fig.text(.012, .48, 'Share of edits (%)', rotation=90, va='center', fontsize=11)
        if kind == 'global':
            fig.text(.52, .072, 'Maze difference (%)', ha='center', fontsize=11)
            footer = 'Solves the Edited Maze (higher is better)'
        else:
            footer = 'New move / door: Acknowledges Edit (higher is better)    ·    Old move / new wall: Ignores Edit (lower is better)'
        fig.text(.52, .025, footer, ha='center', fontsize=8.5, color='#555555')
        stem = M.PS.FIGS / f'fig_steering_{kind}_{graft}_by_size'
        for ext in ('pdf','svg'): fig.savefig(stem.with_suffix('.'+ext), facecolor='white')
        plt.close(fig)
        print(stem)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graft', choices=['fixed','trainable'], default='trainable')
    main(p.parse_args().graft)
