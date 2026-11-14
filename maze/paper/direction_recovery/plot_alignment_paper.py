"""Paper styling for the all-directions alignment violin figure.

Statistical calculations match plot_alignment_violins; only presentation changes.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from figure_paths import figure_path
from plot_alignment_distributions import ROOT, METHODS, LABELS, weighted_samples


def plot(rows, metric, name):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.5, 3.65))
    ax = fig.add_axes([.29, .18, .685, .67])
    positions = [0, 1, 2, 3, 4.5, 5.5, 6.5, 7.5]
    ax.set(ylim=(8.1, -.6), xlim=(0, 1.025))
    summaries = []
    for method, label, y in zip(METHODS, LABELS, positions):
        color = '#000c7a' if y < 4 else '#2a7a57'
        fill = '#dddff0' if y < 4 else '#d6e5de'
        values, weights, fits = weighted_samples(rows, metric, method)
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        quantiles = values[order][np.searchsorted(cumulative, [.1, .25, .75, .9])]
        mean = float(np.dot(values, weights))
        fraction = float(weights[values >= .8].sum())
        edges = np.linspace(0, 1, 2001)
        dx = edges[1]-edges[0]
        centers = (edges[:-1]+edges[1:])/2
        mass, _ = np.histogram(values, edges, weights=weights)
        density = gaussian_filter1d(mass, .015/dx, mode='reflect')/dx
        assert np.isclose(density.sum()*dx, 1)
        xx = np.r_[values.min(), centers[(centers > values.min()) & (centers < values.max())], values.max()]
        yy = np.interp(xx, centers, density)
        widths = .34*yy/yy.max()
        ax.fill_between(xx, y-widths, y+widths, facecolor=fill, edgecolor=color, lw=1, zorder=2)
        ax.plot(quantiles[[1, 2]], [y, y], color=color, lw=2.6, solid_capstyle='round', zorder=3)
        ax.scatter(mean, y, s=29, color=color, edgecolor='white', linewidth=.8, zorder=4)
        ax.text(-.035, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=9.5, color='black')
        summaries.append({'method': method, 'metric': metric, 'fits': fits, 'mean': mean,
                          'q10_q25_q75_q90': quantiles.tolist(), 'fraction_at_least_08': fraction,
                          'weighting': 'equal_model_then_fit_then_direction', 'violin_bandwidth': .015,
                          'violin_width_normalization': 'per_method_peak'})
    ax.set_yticks([])
    ax.set_xticks([0, .2, .4, .6, .8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.tick_params(axis='x', labelsize=10, length=3.5, width=.9, pad=5)
    ax.set_xlabel('Alignment with closest graft direction (|cosine|)', fontsize=10, labelpad=6)
    ax.grid(axis='x', color='#e8e7e2', lw=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'left', 'right'): ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('black'); ax.spines['bottom'].set_linewidth(1.2)
    fig.text(.03, .965, 'Graft-direction alignment', fontsize=13, weight='bold', va='top')
    legend = [Line2D([], [], color='black', lw=2.6, label='IQR'),
              Line2D([], [], color='black', marker='o', lw=0, markersize=5, label='Mean')]
    fig.legend(handles=legend, loc='upper right', bbox_to_anchor=(.985, .99),
               ncol=2, frameon=False, fontsize=9, handlelength=1.5, columnspacing=1.4)
    for ext in ('pdf', 'svg'):
        fig.savefig(figure_path(name, ext), facecolor='white')
    (ROOT / f'{name}_summary.json').write_text(json.dumps(summaries, indent=2))
    plt.close(fig)
