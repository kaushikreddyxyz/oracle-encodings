"""Horizontal weighted violins, means, and fractions reaching cosine 0.8."""
import csv
from figure_paths import figure_path
import json
from plot_alignment_distributions import ROOT, METHODS, LABELS, weighted_samples
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

def plot(rows, metric, name):
    if name == 'alignment_violins_all_directions':
        from plot_alignment_paper import plot as paper_plot
        return paper_plot(rows, metric, name)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(10.8, 6.4))
    ax = fig.add_axes([0.22, 0.25, 0.55, 0.56])
    stats = fig.add_axes([0.8, 0.25, 0.18, 0.56], sharey=ax)
    stats.set_xlim(0, 1)
    stats.axis('off')
    summaries = []
    for i, (method, label) in enumerate(zip(METHODS, LABELS)):
        values, weights, fits = weighted_samples(rows, metric, method)
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        quantiles = values[order][np.searchsorted(cumulative, [0.1, 0.25, 0.75, 0.9])]
        mean = float(np.dot(values, weights))
        fraction = float(weights[values >= 0.8].sum())
        assert np.all(np.diff(quantiles) >= 0) and 0 <= fraction <= 1.000001
        edges = np.linspace(0, 1, 2001)
        dx = edges[1] - edges[0]
        centers = (edges[:-1] + edges[1:]) / 2
        mass, _ = np.histogram(values, edges, weights=weights)
        density = gaussian_filter1d(mass, 0.015 / dx, mode='reflect') / dx
        assert np.isclose(density.sum() * dx, 1)
        xx = np.r_[values.min(), centers[(centers > values.min()) & (centers < values.max())], values.max()]
        yy = np.interp(xx, centers, density)
        widths = 0.34 * yy / yy.max()
        ax.fill_between(xx, i - widths, i + widths, facecolor='#b3cbdc', edgecolor='#477da8', linewidth=1.1)
        ax.plot(quantiles[[1, 2]], [i, i], color='#202e38', lw=2.2)
        ax.scatter([mean], [i], s=42, color='#202e38', edgecolors='white', linewidths=0.8, zorder=3)
        stats.text(0.15, i, f'{mean:.3f}', va='center', ha='center', fontsize=11)
        stats.text(0.77, i, f'{100 * fraction:.1f}%', va='center', ha='center', fontsize=11)
        summaries.append({'method': method, 'metric': metric, 'fits': fits, 'mean': mean, 'q10_q25_q75_q90': quantiles.tolist(), 'fraction_at_least_08': fraction, 'weighting': 'equal_model_then_fit_then_direction', 'violin_bandwidth': 0.015, 'violin_width_normalization': 'per_method_peak'})
    ax.set(yticks=range(len(METHODS)), yticklabels=LABELS, ylim=(7.6, -0.6), xlim=(0, 1.025), xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.tick_params(axis='y', length=0, pad=12)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Absolute cosine similarity to original graft direction', labelpad=12)
    ax.grid(axis='x', color='#e6e9ec', lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(3.5, color='#e6e9ec', lw=1)
    stats.text(0.15, 1.035, 'Mean', transform=stats.transAxes, ha='center', fontsize=10, color='#555555')
    stats.text(0.77, 1.035, 'Fraction ≥ 0.8', transform=stats.transAxes, ha='center', fontsize=10, color='#555555')
    fig.suptitle('How well do the methods recover graft directions?', fontsize=17, y=0.97)
    legend = [Patch(facecolor='#b3cbdc', edgecolor='#477da8', label='Distribution'), Line2D([0], [0], color='#202e38', lw=2.2, label='IQR'), Line2D([0], [0], color='#202e38', marker='o', lw=0, markersize=6, label='Mean')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.56, 0.91), ncol=3, frameon=False, fontsize=10, handlelength=2.5, columnspacing=2.5)
    counts = {s['fits'] for s in summaries}
    assert len(counts) == 1
    caption = 'Optimal one-to-one matching; unmatched SAE latents excluded.' if metric == 'matched' else 'All active learned directions; each compared to its closest graft direction.'
    detail = 'Violin widths scaled separately; Gaussian bandwidth 0.015, trimmed to observed range.'
    fig.text(0.22, 0.035, f'{caption}\n{counts.pop()} fits per method across grids 5–10; equal model and fit weights.\n' + detail, fontsize=9, color='#555555', linespacing=1.5)
    for ext in ('svg', 'pdf'):
        fig.savefig(figure_path(name, ext), dpi=200, bbox_inches='tight', facecolor='white')
    (ROOT / f'{name}_summary.json').write_text(json.dumps(summaries, indent=2))
    plt.close(fig)
if __name__ == '__main__':
    rows = list(csv.DictReader((ROOT / 'alignment_distribution_values.csv').open()))
    expanded = ROOT / 'expanded_alignment' / 'alignment_distribution_values.csv'
    if expanded.exists():
        rows += list(csv.DictReader(expanded.open()))
    plot(rows, 'matched', 'alignment_violins')
    plot(rows, 'all_directions_nearest', 'alignment_violins_all_directions')
