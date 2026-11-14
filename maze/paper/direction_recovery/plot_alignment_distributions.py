"""Reconstruct individual raw-graft alignments from archived fit checkpoints.

Hungarian distributions contain one distinct learned direction per graft edge.
The companion all-directions plot includes every validation-active SAE atom,
using its nearest graft direction (no one-to-one constraint).
"""
import csv
from figure_paths import figure_path
import json
import os
from pathlib import Path
import tempfile
from collections import Counter
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'maze-mpl'))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from methods import SAE, unit
from maze_distance import free_mask

ROOT = Path(__file__).resolve().parent
METHODS = ['relu', 'batchtopk', 'matryoshka', 'mp', 'difference_means',
           'ridge', 'logistic', 'generative_ols']
LABELS = ['ReLU SAE', 'BatchTopK SAE', 'Matryoshka SAE', 'Matching Pursuit SAE',
          'Difference of means', 'Ridge probe', 'Logistic probe', 'Generative OLS']


@torch.no_grad()
def recover():
    torch.set_num_threads(4)
    rows, checks = [], []
    for item in json.loads((ROOT / 'input_manifest.json').read_text()):
        if item['condition'] not in ('fixed', 'trainable'):
            continue
        n = item['grid']
        tag = f"g{n}-s{item['seed']}-{item['condition']}"
        root = ROOT / 'results' / f'g{n}'
        run = root / tag
        source = Path(item['source'])
        inj = torch.load(source / 'injector.final.pt', map_location='cpu', weights_only=True)
        w = inj['W_inj'].float() * float(inj.get('scale', 1))
        mask = free_mask(n)
        gt = w[mask]
        edge_ids = np.flatnonzero(mask)
        x = None
        linear = torch.load(run / 'linear-injection.pt', map_location='cpu', weights_only=True)['readouts']
        for method in METHODS:
            saved = json.loads((run / f'injection-{method}-geometry.json').read_text())
            if method in linear:
                dictionary = linear[method]
                ids = torch.arange(len(dictionary))
            else:
                fit = torch.load(run / f'injection-{method}.pt', map_location='cpu', weights_only=True)
                net = SAE(kind=method, k=fit['k'])
                net.load_state_dict(fit['state'])
                net.eval()
                if saved['training'][-1]['dead'] == 0:
                    ids = torch.arange(len(net.D))
                else:
                    if x is None:
                        cache = root / f'train-g{n}-n20000-dedup.npz'
                        if not cache.exists():
                            cache = root / f'train-g{n}-n20000.npz'
                        data = np.load(cache)
                        conn = torch.from_numpy(data['conn'][16000:]).float() * 2 - 1
                        tokens = data['tokens'][16000:]
                        model = torch.load(source / 'model.final.pt', map_location='cpu', weights_only=False)['state_dict']
                        x = (model['embed.weight'][torch.from_numpy(tokens[:, -2]).long()]
                             + model['pos_embed.weight'][tokens.shape[1] - 2] + conn @ w)
                        del model
                    activity = sum(net.encode((b-fit['mean'])/fit['scale']).abs().sum(0)
                                   for b in x.split(512))
                    ids = torch.where(activity > 0)[0]
                assert 1024-len(ids) == saved['training'][-1]['dead'], (tag, method, len(ids))
                dictionary = net.D[ids] * fit['scale']
            sim = (unit(gt) @ unit(dictionary).T).abs().numpy()
            ri, ci = linear_sum_assignment(-sim)
            assert len(ri) == len(gt), (tag, method, 'Incomplete matching')
            error = abs(float(sim[ri, ci].mean()) - saved['raw']['assignment_mean'])
            assert error < 2e-5, (tag, method, error)
            checks.append({'run': tag, 'method': method, 'mean_error': error,
                           'active_directions': len(ids), 'matched_directions': len(ri)})
            for r, c in zip(ri, ci):
                rows.append(dict(run=tag, method=method, metric='matched',
                                 direction=int(ids[c]), edge=int(edge_ids[r]), cosine=float(sim[r, c])))
            for c, r in enumerate(sim.argmax(axis=0)):
                rows.append(dict(run=tag, method=method, metric='all_directions_nearest',
                                 direction=int(ids[c]), edge=int(edge_ids[r]), cosine=float(sim[r, c])))
        print(tag, 'verified', flush=True)
    assert len(checks) == 24 * 8
    with (ROOT / 'alignment_distribution_values.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'alignment_distribution_validation.json').write_text(json.dumps(checks, indent=2))
    return rows


def weighted_samples(rows, metric, method):
    subset = [r for r in rows if r['method'] == method and r['metric'] == metric]
    values = np.clip([float(r['cosine']) for r in subset], 0, 1)
    fit_keys = [(r['run'], str(r.get('fit_seed', 'original'))) for r in subset]
    fit_sizes = Counter(fit_keys)
    fits_per_run = Counter(run for run, seed in fit_sizes)
    weights = np.array([1 / len(fits_per_run) / fits_per_run[key[0]] / fit_sizes[key]
                        for key in fit_keys])
    assert len(values) and np.isclose(weights.sum(), 1)
    return values, weights, len(fit_sizes)


def plot_density(rows, metric, name, bandwidth=.015):
    """Reflected Gaussian KDE on [0,1], using a fine histogram convolution.

    Reflection preserves probability at the cosine boundaries. A single fixed
    absolute bandwidth makes the smoothing comparable across methods. Means
    are calculated from the samples, never from the smoothed curves.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True)
    bins = np.linspace(0, 1, 2001)
    dx = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    summaries = []
    for ax, method, label in zip(axes.flat, METHODS, LABELS):
        # Equal model weight, then equal fit weight within each model, then
        # equal direction weight within a fit. Bigger mazes have more edges
        # but do not dominate the aggregate solely for that reason.
        values, weights, fits = weighted_samples(rows, metric, method)
        counts, _ = np.histogram(values, bins, weights=weights)
        density = gaussian_filter1d(counts.astype(float), bandwidth/dx, mode='reflect') / dx
        assert np.isclose(density.sum()*dx, 1)
        xx = np.r_[0, centers, 1]
        yy = np.r_[density[0], density, density[-1]]
        ax.fill_between(xx, yy, color='#477da8', alpha=.25)
        ax.plot(xx, yy, color='#477da8', lw=1.8)
        mean = float(np.dot(values, weights))
        ax.axvline(mean, color='#a43d2c', ls='--', lw=1.5)
        ax.text(.04, .90, f'Mean {mean:.3f}', transform=ax.transAxes,
                fontsize=9, color='#a43d2c',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1})
        ax.set_title(label, loc='left', fontsize=11)
        ax.set(xlim=(0, 1), ylim=(0, None), xticks=[0, .5, 1])
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        order = np.argsort(values)
        median = float(values[order][np.searchsorted(np.cumsum(weights[order]), .5)])
        summaries.append(dict(method=method, metric=metric, count=len(values), fits=fits,
                              mean=mean, median=median,
                              fraction_above_08=float(weights[values >= .8].sum()),
                              bandwidth=bandwidth, weighting='equal_model_then_fit_then_direction'))
    fig.suptitle('How closely do learned directions align with the graft?', fontsize=16, y=.98)
    fig.supxlabel('Absolute cosine similarity to original graft direction', y=.105, fontsize=11)
    fig.supylabel('Probability density', x=.015, fontsize=11)
    caption = ('One-to-one optimal matching; unmatched SAE latents excluded.' if metric == 'matched' else
               'All active learned directions, each compared to its nearest graft direction.')
    fit_counts = {r['fits'] for r in summaries}
    fit_text = (f"{next(iter(fit_counts))} fits per method" if len(fit_counts) == 1 else
                f"{min(fit_counts)}–{max(fit_counts)} fits per method")
    grids = sorted({int(r['run'].split('-')[0][1:]) for r in rows})
    size_text = f'Grids {grids[0]}–{grids[-1]}' if len(grids) > 1 else f'Grid {grids[0]}'
    fig.text(.07, .025, f'{caption} {fit_text}; dashed lines mark means.\n'
             f'{size_text}; equal model and fit weights. Area over an interval gives its fraction.\n'
             f'Gaussian bandwidth = {bandwidth:g}; boundary reflection. Vertical scales differ.',
             fontsize=9, color='#555555', linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.985, top=.875, bottom=.20, hspace=.45, wspace=.32)
    for ext in ('svg', 'pdf'):
        fig.savefig(figure_path(name, ext), dpi=200, bbox_inches='tight', facecolor='white')
    (ROOT / f'{name}_summary.json').write_text(json.dumps(summaries, indent=2))
    plt.close(fig)


if __name__ == '__main__':
    cache = ROOT / 'alignment_distribution_values.csv'
    rows = list(csv.DictReader(cache.open())) if cache.exists() else recover()
    expanded = ROOT / 'expanded_alignment' / 'alignment_distribution_values.csv'
    if expanded.exists():
        rows += list(csv.DictReader(expanded.open()))
    for grid in range(5, 11):
        grid_rows = [r for r in rows if r['run'].startswith(f'g{grid}-')]
        plot_density(grid_rows, 'matched', f'alignment_density_g{grid}')
