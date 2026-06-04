# %%
"""Visualize the geometry of sinusoidal positional encoding vectors in 3D (via PCA)."""

from itertools import combinations

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Render inline in VS Code / Jupyter cells; fall back to browser for plain `python script.py`.
try:
    get_ipython()  # type: ignore[name-defined]  # noqa: F821
    pio.renderers.default = "plotly_mimetype+notebook+vscode"
except NameError:
    pio.renderers.default = "browser"

# %%
DIM = 16384
CONTEXT_LEN = 500
BASE = 10000.0
N_PCS = 5  # number of principal components to compute and project onto

# %%
def sinusoidal_pe(context_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Standard 'Attention is All You Need' sinusoidal positional encoding.
    Returns array of shape (context_len, dim). Odd `dim` is supported by
    computing dim+1 channels and truncating to `dim`."""
    d_even = dim if dim % 2 == 0 else dim + 1
    pos = np.arange(context_len, dtype=np.float64)[:, None]
    i = np.arange(d_even // 2, dtype=np.float64)[None, :]
    inv_freq = base ** (-2.0 * i / d_even)
    angles = pos * inv_freq
    pe = np.empty((context_len, d_even), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe[:, :dim]


def channel_label(channel_idx: int, dim: int, base: float) -> str:
    """Human-readable label: 'sin(t·ω)' or 'cos(t·ω)' with ω formatted."""
    d_even = dim if dim % 2 == 0 else dim + 1
    i_pair = channel_idx // 2
    omega = base ** (-2 * i_pair / d_even)
    phi = "sin" if channel_idx % 2 == 0 else "cos"
    return f"{phi}(t·{omega:.4g})"


def take_channels(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Take the first k channels of x (no rotation). Returns (channels (N,k), var_frac (k,))."""
    kk = min(k, x.shape[1])
    channels = x[:, :kk]
    if kk < k:
        channels = np.pad(channels, ((0, 0), (0, k - kk)))
    total_var = x.var(axis=0, ddof=0).sum()
    var_frac = np.zeros(k)
    var_frac[:kk] = channels.var(axis=0, ddof=0) / total_var if total_var > 0 else 0.0
    return channels, var_frac


def theoretical_channel_density(channel_idx: int, dim: int, base: float,
                                 context_len: int, grid: np.ndarray,
                                 bandwidth: float) -> np.ndarray:
    """Density of sinusoidal-PE channel `channel_idx` on `grid`.
    Computed via dense numerical sampling of t in [0, context_len) (i.e. as if
    t were continuous), KDE-smoothed at `bandwidth` so it's comparable to the
    empirical KDE at the same bandwidth."""
    d_even = dim if dim % 2 == 0 else dim + 1
    i_pair = channel_idx // 2
    omega = base ** (-2 * i_pair / d_even)
    phi_fn = np.sin if channel_idx % 2 == 0 else np.cos
    t = np.linspace(0, context_len, 20_000, endpoint=False)
    y = phi_fn(t * omega)
    diff = (grid[:, None] - y[None, :]) / bandwidth
    return np.exp(-0.5 * diff ** 2).sum(axis=1) / (len(y) * bandwidth * np.sqrt(2 * np.pi))

# %%
pe = sinusoidal_pe(CONTEXT_LEN, DIM, BASE)
print(f"PE shape: {pe.shape}")
print(f"PE norm (per position) mean={np.linalg.norm(pe, axis=1).mean():.3f} "
      f"std={np.linalg.norm(pe, axis=1).std():.3f}")

# Use raw channels (no PCA rotation) so axes correspond to the actual sin/cos basis.
proj, evr = take_channels(pe, N_PCS)
ch_labels = [channel_label(k, DIM, BASE) for k in range(N_PCS)]
print(f"Channels: {ch_labels}")
print(f"Variance fraction (per channel): {evr.round(4).tolist()}")
print(f"Cumulative: {evr.sum():.4f}")

# %%
# Raw encoding plotted as-is (no channel selection, no PCA). Only works for DIM in {1, 2, 3}.
positions = np.arange(CONTEXT_LEN)
all_ch_labels = [channel_label(k, DIM, BASE) for k in range(DIM)]
if DIM == 1:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=pe[:, 0], y=np.zeros(CONTEXT_LEN),
        mode="markers",
        marker=dict(size=5, color=positions, colorscale="Viridis",
                    colorbar=dict(title="position"), showscale=True, opacity=0.7),
        hovertemplate=f"pos=%{{marker.color}}<br>{all_ch_labels[0]}=%{{x:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Sinusoidal PE as-is — dim=1, context_len={CONTEXT_LEN}",
        xaxis_title=all_ch_labels[0], yaxis=dict(visible=False),
        width=1100, height=300,
    )
    fig.show()
elif DIM == 2:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=pe[:, 0], y=pe[:, 1],
        mode="markers",
        marker=dict(size=4, color=positions, colorscale="Viridis",
                    colorbar=dict(title="position"), showscale=True),
        hovertemplate=(f"pos=%{{marker.color}}<br>{all_ch_labels[0]}=%{{x:.3f}}"
                       f"<br>{all_ch_labels[1]}=%{{y:.3f}}<extra></extra>"),
    ))
    fig.update_xaxes(title_text=all_ch_labels[0], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text=all_ch_labels[1])
    fig.update_layout(
        title=f"Sinusoidal PE as-is — dim=2, context_len={CONTEXT_LEN}",
        width=800, height=800,
    )
    fig.show()
elif DIM == 3:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pe[:, 0], y=pe[:, 1], z=pe[:, 2],
        mode="lines+markers",
        marker=dict(size=3, color=positions, colorscale="Viridis",
                    colorbar=dict(title="position"), showscale=True),
        line=dict(color="rgba(120,120,120,0.4)", width=2),
        hovertemplate=(f"pos=%{{marker.color}}<br>{all_ch_labels[0]}=%{{x:.3f}}"
                       f"<br>{all_ch_labels[1]}=%{{y:.3f}}<br>{all_ch_labels[2]}=%{{z:.3f}}<extra></extra>"),
    ))
    fig.update_layout(
        title=f"Sinusoidal PE as-is — dim=3, context_len={CONTEXT_LEN}",
        scene=dict(
            xaxis_title=all_ch_labels[0],
            yaxis_title=all_ch_labels[1],
            zaxis_title=all_ch_labels[2],
            aspectmode="data",
        ),
        width=900, height=750,
    )
    fig.show()
else:
    print(f"dim={DIM} is not directly plottable (need DIM in {{1, 2, 3}}); "
          f"see the channel-pair / channel-triple plots below.")

# %%
triples = list(combinations(range(N_PCS), 3))
n_3d_cols = min(2, len(triples))
n_3d_rows = (len(triples) + n_3d_cols - 1) // n_3d_cols
specs = [[{"type": "scene"} if r * n_3d_cols + c < len(triples) else None
          for c in range(n_3d_cols)] for r in range(n_3d_rows)]
fig = make_subplots(
    rows=n_3d_rows, cols=n_3d_cols, specs=specs,
    subplot_titles=[f"Ch{a+1}, Ch{b+1}, Ch{c+1}" for a, b, c in triples],
    horizontal_spacing=0.04, vertical_spacing=0.08,
)
for idx, (a, b, c) in enumerate(triples):
    row = idx // n_3d_cols + 1
    col = idx % n_3d_cols + 1
    is_last = idx == len(triples) - 1
    fig.add_trace(go.Scatter3d(
        x=proj[:, a], y=proj[:, b], z=proj[:, c],
        mode="lines+markers",
        marker=dict(size=2, color=positions, colorscale="Viridis",
                    showscale=is_last,
                    colorbar=dict(title="position") if is_last else None),
        line=dict(color="rgba(120,120,120,0.4)", width=2),
        hovertemplate=(f"pos=%{{marker.color}}<br>{ch_labels[a]}=%{{x:.3f}}"
                       f"<br>{ch_labels[b]}=%{{y:.3f}}<br>{ch_labels[c]}=%{{z:.3f}}<extra></extra>"),
    ), row=row, col=col)
    scene_name = f"scene{idx + 1}" if idx > 0 else "scene"
    fig.update_layout({scene_name: dict(
        xaxis_title=f"Ch{a+1}: {ch_labels[a]}",
        yaxis_title=f"Ch{b+1}: {ch_labels[b]}",
        zaxis_title=f"Ch{c+1}: {ch_labels[c]}",
        aspectmode="data",
    )})
fig.update_layout(
    title=(f"Sinusoidal PE geometry (basis-aligned, no rotation) — "
           f"dim={DIM}, context_len={CONTEXT_LEN}<br>"
           f"<sup>all C({N_PCS},3)={len(triples)} channel triples</sup>"),
    width=1200, height=600 * n_3d_rows, showlegend=False,
)
fig.show()

# %%
pairs = list(combinations(range(N_PCS), 2))
n_pair_cols = 3
n_pair_rows = (len(pairs) + n_pair_cols - 1) // n_pair_cols
fig = make_subplots(rows=n_pair_rows, cols=n_pair_cols,
                    subplot_titles=[f"Ch{a+1} ({ch_labels[a]}) vs Ch{b+1} ({ch_labels[b]})"
                                    for a, b in pairs],
                    horizontal_spacing=0.08, vertical_spacing=0.14)
for idx, (a, b) in enumerate(pairs):
    row = idx // n_pair_cols + 1
    col = idx % n_pair_cols + 1
    is_last = idx == len(pairs) - 1
    axis_idx = idx + 1
    fig.add_trace(go.Scattergl(
        x=proj[:, a], y=proj[:, b],
        mode="markers",
        marker=dict(size=4, color=positions, colorscale="Viridis",
                    showscale=is_last,
                    colorbar=dict(title="position") if is_last else None),
        hovertemplate=f"pos=%{{marker.color}}<br>{ch_labels[a]}=%{{x:.3f}}<br>{ch_labels[b]}=%{{y:.3f}}<extra></extra>",
    ), row=row, col=col)
    fig.update_xaxes(title_text=f"Ch{a+1}", row=row, col=col,
                     scaleanchor=f"y{axis_idx}", scaleratio=1)
    fig.update_yaxes(title_text=f"Ch{b+1}", row=row, col=col)
fig.update_layout(
    title=f"2D projections (basis-aligned) — dim={DIM}, context_len={CONTEXT_LEN}",
    width=1300, height=380 * n_pair_rows, showlegend=False,
)
fig.show()

# %%
def gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Cheap gaussian KDE — no scipy dep. Silverman's rule by default."""
    n = samples.shape[0]
    if bandwidth is None:
        bandwidth = 1.06 * samples.std(ddof=1) * n ** (-1 / 5)
    diff = (grid[:, None] - samples[None, :]) / bandwidth
    return np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))

xmax = float(np.abs(proj[:, :N_PCS]).max()) * 1.05
grid = np.linspace(-xmax, xmax, 400)
# Per-channel Silverman bandwidth — used for BOTH empirical and theoretical curves,
# so any deviation between them reflects finite-N sampling, not bandwidth choice.
bws = [max(1.06 * proj[:, k].std(ddof=1) * len(proj) ** (-1 / 5), 1e-4)
       for k in range(N_PCS)]
empirical_kdes = [gaussian_kde_1d(proj[:, k], grid, bandwidth=bws[k]) for k in range(N_PCS)]
theoretical_kdes = [theoretical_channel_density(k, DIM, BASE, CONTEXT_LEN, grid, bws[k])
                    for k in range(N_PCS)]
kde_ymax = max(max(k.max() for k in empirical_kdes),
               max(k.max() for k in theoretical_kdes)) * 1.1

fig = make_subplots(rows=N_PCS, cols=1,
                    subplot_titles=[f"Ch{k+1}: {ch_labels[k]} ({evr[k]*100:.1f}% var)"
                                    for k in range(N_PCS)],
                    vertical_spacing=0.22 / max(N_PCS - 1, 1) * 2)
for k in range(N_PCS):
    is_last = k == N_PCS - 1
    # Empirical KDE (from the actual integer-position samples)
    fig.add_trace(go.Scatter(
        x=grid, y=empirical_kdes[k],
        mode="lines", name="empirical (KDE)",
        line=dict(color="rgba(60,60,60,0.9)", width=1.5),
        fill="tozeroy", fillcolor="rgba(120,120,180,0.25)",
        showlegend=(k == 0),
        hovertemplate=f"{ch_labels[k]}=%{{x:.3f}}<br>density=%{{y:.4f}}<extra></extra>",
    ), row=k + 1, col=1)
    # Theoretical density: density of phi(t·ω) for t uniform on [0, context_len)
    fig.add_trace(go.Scatter(
        x=grid, y=theoretical_kdes[k],
        mode="lines", name="theoretical",
        line=dict(color="rgba(200,30,30,0.9)", width=2, dash="dash"),
        showlegend=(k == 0),
        hovertemplate=f"{ch_labels[k]}=%{{x:.3f}}<br>theory=%{{y:.4f}}<extra></extra>",
    ), row=k + 1, col=1)
    # Rug strip below the axis
    fig.add_trace(go.Scattergl(
        x=proj[:, k], y=np.full_like(proj[:, k], -kde_ymax * 0.08),
        mode="markers",
        marker=dict(size=5, color=positions, colorscale="Viridis",
                    showscale=is_last,
                    colorbar=dict(title="position", len=0.9) if is_last else None,
                    opacity=0.7),
        showlegend=False,
        hovertemplate=f"pos=%{{marker.color}}<br>{ch_labels[k]}=%{{x:.4f}}<extra></extra>",
    ), row=k + 1, col=1)
    fig.update_xaxes(title_text=f"Ch{k+1}", range=[-xmax, xmax],
                     zeroline=True, zerolinewidth=1, zerolinecolor="black",
                     row=k + 1, col=1)
    fig.update_yaxes(visible=False, range=[-kde_ymax * 0.15, kde_ymax],
                     row=k + 1, col=1)
fig.update_layout(
    title=(f"1D projection density per channel — dim={DIM}, context_len={CONTEXT_LEN}<br>"
           f"<sup>dashed red = analytical density of φ(t·ω) for t uniform on [0,T)</sup>"),
    width=1100, height=240 * N_PCS,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.show()

# %%
