#%%
import numpy as np
import plotly.graph_objects as go

N = 113
k = 52

x = np.linspace(-10, 130, 5000)

n = np.arange(1, N + 1)[:, None]
theta = 2 * np.pi * n / N
y = (np.cos(theta * k) * np.cos(theta * x) + np.sin(theta * k) * np.sin(theta * x)).sum(axis=0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=1)))
fig.add_hline(y=0, line=dict(color="black", width=0.5))
fig.update_layout(
    title=r"$\sum_{n=1}^{113} \cos\!\left(\frac{2\pi n}{113}(x-52)\right)$",
    xaxis_title="x",
    yaxis_title="f(x)",
    width=1200,
    height=450,
)
fig.write_html("dirichlet_plot.html")

fig.show()

# %%
