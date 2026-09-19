"""Render figures from pod results (icl_oracle/results/) -> icl_oracle/figures/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = f"{HERE}/results", f"{HERE}/figures"
os.makedirs(FIG, exist_ok=True)
N = 16

meta = json.load(open(f"{RES}/replicate.json"))
h_star, words, coords = meta["h_star"], meta["words"], [tuple(c) for c in meta["coords"]]
d = np.load(f"{RES}/replicate.npz")
mass, reps, energy, ckpts = d["mass"], d["reps"], d["energy"], d["ckpts"]

A = np.zeros((N, N))
for i, (r, c) in enumerate(coords):
    for j, (r2, c2) in enumerate(coords):
        if abs(r - r2) + abs(c - c2) == 1:
            A[i, j] = 1

# ---- Fig 1: rule-following accuracy vs context (+ memorization baselines)
t = np.arange(1, mass.shape[1] + 1)
m, sd = mass.mean(0), mass.std(0)
k = 25
sm = np.convolve(m, np.ones(k) / k, mode="valid")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t[k - 1:], sm, lw=2, color="tab:blue", label="gemma-2-9b (4 seeds, smoothed)")
ax.fill_between(t, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), alpha=0.12, color="tab:blue")
ax.plot(t, 1 - ((N - 1) / N) ** t, "--", color="gray", label="1-shot memorization (approx)")
ax.plot(t, 1 - ((N - 1) / N) ** t - t * (1 / N) * ((N - 1) / N) ** (t - 1),
        ":", color="gray", label="2-shot memorization (approx)")
ax.set(xscale="log", xlabel="context length (tokens)", ylabel="rule-following prob. mass",
       title="In-context graph tracing: 4x4 grid, gemma-2-9b", ylim=(0, 1))
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/fig1_accuracy.png", dpi=150)

# ---- Fig 2: PCA snapshots at h* across context sizes
show = [0, 2, 3, 5] if len(ckpts) == 6 else list(range(len(ckpts)))
fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.4))
for ax, ci in zip(axes, show):
    H = np.nan_to_num(reps[:, ci, h_star].astype(np.float64)).mean(0)  # avg over seeds? no: per-seed PCA differs
    H = np.nan_to_num(reps[0, ci, h_star].astype(np.float64))
    Hc = H - H.mean(0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    P = Hc @ Vt[:2].T
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                ax.plot(P[[i, j], 0], P[[i, j], 1], "-", color="lightgray", lw=1, zorder=1)
    cols = [(r / 3, 0.2, c / 3) for r, c in coords]
    ax.scatter(P[:, 0], P[:, 1], c=cols, s=60, zorder=2)
    for i, w in enumerate(words):
        ax.annotate(w, P[i], fontsize=6, ha="center", va="bottom")
    ax.set_title(f"{ckpts[ci]} tokens", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"PCA of mean node reps (hidden idx {h_star}, seed 0) — grid emerges with context", y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_pca.png", dpi=150, bbox_inches="tight")

# ---- Fig 3: Dirichlet energy vs context per layer
fig, ax = plt.subplots(figsize=(6, 4))
n_hs = energy.shape[2]
layer_list = sorted({*np.linspace(4, n_hs - 2, 8).astype(int).tolist(), h_star})
for l in layer_list:
    e = energy[:, :, l].mean(0)
    ax.plot(ckpts, e, marker="o", lw=2 if l == h_star else 1,
            label=f"hs {l}" + (" (h*)" if l == h_star else ""))
ax.set(xscale="log", xlabel="context length (tokens)", ylabel="normalized Dirichlet energy",
       title="Energy of grid structure in representations")
ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(f"{FIG}/fig3_energy.png", dpi=150)

# ---- Fig 4: injection A — dense accuracy-vs-context curves, injected vs baseline
if os.path.exists(f"{RES}/fast_curve.json"):
    fc = json.load(open(f"{RES}/fast_curve.json"))
    styles = {"none": ("k", "baseline (no injection)"),
              "extracted_b2": ("tab:green", "extracted coords, β=2"),
              "shuffled_b2": ("tab:red", "shuffled assignment (control)"),
              "arbitrary_b2": ("tab:purple", "arbitrary subspace coords")}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    k = 25
    for cname, (col, lab) in styles.items():
        if cname not in fc["curves"]:
            continue
        M = np.array(fc["curves"][cname])           # [seeds, T-1]
        m, sd = M.mean(0), M.std(0)
        t = np.arange(1, M.shape[1] + 1)
        sm = np.convolve(m, np.ones(k) / k, mode="valid")
        ax.plot(t[k - 1:], sm, lw=2, color=col, label=lab)
        ax.fill_between(t, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), alpha=0.08, color=col)
    ax.set(xscale="log", xlabel="context length (tokens)", ylabel="rule-following prob. mass",
           title=f"Injection at every node token (layer {fc['L']}) vs baseline — same walks",
           ylim=(0, 1))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/fig4_inject_fast.png", dpi=150)
else:
    fj = json.load(open(f"{RES}/inject_fast.json"))
    lengths, res = fj["lengths"], fj["res"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    styles = {"none": ("k", "-", "o"), "extracted": ("tab:green", "-", "s"),
              "shuffled": ("tab:red", "--", "x"), "arbitrary": ("tab:purple", "-.", "^")}
    for key, vals in sorted(res.items()):
        cname = key.split("|")[0]
        col, ls, mk = styles[cname]
        ax.plot(lengths, vals, ls, color=col, marker=mk, alpha=0.85, label=key, lw=1.5)
    ax.set(xscale="log", xlabel="context length (tokens)", ylabel="rule-following prob. at query",
           title="Injection A: structure coords injected into short contexts", ylim=(0, 1))
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(f"{FIG}/fig4_inject_fast.png", dpi=150)

# ---- Fig 5: injection B — phantom neighbor
rows = json.load(open(f"{RES}/inject_phantom.json"))
conds = ["null", "move", "phantom_b1.0", "phantom_b2.0"]
mets = ["P_true_nbrs", "P_k", "P_k_nbrs"]
vals = {c: [np.mean([r[m] for r in rows if r["cond"] == c]) for m in mets] for c in conds}
x = np.arange(len(mets)); w = 0.2
fig, ax = plt.subplots(figsize=(6.5, 4))
for i, c in enumerate(conds):
    ax.bar(x + (i - 1.5) * w, vals[c], w, label=c)
ax.set_xticks(x); ax.set_xticklabels(["P(true neighbors of i)", "P(fake neighbor k)", "P(neighbors of k)"])
ax.set(ylabel="probability mass", title="Injection B: world-state edit at query token")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/fig5_phantom.png", dpi=150)

print("figures written to", FIG)
