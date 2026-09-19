"""Three story figures for the QK-port result (from results_qk/port3.json + port2.json)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES, FIG = f"{HERE}/results_qk", f"{HERE}/figures_qk"
p3 = json.load(open(f"{RES}/port3.json")); D, O, rows, A = p3["dense"], p3["override"], p3["unseen"], p3["attn"]
COL = {"none": "k", "rand26_b4": "tab:purple", "own_b2": "tab:blue", "late_b4": "#006d2c", "lateanti_b4": "tab:red", "latenon_b4": "tab:gray"}
LAB = {"none": "no injection", "rand26_b4": "random plane\n(same layer, same strength)", "own_b2": "model's own plane\n(calibrated per context)",
       "late_b4": "QK-port\n(weights only, no calibration)", "lateanti_b4": "anti-port\n(same heads, negative directions)", "latenon_b4": "port from 4 random\nlate-layer heads"}


def sel(src, c, new):
    return [x for k, v in src.items() if k.endswith("|" + c) and (not k.startswith("set4")) == new for x in v]


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v))


# ============================================================ Figure 1: headline — control on unseen words
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
chans = ["rand26_b4", "own_b2", "late_b4"]
rs = [r for r in rows if r["set"] != 4]                              # new word sets only (n = 27 items)
base_tx = np.mean([r["P_nbrs_X"] for r in rs if r["cond"] == "none"]); base_fy = np.mean([r["P_nbrs_Y"] for r in rs if r["cond"] == "none"])
x = np.arange(len(chans)); w = 0.19
for i, c in enumerate(chans):
    tx = [r["P_nbrs_X"] for r in rs if r["cond"] == f"{c}|trueX"]; ty = [r["P_nbrs_Y"] for r in rs if r["cond"] == f"{c}|trueX"]
    fx = [r["P_nbrs_X"] for r in rs if r["cond"] == f"{c}|claimY"]; fy = [r["P_nbrs_Y"] for r in rs if r["cond"] == f"{c}|claimY"]
    ax[0].bar(x[i] - 1.5 * w, np.mean(tx), w, yerr=sem(tx), color="tab:green", capsize=2, label="write TRUE coordinate → mass on true neighbors" if i == 0 else None)
    ax[0].bar(x[i] - 0.5 * w, np.mean(ty), w, yerr=sem(ty), color="tab:red", alpha=0.35, capsize=2, label="write TRUE coordinate → mass on fake location's neighbors" if i == 0 else None)
    ax[0].bar(x[i] + 0.5 * w, np.mean(fx), w, yerr=sem(fx), color="tab:green", alpha=0.35, hatch="//", capsize=2, label="write FAKE coordinate → mass on true neighbors" if i == 0 else None)
    ax[0].bar(x[i] + 1.5 * w, np.mean(fy), w, yerr=sem(fy), color="tab:red", hatch="//", capsize=2, label="write FAKE coordinate → mass on fake location's neighbors" if i == 0 else None)
ax[0].axhline(base_tx, color="tab:green", ls="--", lw=1); ax[0].axhline(base_fy, color="tab:red", ls="--", lw=1)
ax[0].text(len(chans) - 0.55, base_tx + 0.004, "no injection", fontsize=7, color="gray", ha="right")
ax[0].set_xticks(x); ax[0].set_xticklabels([LAB[c] for c in chans], fontsize=8.5)
ax[0].set(ylabel="next-token probability mass", title="Placing a node the model has NEVER seen\n(new word sets, n = 27, hs 26)")
ax[0].legend(fontsize=7, loc="upper left")
# override
n0 = np.array(sel(D, "none", True)); ch = np.array(sel(O, "none_injmap", True))
for i, c in enumerate(chans):
    os_ = sel(O, c, True); tm = np.array([o["true_map"] for o in os_])[:, 200:600].mean(1); im = np.array([o["injected_map"] for o in os_])[:, 200:600].mean(1)
    ax[1].bar(x[i] - w / 2, tm.mean(), w, yerr=sem(tm), color="k", alpha=0.75, capsize=3, label="mass on TRUE grid neighbors (what the tokens say)" if i == 0 else None)
    ax[1].bar(x[i] + w / 2, im.mean(), w, yerr=sem(im), color="tab:orange", alpha=0.9, capsize=3, label="mass on INJECTED-map neighbors (what the channel says)" if i == 0 else None)
ax[1].axhline(n0[:, 200:600].mean(), color="k", ls="--", lw=1); ax[1].axhline(ch[:, 200:600].mean(), color="tab:orange", ls="--", lw=1)
ax[1].text(len(chans) - 0.55, ch[:, 200:600].mean() + 0.01, "chance for injected map", fontsize=7, color="gray", ha="right")
ax[1].text(len(chans) - 0.55, n0[:, 200:600].mean() + 0.01, "no injection", fontsize=7, color="gray", ha="right")
ax[1].set_xticks(x); ax[1].set_xticklabels([LAB[c] for c in chans], fontsize=8.5)
ax[1].set(ylabel="next-token probability mass (tokens 200–600)", title="Writing a WRONG map: does the model follow the tokens or the channel?\n(new word sets, 9 contexts)")
ax[1].legend(fontsize=7.5, loc="upper right")
fig.suptitle("A weight-defined port gives causal control on words it was never calibrated on — stronger than the calibrated plane", fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/story1_control.png", dpi=160, bbox_inches="tight")

# ============================================================ Figure 2: what the effect is — controls (Δ vs none, new word sets)
fig, ax = plt.subplots(figsize=(11, 4.6))
chans = ["rand26_b4", "lateanti_b4", "late_b4", "latenon_b4", "own_b2"]
nn = {(r["set"], r["seed"], r["X"]): r for r in rs if r["cond"] == "none"}
metrics = []
for c in chans:
    a = np.array(sel(D, c, True)); du = a[:, 100:300].mean(1) - n0[:, 100:300].mean(1)
    os_ = sel(O, c, True); im = np.array([o["injected_map"] for o in os_])[:, 200:600].mean(1) - ch[:, 200:600].mean(1)
    dx = [r["P_nbrs_X"] - nn[(r["set"], r["seed"], r["X"])]["P_nbrs_X"] for r in rs if r["cond"] == f"{c}|trueX"]
    dy = [r["P_nbrs_Y"] - nn[(r["set"], r["seed"], r["X"])]["P_nbrs_Y"] for r in rs if r["cond"] == f"{c}|claimY"]
    metrics.append([(du.mean(), sem(du)), (im.mean(), sem(im)), (np.mean(dx + dy), sem(dx + dy))])
x = np.arange(len(chans)); w = 0.26
for j, (mlab, col) in enumerate((("faster learning (Δ mass, tokens 100–300)", "#74c476"), ("override (injected-map mass − chance)", "tab:orange"), ("unseen-node control (mean Δ, true & fake)", "tab:green"))):
    ax.bar(x + (j - 1) * w, [m[j][0] for m in metrics], w, yerr=[m[j][1] for m in metrics], color=col, capsize=2.5, label=mlab)
ax.axhline(0, color="k", lw=0.7); ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in chans], fontsize=8.5)
ax.set(ylabel="Δ probability mass vs no injection", title="It is the sign of the directions that matters — not their strength, and not which heads (new word sets)")
ax.legend(fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(f"{FIG}/story2_controls.png", dpi=160)

# ============================================================ Figure 3: mechanism — learning curves + reader-head attention
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
K = 25
for c in ["none", "rand26_b4", "own_b2", "late_b4", "lateanti_b4"]:
    M = np.array(sel(D, c, False)); m = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
    t = np.arange(K, len(m) + 1); sm = np.convolve(m, np.ones(K) / K, "valid"); ss = np.convolve(se, np.ones(K) / K, "valid")
    ax[0].plot(t, sm, color=COL[c], lw=2, label=LAB[c].replace("\n", " ")); ax[0].fill_between(t, sm - ss, sm + ss, color=COL[c], alpha=0.1)
ax[0].set(xscale="log", ylim=(0.2, 1), xlabel="context length (tokens)", ylabel="rule-following probability mass", title="Learning curves (word set 4, 6 seeds)")
ax[0].legend(fontsize=7.5)
readers = list(next(iter(A.values()))[0].keys())
chans = ["none", "rand26_b4", "own_b2", "late_b4", "lateanti_b4", "latenon_b4"]
vals = [[np.mean([a[r] for a in sel(A, c, False)]) for r in readers] for c in chans]
x = np.arange(len(readers)); w = 0.8 / len(chans)
for i, c in enumerate(chans):
    ax[1].bar(x + (i - len(chans) / 2 + 0.5) * w, vals[i], w, color=COL[c], label=LAB[c].replace("\n", " "))
ax[1].axhline(1, color="k", ls=":", lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(readers)
ax[1].set(ylabel="attention to neighbor occurrences (× chance)", title="Reader heads: attention to neighbor occurrences")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/story3_mechanism.png", dpi=160)
print("ok")
