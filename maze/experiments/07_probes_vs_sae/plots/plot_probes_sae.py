"""Figures for experiment 07: probe steering at the storage site vs SAE features vs injection.
Inputs (results/): probe_acc.json, steer_L0.json, steer_L1.json, steer_L236.json, steer_oracle.json, sae_steer_*.json (any).
Keys: '<kind>|L<layer>|a<alpha>|<add_at>' (probes) and 'sae_<mode>|L<layer>|v<value>|<add_at>|n<nlat>' (SAE)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results"); FIG = os.path.join(os.path.dirname(HERE), "figures"); os.makedirs(FIG, exist_ok=True)
INJ = {"fixed injection\n(channel edit; earlier, full set)": dict(crossed=0.1406, det_used=0.7731, sho_used=0.6097), "trainable injection\n(channel edit; earlier, full set)": dict(crossed=0.0691, det_used=0.8305, sho_used=0.6814)}
S = {}
for f in ("steer_L0.json", "steer_L1.json", "steer_L236.json", "steer_oracle.json"):
    p = os.path.join(RES, f)
    if os.path.exists(p): S.update(json.load(open(p))["results"])
FULL = json.load(open(os.path.join(RES, "steer_full.json")))["results"] if os.path.exists(os.path.join(RES, "steer_full.json")) else {}
S.update(FULL)                                                     # full-set (4,522 edits) numbers replace the subsample for the conditions that have them
SAE = {}
for p in sorted(glob.glob(os.path.join(RES, "sae", "steer_*.json"))):
    tag = os.path.basename(p)[len("steer_"):-5]
    for k, v in json.load(open(p))["results"].items(): SAE[f"{tag}::{k}"] = v
g = lambda r, fam, k: r.get(fam, {}).get(k, np.nan)
score = lambda r: g(r, "detour", "solves_edited") + g(r, "shortcut_open", "solves_edited") - g(r, "detour", "blocked_edge_crossed") - g(r, "zerogain_open", "opened_wall_used")   # rewards faithful edits, penalizes side effects


def best(d, pred):
    c = [(k, r) for k, r in d.items() if pred(k)]
    return max(c, key=lambda kr: score(kr[1])) if c else (None, None)


# ---------------- Fig 1: probe accuracy by site and layer
if os.path.exists(f"{RES}/probe_acc.json"):
    a = json.load(open(f"{RES}/probe_acc.json")); L = a["layers"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for site, col, lab in (("A", "tab:orange", "A token (first cell of the chunk)"), ("conn", "tab:gray", "<--> token"), ("B", "#006d2c", "B token (second cell)"), ("semi", "tab:red", "; token (the paper's site)")):
        ax.plot(L, a[f"{site}_logit"], marker="o", color=col, label=f"{lab}: which connection? (72-way)")
    ax.plot(L, a["path_binary"], marker="^", color="tab:blue", ls="--", label="<PATH_START>: connection present? (mean of 72 binary)")
    ax.axhline(1 / 60, color="k", lw=0.6, ls=":"); ax.set(xlabel="layer (resid_post)", ylabel="validation accuracy", ylim=(0, 1.03), title="Where the connection is readable in this model"); ax.legend(fontsize=7.5, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_probe_accuracy.png", dpi=150)

# ---------------- Fig 2: steering comparison (best of each method family)
rows = [("baseline\n(no intervention)", S.get("none|L0|a0|-"))]
for key, lab in (("prompt|L0|a0|-", "token oracle\n(insert/delete chunk;\nlength changes)"), ("prompt_dup|L0|a0|-", "token oracle, fixed length\n(duplicate/replace chunk)")):
    if key in S: rows.append((lab, S[key]))
k, r = best(S, lambda k: k.startswith("old_style")); rows.append((f"old protocol (hunch #4)\n<PATH_START> probe, L6, α{k.split('|a')[1].split('|')[0]}", r)) if k else None
for pref, lab in (("copy_chunk", "activation patch: copy another\nchunk over f's chunk (remove only)"), ("dir_dim", "probe direction (diff-in-means)\nat the chunk / semicolons"), ("dir_logit", "probe direction (logistic)\nat the chunk / semicolons"), ("mean_patch", "full class-mean patch\nat the chunk / semicolon")):
    k, r = best(S, lambda k, pref=pref: k.startswith(pref))
    if k: rows.append((f"{lab}\n[{k.split('|')[1]}, α{k.split('|a')[1].split('|')[0]}, add {k.split('|')[3]}]", r))
k, r = best(S, lambda k: k.startswith("dir_") and k.endswith("B_all") and "|L0|" in k)
if k: rows.append((f"probe direction FLOODED\ninto every B token (over-steers)\n[{k.split('|')[0]}, {k.split('|')[1]}, α{k.split('|a')[1].split('|')[0]}]", r))
for pref, lab in (("::sae_own", "SAE latent toggle\n(remove at own chunk)"), ("::sae_recon", "SAE latent toggle via\nreconstruction patch")):
    k, r = best(SAE, lambda k, pref=pref: pref in k)
    if k: rows.append((f"{lab}\n[{k.split('::')[0]}, {' '.join(k.split('|')[3:])}]", r))
labels = [l for l, _ in rows] + list(INJ)
def col_(m, fam):
    v = [g(r, fam, m) if r else np.nan for _, r in rows]; return v
crossed = col_("blocked_edge_crossed", "detour") + [v["crossed"] for v in INJ.values()]; det_used = col_("detour_wall_used", "detour") + [v["det_used"] for v in INJ.values()]
sho_used = col_("shortcut_used", "shortcut_open") + [v["sho_used"] for v in INJ.values()]
solves_d = col_("solves_edited", "detour") + [np.nan] * len(INJ); solves_s = col_("solves_edited", "shortcut_open") + [np.nan] * len(INJ); zg = col_("opened_wall_used", "zerogain_open") + [np.nan] * len(INJ); offr = col_("reached_target", "offpath_block") + [np.nan] * len(INJ)
x = np.arange(len(labels)); w = 0.27
fig, ax = plt.subplots(2, 1, figsize=(max(14, 1.9 * len(labels)), 10.5), sharex=True)
ax[0].bar(x - w, crossed, w, color="tab:red", label="detour edit: blocked edge still crossed (↓ good)"); ax[0].bar(x, det_used, w, color="tab:green", label="detour edit: opened repair wall used (↑ good)"); ax[0].bar(x + w, sho_used, w, color="tab:blue", label="shortcut edit: opened shortcut used (↑ good)")
ax[0].set(ylabel="fraction of edits", ylim=(0, 1.02), title="Does the model act on the edited maze? (BFS-verified detour and shortcut edits; full 4,522-edit set where available, else 400 per family)"); ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=0.3)
ax[1].bar(x - w, solves_d, w, color="#238b45", label="detour edit: solves the EDITED maze optimally"); ax[1].bar(x, solves_s, w, color="#08519c", label="shortcut edit: solves the EDITED maze optimally")
ax[1].bar(x + w, zg, w, color="tab:gray", label="zero-gain control: irrelevant opened wall used (should stay ≈ 0)"); ax[1].plot(x, offr, "kv", ms=5, label="off-path block control: target still reached (should stay ≈ 0.98)")
ax[1].set(ylabel="fraction of edits", ylim=(0, 1.02), title="Paper-style efficacy (rollout = a shortest path of the edited maze) and specificity"); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=0.3)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
fig.suptitle("Steering the maze world model: probes at the storage site vs SAE features vs the trained injection channel", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_steering_comparison.png", dpi=150)

# ---------------- Fig 3: layer dependence (probe-based interventions)
layers = sorted({int(k.split("|L")[1].split("|")[0]) for k in S if k.split("|")[0] in ("copy_chunk", "mean_patch", "dir_dim", "dir_logit")})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
series = (("copy_chunk|L{L}|a1|-", "copy another chunk over f's chunk", "tab:orange", "-"), ("mean_patch|L{L}|a1|semi_last", "class-mean patch (add at last ;)", "#006d2c", "-"), ("mean_patch|L{L}|a1|semi_all", "class-mean patch (add at all ;)", "#74c476", "--"),
          ("dir_dim|L{L}|a1|semi_last", "probe direction, diff-in-means α1", "tab:purple", "-"), ("dir_logit|L{L}|a2|semi_all", "probe direction, logistic α2", "tab:pink", "--"))
for j, (metric, fam, lab) in enumerate((("blocked_edge_crossed", "detour", "detour: blocked edge crossed (↓)"), ("detour_wall_used", "detour", "detour: repair wall used (↑)"), ("shortcut_used", "shortcut_open", "shortcut used (↑)"))):
    for pat, slab, colr, ls in series:
        v = [g(S.get(pat.format(L=L), {}), fam, metric) for L in layers]
        if not all(np.isnan(v)): ax[j].plot(layers, v, marker="o", color=colr, ls=ls, label=slab)
    ax[j].axhline(g(S.get("none|L0|a0|-", {}), fam, metric), color="k", ls=":", lw=1, label="baseline")
    ax[j].set(xlabel="intervention layer (resid_post)", title=lab, ylim=(0, 1.02)); ax[j].grid(alpha=0.3)
ax[0].legend(fontsize=7)
fig.suptitle("Interventions at the storage site, by layer", fontsize=11); fig.tight_layout(); fig.savefig(f"{FIG}/fig3_layer_sweep.png", dpi=150)

# ---------------- summary text
lines = []
for name, d in (("probe-based", S), ("SAE", SAE)):
    lines.append(f"== {name}")
    for k, r in d.items():
        lines.append(f"{k:60s} detour crossed {g(r, 'detour', 'blocked_edge_crossed'):.3f} used {g(r, 'detour', 'detour_wall_used'):.3f} solves {g(r, 'detour', 'solves_edited'):.3f} reach {g(r, 'detour', 'reached_target'):.3f} | shortcut used {g(r, 'shortcut_open', 'shortcut_used'):.3f} solves {g(r, 'shortcut_open', 'solves_edited'):.3f} | offpath crossed {g(r, 'offpath_block', 'blocked_edge_crossed'):.3f} reach {g(r, 'offpath_block', 'reached_target'):.3f} | zerogain used {g(r, 'zerogain_open', 'opened_wall_used'):.3f}")
open(f"{FIG}/summary.txt", "w").write("\n".join(lines)); print("\n".join(lines)); print("figures ->", FIG)
