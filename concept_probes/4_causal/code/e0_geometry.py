"""Stage 6.1 E0 — direction geometry (task.md §6.1.2; CPU, $0, run first).

Computes, from local Stage-5 probe npz only:
  1. Cross-layer same-concept cosine (12x12 per concept) in THREE spaces:
     standardized (w), raw-read (w/sigma_l), raw-write (sigma_l*w) — each
     layer's own natstats. Fleet: adjacent-layer median + decay vs |l-l'|.
  2. Within-family within-layer C x C cosine (standardized), circulant test
     for cyclic families (cycle-adjacent vs cycle-distant sibling cosine).
  3. Arm angles per (concept, layer): cos(ridge,DoM), cos(ridge,LDA),
     cos(DoM,LDA), cos(sigma*w, w/sigma) (std-arm vs grad-arm steering).
  4. Random baseline: |cos| among saved rand_dirs and rand-vs-ridge
     (expect ~half-normal with sigma = 1/sqrt(2304) ~= 0.021).

Outputs: out/e0/{all_crosslayer,within_family,arm_angles,rand_baseline}.npz,
out/e0/E0_SUMMARY.md, figures under figures/e0/. Deterministic, CPU-only.

  python e0_geometry.py --probes-root ../../2_probes/probes \
      --probe-cards ../../3_validation/artifacts/probe_cards.json \
      --out ../out/e0 --figures ../figures/e0
"""
from __future__ import annotations
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

LAYERS = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25]
FAMILIES = ["months", "weekdays", "seasons", "color_wheel", "directions",
            "moon_phases", "continents", "location_type", "costliness",
            "physical_size", "lovingness", "duration", "harmfulness",
            "glorptitude"]          # glorptitude = nonsense CONTROL
CONTROL = "glorptitude"

# cycle orders from 3_validation/code/plots.py CYCLES; npz class names use
# underscores (moon_phases), so normalize spaces -> underscores here.
CYCLES = {
    "months": ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"],
    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday",
                 "saturday", "sunday"],
    "seasons": ["spring", "summer", "autumn", "winter"],
    "color_wheel": ["red", "red-orange", "orange", "yellow-orange", "yellow",
                    "yellow-green", "green", "blue-green", "blue",
                    "blue-violet", "violet", "red-violet"],
    "directions": ["north", "northeast", "east", "southeast", "south",
                   "southwest", "west", "northwest"],
    "moon_phases": ["new_moon", "waxing_crescent", "first_quarter",
                    "waxing_gibbous", "full_moon", "waning_gibbous",
                    "last_quarter", "waning_crescent"],
}

COS_CMAP, DIVERGING = "RdBu_r", dict(vmin=-1, vmax=1)
C_BLUE, C_RED, C_ORANGE, C_GREEN = "#1965b0", "#dc050c", "#f1932d", "#4eb265"


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


def load_family(root: Path, fam: str) -> dict:
    """-> {'classes': [str], 'per_layer': {L: {'ridge','dom','lda' [C,2304] unit,
    'rand' [20,2304] unit, 'mu','sigma' [2304]}}} — all fp64, standardized space."""
    out = {"per_layer": {}}
    for L in LAYERS:
        z = np.load(root / fam / f"probes_l{L}.npz")
        classes = [str(c) for c in z["classes"]]
        out.setdefault("classes", classes)
        assert out["classes"] == classes, f"{fam} class order differs at L{L}"
        li = z["chosen_lambda_ridge"].astype(int)                  # [C]
        ridge = z["W_ridge"][li, np.arange(len(classes))]          # [C,2304]
        out["per_layer"][L] = dict(
            ridge=unit(ridge.astype(np.float64)),
            dom=unit(z["W_dom"].astype(np.float64)),
            lda=unit(z["W_lda"].astype(np.float64)),
            rand=unit(z["rand_dirs"].astype(np.float64)),
            mu=z["nat_mean"].astype(np.float64),
            sigma=z["nat_std"].astype(np.float64))
    return out


def heartbeat(log: Path, msg: str):
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} e0_geometry {msg}\n")


# ---------------------------------------------------------------- analysis 1
def crosslayer(data, concepts, out, figdir, rep_concepts, log):
    nL = len(LAYERS)
    mats = {sp: np.zeros((len(concepts), nL, nL)) for sp in
            ("std", "rawread", "rawwrite")}
    for i, (fam, cls) in enumerate(tqdm(concepts, desc="E0.1 cross-layer")):
        ci = data[fam]["classes"].index(cls)
        V = {sp: np.zeros((nL, 2304)) for sp in mats}
        for j, L in enumerate(LAYERS):
            pl = data[fam]["per_layer"][L]
            w, sig = pl["ridge"][ci], pl["sigma"]
            V["std"][j], V["rawread"][j], V["rawwrite"][j] = \
                w, unit(w / sig), unit(sig * w)
        for sp in mats:
            mats[sp][i] = V[sp] @ V[sp].T
        heartbeat(log, f"{fam}.{cls} {i+1}/{len(concepts)}")

    fams = np.array([f for f, _ in concepts])
    np.savez(out / "all_crosslayer.npz",
             cos_std=mats["std"], cos_rawread=mats["rawread"],
             cos_rawwrite=mats["rawwrite"], layers=np.array(LAYERS),
             families=fams, classes=np.array([c for _, c in concepts]))

    # fleet stats: adjacent-layer (consecutive in LAYERS) + decay vs |l-l'|
    iu = np.triu_indices(nL, 1)
    dl = np.abs(np.subtract.outer(LAYERS, LAYERS))[iu]
    stats = {}
    for sp, M in mats.items():
        adj = np.array([np.diagonal(m, 1).mean() for m in M])     # per concept
        per_d = {int(d): np.median([m[iu][dl == d].mean() for m in M])
                 for d in np.unique(dl)}
        stats[sp] = dict(adj_per_concept=adj, decay=per_d)

    # figures: decay curves + rep-concept heatmap grid (std space)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sp, col in zip(("std", "rawread", "rawwrite"), (C_BLUE, C_RED, C_ORANGE)):
        d = stats[sp]["decay"]
        axes[0].plot(list(d), list(d.values()), "o-", ms=4, color=col,
                     label=f"{sp} (adj med {np.median(stats[sp]['adj_per_concept']):.3f})")
    axes[0].set(xlabel="|l - l'|", ylabel="median cross-layer cosine",
                title="same-concept cosine decay vs layer distance")
    axes[0].axhline(0, color="gray", lw=0.6); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, lw=0.4)
    ctl = fams == CONTROL
    for sp, col in zip(("std", "rawread", "rawwrite"), (C_BLUE, C_RED, C_ORANGE)):
        a = stats[sp]["adj_per_concept"]
        axes[1].hist(a[~ctl], bins=24, histtype="step", color=col, label=sp)
        for v in a[ctl]:
            axes[1].axvline(v, color=col, ls=":", lw=1.5)
    axes[1].set(xlabel="mean adjacent-layer cosine (per concept)", ylabel="n",
                title=f"adjacent-layer cosine (dotted = {CONTROL} control)")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "crosslayer_decay.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9.5))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (fam, cls) in zip(axes.flat, rep_concepts):
        i = concepts.index((fam, cls))
        ax.set_axis_on()
        im = ax.imshow(mats["std"][i], cmap=COS_CMAP, **DIVERGING)
        tag = " (CONTROL)" if fam == CONTROL else ""
        ax.set_title(f"{fam}.{cls}{tag}", fontsize=8)
        ax.set_xticks(range(nL)); ax.set_xticklabels(LAYERS, fontsize=5)
        ax.set_yticks(range(nL)); ax.set_yticklabels(LAYERS, fontsize=5)
    fig.colorbar(im, ax=axes, fraction=0.02, label="cosine (standardized)")
    fig.suptitle("E0.1 cross-layer same-concept cosine, one concept per family")
    fig.savefig(figdir / "crosslayer_grid.png", dpi=150); plt.close(fig)
    return stats


# ---------------------------------------------------------------- analysis 2
def within_family(data, out, figdir, modal_layer):
    payload, circ = {}, {}
    for fam in FAMILIES:
        classes = data[fam]["classes"]
        if len(classes) < 2:
            continue
        C = len(classes)
        cos = np.zeros((len(LAYERS), C, C))
        for j, L in enumerate(LAYERS):
            W = data[fam]["per_layer"][L]["ridge"]
            cos[j] = W @ W.T
        payload[f"{fam}__cos"] = cos
        payload[f"{fam}__classes"] = np.array(classes)

        mL = modal_layer.get(fam, LAYERS[len(LAYERS) // 2])
        jm = LAYERS.index(mL)
        cyc = CYCLES.get(fam)
        if cyc:
            order = [classes.index(c) for c in cyc]
            dist = np.abs(np.subtract.outer(np.arange(C), np.arange(C)))
            dist = np.minimum(dist, C - dist)                      # cycle dist
            com = cos[:, order][:, :, order]                       # cycle-ordered
            adj = np.array([m[dist == 1].mean() for m in com])
            far = np.array([m[dist >= 2].mean() for m in com])
            by_d = np.array([[m[dist == d].mean() for d in range(1, dist.max() + 1)]
                             for m in com])                        # [12, maxd]
            circ[fam] = dict(adj=adj, far=far, by_d=by_d, modal_layer=mL)
            payload[f"{fam}__adj_minus_far"] = adj - far

            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
            im = axes[0].imshow(com[jm], cmap=COS_CMAP, **DIVERGING)
            axes[0].set_xticks(range(C)); axes[0].set_yticks(range(C))
            axes[0].set_xticklabels(cyc, rotation=90, fontsize=6)
            axes[0].set_yticklabels(cyc, fontsize=6)
            axes[0].set_title(f"{fam} L{mL} (cycle order)", fontsize=9)
            fig.colorbar(im, ax=axes[0], fraction=0.046)
            for j, L in enumerate(LAYERS):
                axes[1].plot(range(1, dist.max() + 1), by_d[j], "-", lw=0.8,
                             color="0.75" if j != jm else C_BLUE,
                             zorder=3 if j == jm else 2,
                             label=f"L{mL} (modal)" if j == jm else None)
            axes[1].axhline(0, color="gray", lw=0.6)
            axes[1].set(xlabel="cycle distance", ylabel="mean sibling cosine",
                        title="adjacency vs distance (gray = other layers)")
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25, lw=0.4)
        else:
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            im = ax.imshow(cos[jm], cmap=COS_CMAP, **DIVERGING)
            ax.set_xticks(range(C)); ax.set_yticks(range(C))
            ax.set_xticklabels(classes, rotation=90, fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            ax.set_title(f"{fam} L{mL}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figdir / f"within_{fam}.png", dpi=150); plt.close(fig)
    np.savez(out / "within_family.npz", **payload)
    return circ


# ---------------------------------------------------------------- analysis 3
def arm_angles(data, concepts, out, figdir):
    pairs = ["ridge_dom", "ridge_lda", "dom_lda", "write_read"]
    A = np.full((len(concepts), len(LAYERS), 4), np.nan)
    for i, (fam, cls) in enumerate(tqdm(concepts, desc="E0.3 arm angles")):
        ci = data[fam]["classes"].index(cls)
        for j, L in enumerate(LAYERS):
            pl = data[fam]["per_layer"][L]
            r, d, l, sig = pl["ridge"][ci], pl["dom"][ci], pl["lda"][ci], pl["sigma"]
            wr, rd = unit(sig * r), unit(r / sig)   # raw write vs raw read arm
            A[i, j] = [r @ d, r @ l, d @ l, wr @ rd]
    np.savez(out / "arm_angles.npz", cos=A, pairs=np.array(pairs),
             layers=np.array(LAYERS),
             families=np.array([f for f, _ in concepts]),
             classes=np.array([c for _, c in concepts]))

    fams = [f for f, _ in concepts]
    fam_order = [f for f in FAMILIES if f in fams]
    med = np.zeros((4, len(fam_order), len(LAYERS)))
    for k in range(4):
        for fi, fam in enumerate(fam_order):
            rows = [i for i, f in enumerate(fams) if f == fam]
            med[k, fi] = np.nanmedian(A[rows, :, k], axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for k, (ax, p) in enumerate(zip(axes.flat, pairs)):
        im = ax.imshow(med[k], cmap=COS_CMAP, **DIVERGING, aspect="auto")
        ax.set_title(f"median cos({p.replace('_', ', ')})", fontsize=9)
        ax.set_xticks(range(len(LAYERS))); ax.set_xticklabels(LAYERS, fontsize=6)
        ax.set_yticks(range(len(fam_order)))
        ax.set_yticklabels([f + (" (CTRL)" if f == CONTROL else "")
                            for f in fam_order], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("E0.3 arm-angle cosines, family x layer medians")
    fig.tight_layout(); fig.savefig(figdir / "arm_angles.png", dpi=150)
    plt.close(fig)
    return A, pairs


# ---------------------------------------------------------------- analysis 4
def rand_baseline(data, concepts, out, figdir):
    d = 2304
    rr, rp = [], []                       # rand-rand, rand-ridge cosines
    iu = np.triu_indices(20, 1)
    seen = set()
    for fam in FAMILIES:
        for L in LAYERS:
            R = data[fam]["per_layer"][L]["rand"]
            key = R[0, :8].tobytes()      # dedupe identical saved rand sets
            if key not in seen:
                seen.add(key)
                rr.append((R @ R.T)[iu])
    for fam, cls in concepts:
        ci = data[fam]["classes"].index(cls)
        for L in LAYERS:
            pl = data[fam]["per_layer"][L]
            rp.append(pl["rand"] @ pl["ridge"][ci])
    rr, rp = np.concatenate(rr), np.concatenate(rp)
    np.savez(out / "rand_baseline.npz", rand_rand=rr, rand_ridge=rp,
             n_unique_rand_sets=len(seen))

    sig0 = 1 / np.sqrt(d)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(0, max(np.abs(rr).max(), np.abs(rp).max()) * 1.05, 60)
    ax.hist(np.abs(rr), bins=bins, density=True, histtype="step",
            color=C_BLUE, label=f"|cos| rand-rand (n={rr.size})")
    ax.hist(np.abs(rp), bins=bins, density=True, histtype="step",
            color=C_RED, label=f"|cos| rand-ridge (n={rp.size})")
    x = np.linspace(0, bins[-1], 300)
    ax.plot(x, 2 * np.exp(-x**2 / (2 * sig0**2)) / (sig0 * np.sqrt(2 * np.pi)),
            "--", color="0.4", label=f"half-N(0, 1/sqrt({d}))")
    ax.set(xlabel="|cosine|", ylabel="density", title="E0.4 random baseline")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(); fig.savefig(figdir / "rand_baseline.png", dpi=150)
    plt.close(fig)
    return rr, rp, len(seen)


# ------------------------------------------------------------------- summary
def fmt(x):
    return f"{x:.3f}"


def write_summary(path, concepts, xstats, circ, A, pairs, rr, rp, nsets,
                  rep_concepts, out, figdir):
    fams = np.array([f for f, _ in concepts])
    ctl = fams == CONTROL
    k_rd = pairs.index("ridge_dom")
    med_rd_all = np.nanmedian(A[~ctl, :, k_rd])
    per_fam_rd = {f: np.nanmedian(A[fams == f, :, k_rd])
                  for f in dict.fromkeys(fams)}
    adj = xstats["std"]["adj_per_concept"]
    decay = xstats["std"]["decay"]
    far_med = np.median([v for d, v in decay.items() if d >= 12])
    sig0 = 1 / np.sqrt(2304)

    L = ["# E0 summary — direction geometry (Stage 6.1)", "",
         f"Concepts: {len(concepts)} ({(~ctl).sum()} real + "
         f"{ctl.sum()} `{CONTROL}` nonsense control). Layers: {LAYERS}. "
         "Ridge = chosen-lambda row, all arms unit-normalized; standardized "
         "space unless noted (diagonal-sigma whitening proxy — full-Sigma "
         "whitening unavailable, a known limitation).", "",
         "## Headline numbers", ""]
    L += [f"- **median cos(ridge, DoM)** = {fmt(med_rd_all)} overall "
          f"(real concepts); per family: "
          + ", ".join(f"{f} {fmt(v)}" for f, v in per_fam_rd.items()) + "."]
    L += [f"- **Adjacent-layer same-concept cosine** (standardized): median "
          f"{fmt(np.median(adj[~ctl]))} across real concepts "
          f"(control {fmt(np.median(adj[ctl]))}); distant layers "
          f"(|l-l'| >= 12) median {fmt(far_med)}."]
    L += [f"- **Space choice**: adjacent-layer medians std/raw-read/raw-write = "
          + "/".join(fmt(np.median(xstats[sp]['adj_per_concept'][~ctl]))
                     for sp in ("std", "rawread", "rawwrite"))
          + " — see decay figure for the full curves."]
    for fam, st in circ.items():
        jm = LAYERS.index(st["modal_layer"])
        L += [f"- Cyclic `{fam}`: cycle-adjacent vs cycle-distant sibling cosine "
              f"at modal L{st['modal_layer']}: {fmt(st['adj'][jm])} vs "
              f"{fmt(st['far'][jm])} (gap {fmt(st['adj'][jm]-st['far'][jm])}); "
              f"mean gap across layers {fmt((st['adj']-st['far']).mean())}."]
    L += [f"- Arm-angle medians (real concepts): "
          + ", ".join(f"cos({p.replace('_',',')}) "
                      f"{fmt(np.nanmedian(A[~ctl,:,k]))}"
                      for k, p in enumerate(pairs)) + "."]
    L += [f"- `{CONTROL}` control arm angles: "
          + ", ".join(f"cos({p.replace('_',',')}) "
                      f"{fmt(np.nanmedian(A[ctl,:,k]))}"
                      for k, p in enumerate(pairs)) + "."]
    L += [f"- Random baseline: mean|cos| rand-rand {np.abs(rr).mean():.4f}, "
          f"rand-ridge {np.abs(rp).mean():.4f} vs expected "
          f"{sig0*np.sqrt(2/np.pi):.4f} (sigma0={sig0:.4f}); "
          f"max|cos| {np.abs(rr).max():.3f}/{np.abs(rp).max():.3f}; "
          f"frac>3*sigma0: {(np.abs(rr)>3*sig0).mean():.4f}/"
          f"{(np.abs(rp)>3*sig0).mean():.4f} (expect 0.0027 if isotropic); "
          f"{nsets} unique saved rand set(s)."]
    L += ["", "## Notes", ""]
    L += ["- Cross-layer similarity is *consistent with* copying but cannot "
          "establish it (basis drift / recomputation) — E5 is the causal "
          "counterpart; comparing E0 vs E5 matrices is a deliverable.",
          "- moon_phases class names use underscores in the probe npz "
          "(`new_moon`), not spaces as in stage6 plots.py CYCLES — normalized "
          "at load.",
          "- `rand_dirs` is [20, 2304] (not 5); all 20 used here.",
          f"- `{CONTROL}` has no probe_cards entry (64 cards = 13 real "
          "families); its modal layer defaults to the mid-list layer.",
          "", "## Artifacts", ""]
    for p in sorted(out.glob("*.npz")):
        L += [f"- `{p}`"]
    for p in sorted(figdir.glob("*.png")):
        L += [f"- `{p}`"]
    L += [f"- representative concepts in grid: "
          + ", ".join(f"{f}.{c}" for f, c in rep_concepts), ""]
    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    base = Path(__file__).resolve().parents[1]
    root = base.parents[0]              # concept_probes/
    ap.add_argument("--probes-root", default=str(root / "2_probes/probes"))
    ap.add_argument("--probe-cards",
                    default=str(root / "3_validation/artifacts/probe_cards.json"))
    ap.add_argument("--out", default=str(base / "out/e0"))
    ap.add_argument("--figures", default=str(base / "figures/e0"))
    args = ap.parse_args()
    out, figdir = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    log = out.parent / "progress_e0_geometry.log"

    data = {f: load_family(Path(args.probes_root), f) for f in
            tqdm(FAMILIES, desc="load probes")}
    concepts = [(f, c) for f in FAMILIES for c in data[f]["classes"]]

    cards = json.load(open(args.probe_cards))
    modal_layer = {f: Counter(c["layer"] for c in cards
                              if c["family"] == f).most_common(1)[0][0]
                   for f in {c["family"] for c in cards}}
    best = {}                            # representative concept per family
    for c in cards:
        if c["family"] not in best or (c["tier1"]["rho_nat_test"] >
                                       best[c["family"]][1]):
            best[c["family"]] = (c["concept"], c["tier1"]["rho_nat_test"])
    rep = [(f, best[f][0] if f in best else data[f]["classes"][0])
           for f in FAMILIES]
    rep = [(f, c if c in data[f]["classes"] else data[f]["classes"][0])
           for f, c in rep]

    xstats = crosslayer(data, concepts, out, figdir, rep, log)
    circ = within_family(data, out, figdir, modal_layer)
    A, pairs = arm_angles(data, concepts, out, figdir)
    rr, rp, nsets = rand_baseline(data, concepts, out, figdir)
    write_summary(out / "E0_SUMMARY.md", concepts, xstats, circ, A, pairs,
                  rr, rp, nsets, rep, out, figdir)
    heartbeat(log, "DONE")
    print(f"E0 done: {len(concepts)} concepts -> {out} , {figdir}")


if __name__ == "__main__":
    main()
