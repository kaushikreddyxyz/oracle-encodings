"""
geometry.py — Step 4 representation geometry (Tiers 1-5).

Operates on *class-conditional activation clouds* at a given layer, supplied by the
orchestrator from the Step-2/3 caches. Two input shapes:

  - presence clouds:  {concept: {class_name: ndarray (n_examples, d)}}
        e.g. presence_clouds["months"]["January"] -> (n, d)
  - scalar clouds:    {scalar_name: (X ndarray (n, d), y ndarray (n,))}
        e.g. scalar_clouds["numbers"] -> (X, y) with y the true scalar value

All structure is recovered from the clouds (not from a handful of lonely centroids),
and *every* angle / cosine / spacing claim carries a bootstrap 95% CI obtained by
resampling examples within each class/cloud. Each tier writes
`artifacts/geometry/{tier}.json` and a figure to `figures/`, and returns a result dict
with metrics+CIs and a one-paragraph `verdict`.

Run `python geometry.py --smoke` to validate the logic on synthetic clouds whose
ground-truth geometry is known (coplanar Z/12 cycles, clean Z/4, a shared magnitude
axis with log spacing, indoors=-outdoors, etc.); it asserts recovered == planted.

Instruments per tier:
  T1  PCA plane fit + variance fraction; circular winding score; gap-uniformity (CV);
      scipy.linalg.subspace_angles for principal angles; circular phase alignment.
  T2  principal angle (containment); centroid-direction cosine (coarse-graining);
      FFT harmonic energy; bucket-vs-members centroid cosine.
  T3  Ridge direction fit; pairwise cosines; SVD shared-axis variance fraction;
      cross-domain Spearman transfer; linear-vs-log R^2 + isotonic + Spearman.
  T4  PCA layout -> scipy.spatial.procrustes vs true lat/long; regression geographic
      axes vs compass N-S / E-W cosines.
  T5  Ridge directions -> cosine (indoors/outdoors) and angle (loving/harm);
      PCA top-PC variance fraction + participation ratio (is-it-1D).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.linalg import subspace_angles  # noqa: E402
from scipy.spatial import procrustes  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

# --- path wiring so `import config` / `import concepts` work however we're launched ---
_HERE = Path(__file__).resolve().parent          # overnight_run/code
sys.path.insert(0, str(_HERE))                   # config.py
sys.path.insert(0, str(_HERE.parent))            # concepts.py
import config            # noqa: E402
import concepts          # noqa: E402

N_BOOT = int(__import__("os").environ.get("N_BOOT", "1000"))   # env-tunable for time budget
GEODIR = config.ARTIFACTS / "geometry"
FIGDIR = config.FIGURES
GEODIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Low-level geometry helpers
# ===========================================================================
def class_centroids(class_clouds: dict, order=None):
    """{class: (n,d)} -> (names, C (k,d)) of per-class means in `order` (or dict order)."""
    names = [c for c in (order or list(class_clouds)) if c in class_clouds]
    C = np.stack([np.asarray(class_clouds[c]).mean(axis=0) for c in names])
    return names, C


def fit_plane(C: np.ndarray):
    """Best 2-plane of centroids C (k,d). Returns (basis (d,2) orthonormal,
    planarity = var fraction in plane, coords2d (k,2), mu (d,))."""
    mu = C.mean(axis=0)
    Cc = C - mu
    U, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    basis = Vt[:2].T                       # (d, 2) columns orthonormal
    var = s ** 2
    planarity = float(var[:2].sum() / var.sum()) if var.sum() > 0 else 0.0
    coords2d = Cc @ basis
    return basis, planarity, coords2d, mu


def principal_angles_deg(A: np.ndarray, B: np.ndarray):
    """Principal angles (degrees, ascending) between subspaces spanned by columns of A,B."""
    ang = np.degrees(np.sort(np.asarray(subspace_angles(A, B))))
    return ang


def _wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def circular_order_score(coords2d: np.ndarray):
    """Do the centroids (given in cyclic order) wind monotonically around the plane?
    Returns dict(order_score in [.5,1], winding ~ +-1 for one loop, correct bool)."""
    ang = np.arctan2(coords2d[:, 1], coords2d[:, 0])
    diffs = _wrap_pi(np.diff(np.concatenate([ang, ang[:1]])))   # close the loop
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    order_score = max(n_pos, n_neg) / len(diffs)
    winding = float(diffs.sum() / (2 * np.pi))
    correct = bool(order_score == 1.0 and abs(abs(winding) - 1.0) < 0.2)
    return {"order_score": float(order_score), "winding": winding, "correct": correct}


def angular_uniformity(coords2d: np.ndarray):
    """Are the gaps between consecutive (angularly sorted) centroids equal?
    Returns dict(uniformity in (-inf,1], gap_cv, max_gap_dev_deg, ideal_gap_deg)."""
    ang = np.sort(np.arctan2(coords2d[:, 1], coords2d[:, 0]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))   # wraps, sums to 2pi
    gaps_deg = np.degrees(gaps)
    n = len(ang)
    ideal = 360.0 / n
    cv = float(gaps_deg.std() / gaps_deg.mean()) if gaps_deg.mean() > 0 else np.inf
    return {
        "uniformity": float(1.0 - cv),
        "gap_cv": cv,
        "max_gap_dev_deg": float(np.max(np.abs(gaps_deg - ideal))),
        "ideal_gap_deg": ideal,
    }


def _circmean_deg(deg):
    r = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def cycle_phase_deg(coords2d: np.ndarray, n: int):
    """Absolute phase (deg) of a cyclic set in the frame of `coords2d`. Detects winding
    sign so the result is well-defined under reflection of the basis."""
    ang = np.arctan2(coords2d[:, 1], coords2d[:, 0])
    diffs = _wrap_pi(np.diff(np.concatenate([ang, ang[:1]])))
    s = 1.0 if diffs.sum() >= 0 else -1.0
    ideal = s * 2 * np.pi * np.arange(n) / n
    resid = np.degrees(_wrap_pi(ang - ideal))
    return _circmean_deg(resid)


def fourier_harmonics(C: np.ndarray):
    """Energy per harmonic of an ordered cyclic centroid sequence C (k,d).
    Returns dict(energy (k,), fundamental_frac = energy in m=1/(k-1) pair over m>=1)."""
    F = np.fft.fft(C - C.mean(axis=0), axis=0)
    energy = (np.abs(F) ** 2).sum(axis=1)            # (k,)
    k = len(C)
    nonzero = energy[1:].sum()
    fund = energy[1] + (energy[k - 1] if k > 1 else 0.0)
    frac = float(fund / nonzero) if nonzero > 0 else 0.0
    return {"energy": energy, "fundamental_frac": frac}


def fit_direction(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, method: str = "corr"):
    """Unit direction of the scalar y in activation space X.

    method='corr' (default): the correlation/covariance direction w = X_c^T y_c. This is
    the standard, robust linear-probe direction (≡ the brief's "centroid axis" generalized
    to a continuous scalar). Unlike a whitened Ridge weight it does NOT divide by the small
    eigenvalues of nuisance directions, so it stays aligned with the true signal axis even
    when activations carry many low-variance dimensions.
    method='ridge': the Ridge regression weight (whitened) — kept for cross-checks.
    Returns (unit direction (d,), fitted model or None)."""
    if method == "ridge":
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
        w = m.coef_.ravel()
    else:
        w = (X - X.mean(axis=0)).T @ (y - y.mean())
        m = None
    nrm = np.linalg.norm(w)
    return (w / nrm) if nrm > 0 else w, m


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(cosine(a, b), -1.0, 1.0))))


def _r2(y, yhat):
    y = np.asarray(y, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _pearson_r2(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)


# ===========================================================================
# Bootstrap
# ===========================================================================
def bootstrap_ci(metric_fn, data, resample_fn, n=N_BOOT, seed=0):
    """metric_fn(data) -> {name: float}. Resample examples `n` times via resample_fn,
    recompute, return {name: {point, lo, hi, n_boot}} with 95% percentile CIs."""
    rng = np.random.default_rng(seed)
    point = metric_fn(data)
    acc = {k: [] for k in point}
    for _ in range(n):
        try:
            m = metric_fn(resample_fn(data, rng))
        except Exception:
            continue
        for k in point:
            v = m.get(k, np.nan)
            if np.isfinite(v):
                acc[k].append(v)
    out = {}
    for k, p in point.items():
        arr = np.asarray(acc[k], float)
        if arr.size:
            out[k] = {"point": float(p), "lo": float(np.percentile(arr, 2.5)),
                      "hi": float(np.percentile(arr, 97.5)), "n_boot": int(arr.size)}
        else:
            out[k] = {"point": float(p), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return out


def _resample_classes(class_clouds, rng):
    return {k: v[rng.integers(0, len(v), len(v))] for k, v in class_clouds.items()}


def _resample_presence(data, rng):
    return {c: _resample_classes(cc, rng) for c, cc in data.items()}


def _resample_xy(data, rng):
    out = {}
    for k, (X, y) in data.items():
        idx = rng.integers(0, len(y), len(y))
        out[k] = (X[idx], y[idx])
    return out


# ===========================================================================
# IO helpers
# ===========================================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def _write_json(tier, obj, layer=None):
    fn = GEODIR / (f"{tier}.json" if layer is None else f"{tier}_L{layer}.json")
    fn.write_text(json.dumps(_jsonable(obj), indent=2))
    return str(fn)


def _save_fig(fig, name, layer=None):
    fn = FIGDIR / (f"{name}.png" if layer is None else f"{name}_L{layer}.png")
    fig.savefig(fn, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(fn)


def _ci(d):
    return f"{d['point']:.3g} [{d['lo']:.3g},{d['hi']:.3g}]"


def _order_of(concept):
    """Canonical cyclic/class order from concepts.py (falls back to cloud order)."""
    c = concepts.PRESENCE_CONCEPTS.get(concept)
    return list(c["classes"]) if c else None


# ===========================================================================
# Tier 1 — Z/12 collision study (headline)
# ===========================================================================
def tier1(presence_clouds, layer=None, n_boot=N_BOOT, save_fig=True):
    G = concepts.GEOMETRY["tier1_z12"]
    z12 = [c for c in G["cycles"] if c in presence_clouds]
    z4 = [c for c in G["z4_cycles"] if c in presence_clouds]
    orders = {c: _order_of(c) for c in z12 + z4}

    def _per_cycle(cc, order):
        _, C = class_centroids(cc, order)
        basis, planarity, coords, mu = fit_plane(C)
        return {"basis": basis, "planarity": planarity, "coords": coords, "mu": mu,
                "C": C, "n": len(C),
                "order": circular_order_score(coords),
                "unif": angular_uniformity(coords)}

    def _rel_phase(infoA, ccB, orderB):
        _, CB = class_centroids(ccB, orderB)
        coordsB_in_A = (CB - infoA["mu"]) @ infoA["basis"]
        pB = cycle_phase_deg(coordsB_in_A, len(CB))
        pA = cycle_phase_deg(infoA["coords"], infoA["n"])
        return float(((pB - pA) + 180) % 360 - 180)

    def metric_fn(data):
        info = {c: _per_cycle(data[c], orders[c]) for c in z12 + z4}
        out = {}
        for c in z12 + z4:
            out[f"{c}/planarity"] = info[c]["planarity"]
            out[f"{c}/order_score"] = info[c]["order"]["order_score"]
            out[f"{c}/uniformity"] = info[c]["unif"]["uniformity"]
        for a, b in combinations(z12, 2):
            ang = principal_angles_deg(info[a]["basis"], info[b]["basis"])
            out[f"{a}|{b}/theta1"] = float(ang[0])
            out[f"{a}|{b}/theta2"] = float(ang[1])
            out[f"{a}|{b}/phase_deg"] = _rel_phase(info[a], data[b], orders[b])
        for a, b in combinations(z4, 2):
            ang = principal_angles_deg(info[a]["basis"], info[b]["basis"])
            out[f"{a}|{b}/theta1"] = float(ang[0])
            out[f"{a}|{b}/theta2"] = float(ang[1])
        return out

    data = {c: presence_clouds[c] for c in z12 + z4}
    ci = bootstrap_ci(metric_fn, data, _resample_presence, n=n_boot)
    info = {c: _per_cycle(data[c], orders[c]) for c in z12 + z4}

    # verdict
    pair_lines = []
    coplanar = []
    for a, b in combinations(z12, 2):
        t2 = ci[f"{a}|{b}/theta2"]
        same = t2["hi"] < 15.0
        coplanar.append(same)
        ph = ci.get(f"{a}|{b}/phase_deg")
        pair_lines.append(f"{a}/{b}: theta=({_ci(ci[f'{a}|{b}/theta1'])},"
                          f"{_ci(t2)})deg, phase={_ci(ph)}deg")
    verdict = (
        "Z/12 collision: " +
        ("cycles SHARE a cyclic subspace (small principal angles) -> geometry follows the "
         "abstract Z/12 group, not semantics. "
         if coplanar and all(coplanar) else
         "cycles occupy DISTINCT subspaces (large principal angles) -> per-concept geometry. ")
        + "; ".join(pair_lines) + ". "
    )
    if z4:
        z4line = []
        for a, b in combinations(z4, 2):
            z4line.append(f"{a}/{b}: theta2={_ci(ci[f'{a}|{b}/theta2'])}deg")
        verdict += "Z/4 (" + "; ".join(z4line) + ")."

    fig_path = _tier1_figure(info, z12, z4, layer) if save_fig else None
    result = {"tier": "tier1", "layer": layer, "z12": z12, "z4": z4,
              "metrics": ci, "verdict": verdict, "figure": fig_path}
    result["json"] = _write_json("tier1", result, layer)
    return result


def _tier1_figure(info, z12, z4, layer):
    cycles = z12 + z4
    ncol = max(1, len(cycles))
    fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cycles):
        co = info[c]["coords"]
        loop = np.vstack([co, co[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "-o", ms=4, lw=1)
        for i, (x, y) in enumerate(co):
            ax.annotate(str(i), (x, y), fontsize=7)
        ax.set_title(f"{c}\nplan={info[c]['planarity']:.2f} "
                     f"ord={info[c]['order']['order_score']:.2f}", fontsize=8)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.8", lw=.5)
        ax.axvline(0, color="0.8", lw=.5)
    fig.suptitle("Tier 1 — cyclic 2-plane projections", fontsize=10)
    return _save_fig(fig, "tier1_cycles", layer)


# ===========================================================================
# Tier 2 — Harmonic nesting
# ===========================================================================
def tier2(presence_clouds, layer=None, n_boot=N_BOOT, save_fig=True):
    G = concepts.GEOMETRY["tier2_nesting"]
    sim = G["season_in_month"]
    season_map = sim["map"]
    have_sm = "months" in presence_clouds and "seasons" in presence_clouds
    have_bn = "numbers10" in presence_clouds and "numbers100" in presence_clouds

    m_order = _order_of("months")
    s_order = _order_of("seasons")

    def _dir(c, mu):  # centroid direction (unit), relative to a common mean
        v = c - mu
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def metric_fn(data):
        out = {}
        if have_sm:
            mn, M = class_centroids(data["months"], m_order)
            sn, S = class_centroids(data["seasons"], s_order)
            Mb, _, _, _ = fit_plane(M)
            Sb, _, _, _ = fit_plane(S)
            ang = principal_angles_deg(Mb, Sb)
            out["season_month_theta1"] = float(ang[0])
            out["season_month_theta2"] = float(ang[1])
            gmu = np.vstack([M, S]).mean(0)
            mcen = {name: M[i] for i, name in enumerate(mn)}
            cos = []
            for season, mons in season_map.items():
                if season not in sn:
                    continue
                mons_present = [m for m in mons if m in mcen]
                if not mons_present:
                    continue
                coarse = np.mean([_dir(mcen[m], gmu) for m in mons_present], axis=0)
                fine = _dir(S[sn.index(season)], gmu)
                cos.append(cosine(coarse, fine))
            out["coarse_grain_cosine"] = float(np.mean(cos)) if cos else np.nan
            out["month_fundamental_frac"] = fourier_harmonics(M)["fundamental_frac"]
        if have_bn:
            n10, C10 = class_centroids(data["numbers10"])
            val10 = {name: C10[i] for i, name in enumerate(n10)}
            n100, C100 = class_centroids(data["numbers100"])
            cos = []
            for i, bname in enumerate(n100):
                rng = concepts.NUMBERS_100["classes"][bname]["range"]
                members = [v for v in val10 if v.isdigit() and rng[0] <= int(v) <= rng[1]]
                if not members:
                    continue
                memc = np.mean([val10[v] for v in members], axis=0)
                cos.append(cosine(C100[i], memc))
            out["bucket_member_cosine"] = float(np.mean(cos)) if cos else np.nan
        return out

    data = {k: presence_clouds[k] for k in
            ("months", "seasons", "numbers10", "numbers100") if k in presence_clouds}
    ci = bootstrap_ci(metric_fn, data, _resample_presence, n=n_boot)

    parts = []
    if have_sm:
        contained = ci["season_month_theta2"]["hi"] < 15.0
        parts.append(
            ("Seasons LIE IN the month plane" if contained else "Seasons depart from the month plane")
            + f" (theta=({_ci(ci['season_month_theta1'])},{_ci(ci['season_month_theta2'])})deg); "
            + f"coarse-graining dir(season)~mean(dir(months)) cosine={_ci(ci['coarse_grain_cosine'])}; "
            + f"month centroids' 1st-harmonic energy fraction={_ci(ci['month_fundamental_frac'])} "
            + "(seasons = fundamental Fourier mode of months).")
    if have_bn:
        parts.append(f"Base-10 in base-100: bucket centroid ~ mean of its unit members, "
                     f"cosine={_ci(ci['bucket_member_cosine'])} (multiscale magnitude coding).")
    verdict = "Harmonic nesting: " + " ".join(parts) if parts else "Tier 2: no inputs."

    fig_path = _tier2_figure(presence_clouds, m_order, s_order, season_map, layer) \
        if (save_fig and have_sm) else None
    result = {"tier": "tier2", "layer": layer, "metrics": ci, "verdict": verdict,
              "figure": fig_path}
    result["json"] = _write_json("tier2", result, layer)
    return result


def _tier2_figure(presence_clouds, m_order, s_order, season_map, layer):
    mn, M = class_centroids(presence_clouds["months"], m_order)
    sn, S = class_centroids(presence_clouds["seasons"], s_order)
    basis, planarity, Mc, mu = fit_plane(M)
    Sc = (S - mu) @ basis
    har = fourier_harmonics(M)["energy"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
    loop = np.vstack([Mc, Mc[:1]])
    ax1.plot(loop[:, 0], loop[:, 1], "-o", ms=4, color="steelblue", label="months")
    for i, name in enumerate(mn):
        ax1.annotate(name[:3], Mc[i], fontsize=6)
    ax1.scatter(Sc[:, 0], Sc[:, 1], s=90, marker="*", color="crimson", label="seasons", zorder=5)
    for i, name in enumerate(sn):
        ax1.annotate(name, Sc[i], fontsize=7, color="crimson")
    ax1.set_title(f"seasons on the month plane (plan={planarity:.2f})", fontsize=8)
    ax1.set_aspect("equal"); ax1.legend(fontsize=7)
    k = len(har)
    ax2.bar(range(1, k), har[1:] / har[1:].sum())
    ax2.set_title("month-centroid harmonic energy", fontsize=8)
    ax2.set_xlabel("harmonic m"); ax2.set_ylabel("energy frac")
    return _save_fig(fig, "tier2_nesting", layer)


# ===========================================================================
# Tier 3 — The abstract magnitude axis
# ===========================================================================
def tier3(scalar_clouds, moon_bridge=None, layer=None, n_boot=N_BOOT, save_fig=True):
    """scalar_clouds: {name: (X,y)} for the magnitude scalars.
    moon_bridge: optional (X,y) where y = moon illumination (the cyclic->scalar bridge)."""
    G = concepts.GEOMETRY["tier3_magnitude"]
    domains = [d for d in G["scalars"] if d in scalar_clouds]

    def metric_fn(data):
        out = {}
        dirs = {d: fit_direction(*data[d])[0] for d in domains}
        # pairwise cosines (sign-aware)
        cs = [cosine(dirs[a], dirs[b]) for a, b in combinations(domains, 2)]
        out["mean_pairwise_cosine"] = float(np.mean(cs)) if cs else np.nan
        # shared axis via SVD of stacked unit directions
        D = np.stack([dirs[d] for d in domains])
        s = np.linalg.svd(D, compute_uv=False)
        out["shared_pc1_frac"] = float(s[0] ** 2 / (s ** 2).sum())
        shared_axis = np.linalg.svd(D, full_matrices=False)[2][0]
        # cross-domain transfer: train on numbers, predict the rest
        if "numbers" in data:
            Xn, yn = data["numbers"]
            mnum = Ridge(alpha=1.0).fit(Xn, yn)
            for tgt in domains:
                if tgt == "numbers":
                    continue
                Xt, yt = data[tgt]
                out[f"transfer_{tgt}_spearman"] = float(spearmanr(mnum.predict(Xt), yt).statistic)
        # spacing: projection vs value, linear vs log
        for d in domains:
            X, y = data[d]
            p = (X - X.mean(0)) @ dirs[d]
            out[f"{d}_spearman"] = float(spearmanr(p, y).statistic)
            out[f"{d}_r2_lin"] = _pearson_r2(p, y)
            if np.all(y > 0):
                out[f"{d}_r2_log"] = _pearson_r2(p, np.log(y))
        # moon illumination on the shared axis
        if moon_bridge is not None:
            mb = data.get("__moon__", moon_bridge)
            dm = fit_direction(*mb)[0]
            out["moon_illum_axis_cosine"] = abs(cosine(dm, shared_axis))
        return out

    data = {d: scalar_clouds[d] for d in domains}
    if moon_bridge is not None:
        data["__moon__"] = moon_bridge
    ci = bootstrap_ci(metric_fn, data, _resample_xy, n=n_boot)

    shared = ci["shared_pc1_frac"]
    spacing = []
    for d in domains:
        kl, kg = f"{d}_r2_lin", f"{d}_r2_log"
        if kg in ci:
            kind = "log" if ci[kg]["point"] > ci[kl]["point"] else "linear"
            spacing.append(f"{d}:{kind}(R2lin={ci[kl]['point']:.2f},R2log={ci[kg]['point']:.2f})")
        else:
            spacing.append(f"{d}:linR2={ci[kl]['point']:.2f}")
    transfer = [f"{t.split('_')[1]}={_ci(ci[t])}" for t in ci if t.startswith("transfer_")]
    verdict = (
        f"Abstract magnitude axis: a single shared axis explains {_ci(shared)} of the variance "
        f"across {domains} (mean pairwise cosine {_ci(ci['mean_pairwise_cosine'])}). "
        + (f"Cross-domain Spearman transfer from numbers: {', '.join(transfer)} "
           "(reused magnitude code). " if transfer else "")
        + "Spacing: " + "; ".join(spacing) + ". "
        + (f"Moon illumination loads on the shared axis (|cos|={_ci(ci['moon_illum_axis_cosine'])})."
           if "moon_illum_axis_cosine" in ci else "")
    )

    fig_path = _tier3_figure(data, domains, layer) if save_fig else None
    result = {"tier": "tier3", "layer": layer, "domains": domains, "metrics": ci,
              "verdict": verdict, "figure": fig_path}
    result["json"] = _write_json("tier3", result, layer)
    return result


def _tier3_figure(data, domains, layer):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    # transfer scatter (numbers-trained probe on every other domain)
    if "numbers" in data:
        Xn, yn = data["numbers"]
        m = Ridge(alpha=1.0).fit(Xn, yn)
        for tgt in domains:
            if tgt == "numbers":
                continue
            Xt, yt = data[tgt]
            pr = m.predict(Xt)
            order = np.argsort(yt)
            ax1.scatter(yt[order], pr[order], s=8, alpha=.5, label=tgt)
        ax1.set_xlabel("true scalar (target domain)")
        ax1.set_ylabel("numbers-probe prediction")
        ax1.set_title("cross-domain transfer", fontsize=9)
        ax1.legend(fontsize=7)
    # spacing: projection vs value for each domain
    for d in domains:
        X, y = data[d]
        w, _ = fit_direction(X, y)
        p = (X - X.mean(0)) @ w
        o = np.argsort(y)
        ax2.scatter(y[o], p[o], s=8, alpha=.5, label=d)
    ax2.set_xlabel("true scalar")
    ax2.set_ylabel("projection on magnitude axis")
    ax2.set_title("linear vs log spacing", fontsize=9)
    ax2.legend(fontsize=7)
    return _save_fig(fig, "tier3_magnitude", layer)


# ===========================================================================
# Tier 4 — A recovered world map
# ===========================================================================
def tier4(place_clouds, latlong, compass_clouds=None, layer=None, n_boot=N_BOOT, save_fig=True):
    """place_clouds: {place: (n,d)}; latlong: {place: (lat, lon)};
    compass_clouds: {North,East,South,West: (n,d)} (the directions concept)."""
    places = [p for p in place_clouds if p in latlong]
    true_xy = np.array([latlong[p] for p in places], float)   # (P, 2) [lat, lon]

    def metric_fn(data):
        pc, comp = data["places"], data.get("compass")
        C = np.stack([pc[p].mean(0) for p in places])
        _, _, coords, _ = fit_plane(C)                        # PCA 2-D layout
        _, recovered, disparity = procrustes(true_xy, coords)
        out = {"procrustes_disparity": float(disparity)}
        # geographic axes in activation space via regression
        w_lat, _ = fit_direction(C, true_xy[:, 0])
        w_lon, _ = fit_direction(C, true_xy[:, 1])
        if comp is not None and all(k in comp for k in ("North", "South", "East", "West")):
            ns = comp["North"].mean(0) - comp["South"].mean(0)
            ew = comp["East"].mean(0) - comp["West"].mean(0)
            out["compass_NS_vs_lat_cos"] = abs(cosine(ns, w_lat))
            out["compass_EW_vs_lon_cos"] = abs(cosine(ew, w_lon))
        return out

    def resample(data, rng):
        out = {"places": _resample_classes(data["places"], rng)}
        if data.get("compass") is not None:
            out["compass"] = _resample_classes(data["compass"], rng)
        return out

    data = {"places": {p: place_clouds[p] for p in places}, "compass": compass_clouds}
    ci = bootstrap_ci(metric_fn, data, resample, n=n_boot)

    parts = [f"PCA layout of continent/place centroids Procrustes-aligns to true lat/long with "
             f"disparity={_ci(ci['procrustes_disparity'])} (0=perfect metric map)."]
    if "compass_NS_vs_lat_cos" in ci:
        parts.append(f"Compass shares the map's frame: N-S vs latitude |cos|="
                     f"{_ci(ci['compass_NS_vs_lat_cos'])}, E-W vs longitude |cos|="
                     f"{_ci(ci['compass_EW_vs_lon_cos'])}.")
    verdict = "World map: " + " ".join(parts)

    fig_path = _tier4_figure(place_clouds, places, true_xy, layer) if save_fig else None
    result = {"tier": "tier4", "layer": layer, "places": places, "metrics": ci,
              "verdict": verdict, "figure": fig_path}
    result["json"] = _write_json("tier4", result, layer)
    return result


def _tier4_figure(place_clouds, places, true_xy, layer):
    C = np.stack([place_clouds[p].mean(0) for p in places])
    _, _, coords, _ = fit_plane(C)
    t_std, rec, disp = procrustes(true_xy, coords)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(t_std[:, 1], t_std[:, 0], c="k", marker="o", label="true (lat/long)")
    ax.scatter(rec[:, 1], rec[:, 0], c="crimson", marker="x", label="recovered")
    for i, p in enumerate(places):
        ax.plot([t_std[i, 1], rec[i, 1]], [t_std[i, 0], rec[i, 0]], "0.7", lw=.6)
        ax.annotate(p, (t_std[i, 1], t_std[i, 0]), fontsize=7)
    ax.set_xlabel("longitude (Procrustes)"); ax.set_ylabel("latitude (Procrustes)")
    ax.set_title(f"Tier 4 — recovered world map (disparity={disp:.3f})", fontsize=9)
    ax.legend(fontsize=7)
    return _save_fig(fig, "tier4_worldmap", layer)


# ===========================================================================
# Tier 5 — Antipodal / opponent structure
# ===========================================================================
def tier5(scalar_clouds, layer=None, n_boot=N_BOOT, save_fig=True):
    pairs = [p for p in concepts.GEOMETRY["tier5_antipodal"]["pairs"]
             if all(x in scalar_clouds for x in p)]
    members = sorted({x for p in pairs for x in p})

    def _participation_ratio(X):
        lam = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
        return float(lam.sum() ** 2 / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else 0.0

    def _top_frac(X):
        lam = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
        return float(lam[0] / lam.sum()) if lam.sum() > 0 else 0.0

    def metric_fn(data):
        out = {}
        dirs = {m: fit_direction(*data[m])[0] for m in members}
        for a, b in pairs:
            out[f"{a}|{b}/cosine"] = cosine(dirs[a], dirs[b])
            out[f"{a}|{b}/angle_deg"] = angle_deg(dirs[a], dirs[b])
        for m in members:
            X, _ = data[m]
            out[f"{m}/top_pc_frac"] = _top_frac(X)
            out[f"{m}/participation_ratio"] = _participation_ratio(X)
        return out

    # cap cloud size: top_frac/participation_ratio SVD each cloud (n,3584) per bootstrap;
    # ~200 pts is plenty for these spectral stats and keeps tier5 from dominating runtime.
    _rng5 = np.random.default_rng(0)
    data = {}
    for m in members:
        X, y = scalar_clouds[m]
        if len(y) > 200:
            sel = _rng5.choice(len(y), 200, replace=False)
            X, y = X[sel], y[sel]
        data[m] = (X, y)
    ci = bootstrap_ci(metric_fn, data, _resample_xy, n=n_boot)

    parts = []
    for a, b in pairs:
        cs = ci[f"{a}|{b}/cosine"]
        ag = ci[f"{a}|{b}/angle_deg"]
        if cs["hi"] < -0.8:
            verd = "ONE axis (antipodal, ~ -1)"
        elif abs(cs["point"]) < 0.3:
            verd = "ORTHOGONAL (two independent features)"
        else:
            verd = "partially aligned"
        parts.append(f"{a} vs {b}: cosine={_ci(cs)}, angle={_ci(ag)}deg -> {verd}")
    onedim = "; ".join(f"{m} top-PC={_ci(ci[f'{m}/top_pc_frac'])}" for m in members)
    verdict = "Antipodal structure: " + " | ".join(parts) + f". 1-D check (top-PC var frac): {onedim}."

    fig_path = _tier5_figure(ci, pairs, members, layer) if save_fig else None
    result = {"tier": "tier5", "layer": layer, "pairs": pairs, "metrics": ci,
              "verdict": verdict, "figure": fig_path}
    result["json"] = _write_json("tier5", result, layer)
    return result


def _tier5_figure(ci, pairs, members, layer):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    labels = [f"{a}/{b}" for a, b in pairs]
    vals = [ci[f"{a}|{b}/cosine"]["point"] for a, b in pairs]
    err = np.clip(
        [[ci[f"{a}|{b}/cosine"]["point"] - ci[f"{a}|{b}/cosine"]["lo"] for a, b in pairs],
         [ci[f"{a}|{b}/cosine"]["hi"] - ci[f"{a}|{b}/cosine"]["point"] for a, b in pairs]],
        0, None)
    ax1.bar(labels, vals, yerr=err, capsize=4, color="slateblue")
    ax1.axhline(-1, color="r", ls="--", lw=.8, label="antipodal")
    ax1.axhline(0, color="0.5", ls="--", lw=.8, label="orthogonal")
    ax1.set_ylim(-1.1, 1.1); ax1.set_title("pair direction cosine", fontsize=9)
    ax1.legend(fontsize=7); ax1.tick_params(axis="x", labelsize=7)
    tf = [ci[f"{m}/top_pc_frac"]["point"] for m in members]
    ax2.bar(members, tf, color="seagreen")
    ax2.axhline(1.0, color="r", ls="--", lw=.8)
    ax2.set_ylim(0, 1.1); ax2.set_title("top-PC variance fraction (1=1-D)", fontsize=9)
    ax2.tick_params(axis="x", labelsize=7, rotation=30)
    return _save_fig(fig, "tier5_antipodal", layer)


# ===========================================================================
# Driver
# ===========================================================================
def run_all(presence_clouds=None, scalar_clouds=None, places=None, latlong=None,
            moon_bridge=None, layer=None, n_boot=N_BOOT, save_fig=True):
    """Run every applicable tier given whatever clouds the orchestrator supplies."""
    results = {}
    presence_clouds = presence_clouds or {}
    scalar_clouds = scalar_clouds or {}
    if any(c in presence_clouds for c in concepts.GEOMETRY["tier1_z12"]["cycles"]):
        results["tier1"] = tier1(presence_clouds, layer, n_boot, save_fig)
    if "months" in presence_clouds or "numbers10" in presence_clouds:
        results["tier2"] = tier2(presence_clouds, layer, n_boot, save_fig)
    if scalar_clouds:
        if any(d in scalar_clouds for d in concepts.GEOMETRY["tier3_magnitude"]["scalars"]):
            results["tier3"] = tier3(scalar_clouds, moon_bridge, layer, n_boot, save_fig)
        if any(all(x in scalar_clouds for x in p)
               for p in concepts.GEOMETRY["tier5_antipodal"]["pairs"]):
            results["tier5"] = tier5(scalar_clouds, layer, n_boot, save_fig)
    if places and latlong:
        results["tier4"] = tier4(places, latlong, presence_clouds.get("directions"),
                                 layer, n_boot, save_fig)
    return results


def write_geometry_md(results, path=None):
    path = Path(path) if path else (GEODIR / "geometry.md")
    lines = ["# Representation geometry — verdicts\n"]
    for tier in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        if tier in results:
            lines += [f"## {tier}", results[tier]["verdict"],
                      f"_figure_: `{results[tier].get('figure')}`\n"]
    path.write_text("\n".join(lines))
    return str(path)


# ===========================================================================
# Smoke test — plant known structure, assert recovery
# ===========================================================================
def _synthesize(seed=0, d=48, npc=60, nsc=240, noise=0.03):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, 10)))
    u, v = Q[:, 0], Q[:, 1]          # shared cyclic plane (= world-map plane)
    g = Q[:, 2]                      # magnitude axis
    h = Q[:, 3]                      # indoors/outdoors axis
    p, q = Q[:, 4], Q[:, 5]         # lovingness (p) / harmfulness (q, orthogonal)

    def cyc(names, phase_deg, R=1.0):
        out = {}
        n = len(names)
        for i, nm in enumerate(names):
            th = 2 * np.pi * i / n + np.radians(phase_deg)
            c = R * (np.cos(th) * u + np.sin(th) * v)
            out[nm] = c + noise * rng.standard_normal((npc, d))
        return out

    presence = {}
    presence["months"] = cyc(_order_of("months"), 0.0)
    presence["color_wheel"] = cyc(_order_of("color_wheel"), 40.0)   # planted phase +40
    presence["moon_phases"] = cyc(_order_of("moon_phases"), 0.0)    # planted phase 0
    # directions: North=+u(0),East=+v(90),South=-u(180),West=-v(270)
    presence["directions"] = cyc(_order_of("directions"), 0.0)

    # seasons derived as the coarsening of months (tier-2 ground truth)
    mcen = {m: presence["months"][m].mean(0) for m in presence["months"]}
    season_map = concepts.GEOMETRY["tier2_nesting"]["season_in_month"]["map"]
    presence["seasons"] = {}
    for s in _order_of("seasons"):
        c = np.mean([mcen[m] for m in season_map[s]], axis=0)
        c = c / np.linalg.norm(c)
        presence["seasons"][s] = c + noise * rng.standard_normal((npc, d))

    # numbers 0..10 on a line; base-100 buckets = centroid of their members
    n10 = {}
    for k in range(0, 11):
        n10[str(k)] = (k * 0.5) * g + 2.0 * Q[:, 6] + noise * rng.standard_normal((npc, d))
    presence["numbers10"] = n10
    n100 = {}
    cen10 = {k: n10[k].mean(0) for k in n10}
    for lo in range(0, 100, 10):
        bname = f"{lo}-{lo+9}"
        members = [str(k) for k in range(lo, lo + 10) if str(k) in cen10]
        base = (np.mean([cen10[m] for m in members], axis=0) if members
                else rng.standard_normal(d))
        n100[bname] = base + noise * rng.standard_normal((npc, d))
    presence["numbers100"] = n100

    # ---- scalar clouds (X, y): shared magnitude axis g, different spacings ----
    def mag(yvals, fy, off):
        y = np.asarray(yvals, float)
        X = fy(y)[:, None] * g[None, :] + off + noise * rng.standard_normal((len(y), d))
        return X, y

    yn = rng.uniform(1, 10, nsc)
    yc = rng.uniform(1, 1000, nsc)
    ys = rng.uniform(1, 100, nsc)
    yd = rng.uniform(1, 1e6, nsc)
    scalar = {
        "numbers":       mag(yn, lambda y: y, 1.0 * Q[:, 6]),          # linear
        "costliness":    mag(yc, lambda y: np.log(y), 1.0 * Q[:, 7]),  # log
        "physical_size": mag(ys, lambda y: np.log(y), 1.0 * Q[:, 8]),  # log
        "duration":      mag(yd, lambda y: np.log(y), 1.0 * Q[:, 9]),  # log
    }
    # moon illumination bridge: y = illum, direction = g
    illum = concepts.MOON_PHASES["classes"]
    phases = list(illum)
    yi = rng.choice([illum[ph]["illum"] for ph in phases], nsc)
    moon_bridge = (yi[:, None] * g[None, :] + noise * rng.standard_normal((nsc, d)), yi)

    # antipodal scalars (amplitude A so the planted feature is genuinely 1-D vs the
    # isotropic d-dim noise floor; A^2*var(y) >> d*noise^2)
    A = 5.0
    yi_io = rng.uniform(0, 1, nsc)
    scalar["indoors"] = (A * yi_io[:, None] * h[None, :] + noise * rng.standard_normal((nsc, d)), yi_io)
    yo = rng.uniform(0, 1, nsc)
    scalar["outdoors"] = (A * yo[:, None] * (-h)[None, :] + noise * rng.standard_normal((nsc, d)), yo)
    yl = rng.uniform(-1, 1, nsc)
    scalar["lovingness"] = (A * yl[:, None] * p[None, :] + noise * rng.standard_normal((nsc, d)), yl)
    yh = rng.uniform(-1, 1, nsc)
    scalar["harmfulness"] = (A * yh[:, None] * q[None, :] + noise * rng.standard_normal((nsc, d)), yh)

    # ---- world map: real cities, embed at lat*u + lon*v ----
    latlong = {
        "London": (51.5, -0.1), "Paris": (48.9, 2.4), "Berlin": (52.5, 13.4),
        "NewYork": (40.7, -74.0), "MexicoCity": (19.4, -99.1), "SaoPaulo": (-23.6, -46.6),
        "Cairo": (30.0, 31.2), "Lagos": (6.5, 3.4), "Nairobi": (-1.3, 36.8),
    }
    places = {}
    for nm, (lat, lon) in latlong.items():
        c = lat * u + lon * v
        places[nm] = c + 0.5 * rng.standard_normal((40, d))   # small noise vs O(50) signal

    planted = {"phase_color": 40.0, "phase_moon": 0.0}
    return presence, scalar, moon_bridge, places, latlong, planted


def _smoke(n_boot=200):
    print("=== geometry.py smoke: synthesizing clouds with KNOWN structure ===")
    presence, scalar, moon_bridge, places, latlong, planted = _synthesize()
    fails = []

    def check(name, cond, got):
        flag = "OK " if cond else "FAIL"
        if not cond:
            fails.append(name)
        print(f"  [{flag}] {name}: {got}")

    # ---------------- Tier 1 ----------------
    print("\n-- Tier 1 (Z/12 collision) --  planted: months/colors/moon coplanar; "
          "color phase=+40, moon phase=0; seasons||directions coplanar")
    r1 = tier1(presence, n_boot=n_boot)
    m = r1["metrics"]
    for c in ("months", "color_wheel", "moon_phases"):
        check(f"T1 {c} planarity~1", m[f"{c}/planarity"]["point"] > 0.97,
              f"{m[f'{c}/planarity']['point']:.3f}")
        check(f"T1 {c} ordering correct", m[f"{c}/order_score"]["point"] == 1.0,
              f"{m[f'{c}/order_score']['point']:.2f}")
        check(f"T1 {c} uniform", m[f"{c}/uniformity"]["point"] > 0.95,
              f"{m[f'{c}/uniformity']['point']:.3f}")
    for a, b in combinations(["months", "color_wheel", "moon_phases"], 2):
        t2 = m[f"{a}|{b}/theta2"]["point"]
        check(f"T1 principal angle {a}|{b}~0", t2 < 6.0, f"theta2={t2:.2f}deg")
    pc = m["months|color_wheel/phase_deg"]["point"]
    check("T1 month->color phase~40", min(abs(pc - 40), abs(pc + 40)) < 6.0, f"{pc:.1f}deg")
    pm = m["months|moon_phases/phase_deg"]["point"]
    check("T1 month->moon phase~0", min(abs(pm), abs(abs(pm) - 360)) < 6.0, f"{pm:.1f}deg")
    z4t = m["seasons|directions/theta2"]["point"]
    check("T1 Z/4 seasons|directions~0", z4t < 6.0, f"theta2={z4t:.2f}deg")

    # ---------------- Tier 2 ----------------
    print("\n-- Tier 2 (nesting) --  planted: seasons=coarsening of months; "
          "base100 bucket=centroid of members")
    r2 = tier2(presence, n_boot=n_boot)
    m = r2["metrics"]
    check("T2 season in month plane", m["season_month_theta2"]["point"] < 6.0,
          f"theta2={m['season_month_theta2']['point']:.2f}deg")
    check("T2 coarse-grain cosine~1", m["coarse_grain_cosine"]["point"] > 0.95,
          f"{m['coarse_grain_cosine']['point']:.3f}")
    check("T2 month 1st-harmonic dominates", m["month_fundamental_frac"]["point"] > 0.9,
          f"{m['month_fundamental_frac']['point']:.3f}")
    check("T2 bucket~members cosine~1", m["bucket_member_cosine"]["point"] > 0.95,
          f"{m['bucket_member_cosine']['point']:.3f}")

    # ---------------- Tier 3 ----------------
    print("\n-- Tier 3 (magnitude axis) --  planted: shared axis g; numbers linear, "
          "cost/size/duration log; moon illum on g")
    r3 = tier3(scalar, moon_bridge=moon_bridge, n_boot=n_boot)
    m = r3["metrics"]
    check("T3 shared PC1 frac~1", m["shared_pc1_frac"]["point"] > 0.95,
          f"{m['shared_pc1_frac']['point']:.3f}")
    check("T3 mean pairwise cosine~1", m["mean_pairwise_cosine"]["point"] > 0.95,
          f"{m['mean_pairwise_cosine']['point']:.3f}")
    for tgt in ("costliness", "physical_size", "duration"):
        k = f"transfer_{tgt}_spearman"
        check(f"T3 transfer numbers->{tgt}", m[k]["point"] > 0.9, f"rho={m[k]['point']:.3f}")
    check("T3 numbers linear>log", m["numbers_r2_lin"]["point"] > m["numbers_r2_log"]["point"],
          f"lin={m['numbers_r2_lin']['point']:.3f} log={m['numbers_r2_log']['point']:.3f}")
    check("T3 costliness log>linear", m["costliness_r2_log"]["point"] > m["costliness_r2_lin"]["point"],
          f"log={m['costliness_r2_log']['point']:.3f} lin={m['costliness_r2_lin']['point']:.3f}")
    check("T3 moon illum on shared axis", m["moon_illum_axis_cosine"]["point"] > 0.9,
          f"|cos|={m['moon_illum_axis_cosine']['point']:.3f}")

    # ---------------- Tier 4 ----------------
    print("\n-- Tier 4 (world map) --  planted: places at lat*u+lon*v; compass N-S=u, E-W=v")
    r4 = tier4(places, latlong, presence["directions"], n_boot=n_boot)
    m = r4["metrics"]
    check("T4 procrustes disparity~0", m["procrustes_disparity"]["point"] < 0.05,
          f"{m['procrustes_disparity']['point']:.4f}")
    check("T4 compass N-S aligns latitude", m["compass_NS_vs_lat_cos"]["point"] > 0.9,
          f"|cos|={m['compass_NS_vs_lat_cos']['point']:.3f}")
    check("T4 compass E-W aligns longitude", m["compass_EW_vs_lon_cos"]["point"] > 0.9,
          f"|cos|={m['compass_EW_vs_lon_cos']['point']:.3f}")

    # ---------------- Tier 5 ----------------
    print("\n-- Tier 5 (antipodal) --  planted: indoors=-outdoors; lovingness _|_ harmfulness; all 1-D")
    r5 = tier5(scalar, n_boot=n_boot)
    m = r5["metrics"]
    check("T5 indoors/outdoors cosine~-1", m["indoors|outdoors/cosine"]["point"] < -0.9,
          f"{m['indoors|outdoors/cosine']['point']:.3f}")
    ang = m["lovingness|harmfulness/angle_deg"]["point"]
    check("T5 loving/harm ~orthogonal", 80 < ang < 100, f"{ang:.1f}deg")
    for mem in ("indoors", "outdoors", "lovingness", "harmfulness"):
        check(f"T5 {mem} is 1-D", m[f"{mem}/top_pc_frac"]["point"] > 0.95,
              f"top-PC={m[f'{mem}/top_pc_frac']['point']:.3f}")

    # ---------------- artifacts ----------------
    results = {"tier1": r1, "tier2": r2, "tier3": r3, "tier4": r4, "tier5": r5}
    md = write_geometry_md(results)
    jsons = [r["json"] for r in results.values()]
    figs = [r["figure"] for r in results.values() if r.get("figure")]
    check("artifacts: 5 tier JSONs written", all(Path(j).exists() for j in jsons), jsons)
    check("artifacts: >=1 figure written", len(figs) >= 1 and all(Path(f).exists() for f in figs),
          f"{len(figs)} figures")

    print("\n=== VERDICTS ===")
    for t in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        print(f"[{t}] {results[t]['verdict']}\n")
    print(f"geometry.md -> {md}")
    print(f"figures -> {figs}")

    if fails:
        print(f"\nSMOKE FAILED ({len(fails)}): {fails}")
        return 1
    print(f"\nSMOKE PASSED: all planted structure recovered. "
          f"JSONs={len(jsons)} figures={len(figs)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Step-4 representation geometry (Tiers 1-5)")
    ap.add_argument("--smoke", action="store_true", help="run synthetic-data correctness check")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap iters for smoke")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(_smoke(n_boot=args.n_boot))
    ap.print_help()


if __name__ == "__main__":
    main()
