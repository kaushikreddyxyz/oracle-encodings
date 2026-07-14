"""Unit tests for measure_loudness.py — no network, no gemma weights.

Constructs a synthetic probe geometry (small d, few concepts) with known
w/nat_std/std2 and activations, and asserts:
  * the affine IDENTITY (5): |⟨x,u_c⟩ − m_c| == |z_c − z̄_c|·κ_c to fp64 precision;
  * subspace math on constructed low-rank data;
  * active-conditioning (z_c ≥ 2 masking);
  * schema round-trip + analytic/measure field merge.
"""
import json

import numpy as np
import pytest

import measure_loudness as ml


# --------------------------------------------------------------------------
# Synthetic geometry: d=6, K=3, one "layer". Mirrors ProbeGeom's raw-space math
# (v = w/nat_std, u = v/‖v‖, κ = std2/‖v‖) without the file/npz plumbing.
# --------------------------------------------------------------------------
def _synth(seed=0, d=6, K=3, N=4000):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(K, d))                 # std-space probe weights
    b = rng.normal(size=K)
    nat_mean = rng.normal(size=d)
    nat_std = rng.uniform(0.5, 3.0, size=d)     # strictly positive
    mu2 = rng.normal(size=K)                     # corpus mean of raw scores
    std2 = rng.uniform(0.5, 2.0, size=K)         # corpus std of raw scores
    x = rng.normal(size=(N, d)) * 4.0 + 2.0      # raw activations

    v = w / nat_std                              # [K,d] raw-space read dir
    vnorm = np.linalg.norm(v, axis=1)
    u = v / vnorm[:, None]
    kappa = std2 / vnorm

    xstd = (x - nat_mean) / nat_std
    raw = xstd @ w.T + b                         # [N,K]
    z = (raw - mu2) / std2
    comp = x @ u.T                               # [N,K] ⟨x,u_c⟩
    xnorm = np.linalg.norm(x, axis=1)
    return dict(x=x, w=w, b=b, nat_mean=nat_mean, nat_std=nat_std, mu2=mu2,
                std2=std2, v=v, u=u, kappa=kappa, z=z, comp=comp, xnorm=xnorm)


def test_affine_identity_fp64():
    """(⟨x,u_c⟩ − m_c) == (z_c − z̄_c)·κ_c exactly (the self-audit gate)."""
    s = _synth()
    comp, z, kappa = s["comp"], s["z"], s["kappa"]
    lhs = comp - comp.mean(axis=0)
    rhs = (z - z.mean(axis=0)) * kappa
    assert np.allclose(lhs, rhs, atol=1e-9, rtol=0), \
        f"identity broke: max abs diff {np.abs(lhs - rhs).max():.2e}"
    # the ratio-form the measure gate reports: median rel-err ~ 0
    denom = np.maximum(np.abs(lhs), 1e-9)
    rel = np.abs(lhs - rhs) / denom
    assert float(np.median(rel)) < 1e-6


def test_loudness_stats_and_active_conditioning():
    """ℓ_c = |comp − m_c|/‖x‖; active mask z ≥ 2 selects a subset with its own
    quantiles; n_active matches the mask count."""
    s = _synth()
    m, all_p, act_p = ml.loudness_stats(s["comp"].astype(np.float32),
                                        s["xnorm"].astype(np.float64),
                                        s["z"], active_thresh=2.0)
    assert np.allclose(m, s["comp"].mean(axis=0), atol=1e-4)
    for c in range(s["comp"].shape[1]):
        n_expected = int((s["z"][:, c] >= 2.0).sum())
        assert act_p["n_active"][c] == n_expected
        # all-token p95 >= p50 >= 0
        assert all_p["p95"][c] >= all_p["p50"][c] >= 0.0
    # a concept that never fires >=2 gets zeroed active quantiles
    s2 = _synth(seed=7)
    zcap = np.minimum(s2["z"], 1.0)               # force no active tokens
    _, _, act = ml.loudness_stats(s2["comp"].astype(np.float32),
                                  s2["xnorm"], zcap, active_thresh=2.0)
    assert all(n == 0 for n in act["n_active"])
    assert all(p == 0.0 for p in act["p50"])


def test_subspace_total_lowrank():
    """On data confined to a rank-r subspace spanned by the concept directions,
    ℓ_tot recovers the full centered-norm ratio; QR basis is orthonormal."""
    rng = np.random.default_rng(1)
    d, K, N = 8, 3, 3000
    U = np.linalg.qr(rng.normal(size=(d, K)))[0]          # d×K orthonormal cols
    coeffs = rng.normal(size=(N, K)) * 3.0
    x = coeffs @ U.T + 5.0 * U[:, 0]                       # lives in span(U)
    xnorm = np.linalg.norm(x, axis=1)
    Q = np.linalg.qr(U)[0]
    assert np.allclose(Q.T @ Q, np.eye(K), atol=1e-9)
    Qproj = x @ Q
    res = ml.subspace_total(Qproj, xnorm)
    # since x lies entirely in span(Q), ‖Qᵀ(x−x̄)‖ == ‖x−x̄‖ (x̄ also in span)
    centered = x - x.mean(axis=0)
    ell_true = np.linalg.norm(centered, axis=1) / xnorm
    assert abs(res["p50"] - float(np.percentile(ell_true, 50))) < 1e-6
    assert res["p99"] >= res["p95"] >= res["p90"] >= res["p50"] >= 0.0


def test_spearman_and_rankdata():
    rng = np.random.default_rng(2)
    a = rng.normal(size=500)
    assert abs(ml.spearman(a, a) - 1.0) < 1e-9
    assert abs(ml.spearman(a, -a) + 1.0) < 1e-9
    # monotone transform -> spearman 1
    assert abs(ml.spearman(a, np.exp(a)) - 1.0) < 1e-9
    # ties handled (average ranks)
    r = ml._rankdata(np.array([1.0, 1.0, 2.0]))
    assert np.allclose(r, [1.5, 1.5, 3.0])


def test_schema_roundtrip_and_merge(tmp_path):
    """analytic writes κ + null empirical fields; measure fills them; the merge
    keeps κ and overwrites the empirical fields. Round-trips through JSON."""
    class _Geom:
        concepts = ["a", "b", "c"]
    geom = _Geom()
    schema = ml._empty_schema(geom, model="test-model")
    assert schema["version"] == 1 and schema["concepts"] == ["a", "b", "c"]
    assert schema["residual_norm"]["6"] is None
    assert schema["ridge"]["kappa"]["8"] is None

    p = tmp_path / "loudness.json"
    # analytic pass: κ only
    schema["ridge"]["kappa"]["6"] = [1.0, 2.0, 3.0]
    ml.write_json(p, schema)
    reloaded = ml.load_or_init(p, geom)
    assert reloaded["ridge"]["kappa"]["6"] == [1.0, 2.0, 3.0]
    assert reloaded["residual_norm"]["6"] is None

    # measure pass merges into the same file (κ preserved, empirical filled)
    reloaded["residual_norm"]["6"] = {"mean": 10.0, "p50": 9.0}
    reloaded["ridge"]["loudness_per_sigma"]["6"] = [0.11, 0.22, 0.33]
    reloaded["crosscheck"]["identity_median_rel_err"] = 1e-7
    ml.write_json(p, reloaded)
    final = json.load(open(p))
    assert final["ridge"]["kappa"]["6"] == [1.0, 2.0, 3.0]      # analytic survived
    assert final["residual_norm"]["6"]["p50"] == 9.0
    assert final["crosscheck"]["identity_median_rel_err"] == 1e-7


def test_quantile_dict():
    x = np.arange(101, dtype=np.float64)          # 0..100
    q = ml.quantile_dict(x)
    assert set(q) == set(ml.QUANTS)
    assert abs(q["p50"] - 50.0) < 1e-9
    assert abs(q["p05"] - 5.0) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
