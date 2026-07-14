#!/usr/bin/env python3
"""Reconstruction sufficiency — do the stored int8 probe z-scores + frozen
geometry determine gemma-2-2b's concept-subspace residual component?

Claim (exact arithmetic). With v_c = w_c ⊘ nat_std_L, u_c = v_c/‖v_c‖,
κ_c = std2_c/‖v_c‖, the frozen pipeline (step-1 standardize → ridge → step-2
standardize) is affine in x, so
    ⟨x, u_c⟩ = z_c·κ_c + const_c,  const_c = (mu2_c − b_c)/‖v_c‖ + ⟨nat_mean, u_c⟩
holds as an identity. Stacking B = [u_c] (D×K), G = BᵀB:
    x̂_S = B G⁻¹ p  with  p_c = (z_c − z̄_c)·κ_c   equals   P_S(x − x̄)
(P_S = B G⁻¹ Bᵀ, x̄ = sample mean) — i.e. the stored scores reconstruct the
concept-subspace residual component EXACTLY, up to int8 quantization
(z-step 4/127 ≈ 0.0315σ, ±4σ clip). The absolute (uncentered) component
P_S x is likewise recovered via const_c, no sample mean needed.

Subcommands:
  cpu  ($0): independent κ re-derivation from the raw npz + store
       corpus_stats (no measure_loudness code paths) vs published loudness.json;
       per-concept λ layer-flatness; Gram conditioning (cond/eigenspectrum/
       effective rank/family collinearity); int8-noise propagation through G⁻¹
       (predicted reconstruction floor); loudness.json inequality audit;
       injection-packet invertibility numerics (site math from nanochat
       injection/sites.py replicated in numpy).
  pod  (GPU): fresh gemma-2-2b forwards on stored climbmix windows (loading +
       model conventions imported from measure_loudness). Reconstruct x̂_S from
       the STORED int8 scores, compare to directly-computed P_S(x−x̄):
       R²/cosine per layer, captured fraction of ‖x−x̄‖ (vs ℓ_tot²), observed vs
       predicted quantization floor, ±4σ clip-saturation impact.

Both modes merge their sections into out/reconstruction_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = Path(__file__).resolve().parent / "out"
LAYERS = [6, 8, 14]
D_MODEL = 2304
Z_STEP = 4.0 / 127.0          # int8 z quantization step (scale = 4·std2/127)


# --------------------------------------------------------------------------
# Independent geometry — deliberately NOT ProbeGeom (re-derivation mandate).
# --------------------------------------------------------------------------
class Geometry:
    def __init__(self, out_dir: Path = OUT_DIR):
        meta = json.load(open(out_dir / "probe_set.json"))
        self.concepts = list(meta["main_block_concepts"])   # store axis-2 order
        self.families = dict(meta["families"])
        self.K = len(self.concepts)
        assert list(meta["layers"]) == LAYERS
        arr = np.load(out_dir / "probe_set_arrays.npz")
        self.W = np.asarray(arr["W"], np.float64)            # [3,K,D] std-space
        self.b = np.asarray(arr["b"], np.float64)            # [3,K]
        self.nat_mean = np.asarray(arr["nat_mean"], np.float64)
        self.nat_std = np.asarray(arr["nat_std"], np.float64)
        V = self.W / self.nat_std[:, None, :]                # raw-space read dirs
        self.vnorm = np.linalg.norm(V, axis=2)               # [3,K]
        self.U = V / self.vnorm[:, :, None]                  # unit rows [3,K,D]
        self.G = np.einsum("lkd,ljd->lkj", self.U, self.U)   # [3,K,K]

    def attach_store(self, corpus_stats: dict, quant: dict):
        self.mu2 = np.asarray(corpus_stats["mean"], np.float64)   # [3,K]
        self.std2 = np.asarray(corpus_stats["std"], np.float64)   # [3,K]
        self.kappa = self.std2 / self.vnorm                       # [3,K]
        # affine constant: ⟨x,u_c⟩ = z_c·κ_c + const_c
        self.const = ((self.mu2 - self.b) / self.vnorm
                      + np.einsum("lkd,ld->lk", self.U, self.nat_mean))
        self.q_zero = np.asarray(quant["zero"], np.float64)[:3 * self.K].reshape(3, self.K)
        self.q_scale = np.asarray(quant["scale"], np.float64)[:3 * self.K].reshape(3, self.K)

    def z_from_x(self, x: np.ndarray, li: int) -> np.ndarray:
        """Frozen pipeline: x [N,D] raw -> z [N,K]."""
        xstd = (x - self.nat_mean[li]) / self.nat_std[li]
        raw = xstd @ self.W[li].T + self.b[li]
        return (raw - self.mu2[li]) / self.std2[li]

    def z_from_int8(self, q: np.ndarray, li: int) -> np.ndarray:
        """Store decode: int8 [N,K] -> z [N,K] (dequant then step-2 standardize)."""
        raw = q.astype(np.float64) * self.q_scale[li] + self.q_zero[li]
        return (raw - self.mu2[li]) / self.std2[li]

    def quantize_roundtrip(self, z: np.ndarray, li: int) -> np.ndarray:
        """Exact z -> int8 (store encode incl ±127 clip) -> z. Pure quantization."""
        raw = z * self.std2[li] + self.mu2[li]
        q = np.clip(np.round((raw - self.q_zero[li]) / self.q_scale[li]), -127, 127)
        return self.z_from_int8(q.astype(np.int8), li)


def load_report(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def write_report(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# cpu: conditioning + noise propagation + loudness audit + packet numerics
# --------------------------------------------------------------------------
def conditioning_section(geom: Geometry, loudness: dict) -> dict:
    out = {}
    rn = np.array([loudness["residual_norm"][str(L)]["p50"] for L in LAYERS])
    lam = geom.kappa / rn[:, None]
    # published-κ comparison (independent derivation vs artifact)
    kpub = np.array([loudness["ridge"]["kappa"][str(L)] for L in LAYERS])
    out["kappa_max_rel_diff_vs_published"] = float(np.max(np.abs(geom.kappa / kpub - 1)))
    # per-concept λ flatness (structural if ratios concentrate near 1)
    for tag, r in (("L8_over_L6", lam[1] / lam[0]), ("L14_over_L6", lam[2] / lam[0])):
        out[f"lambda_ratio_{tag}"] = {
            "median": float(np.median(r)), "iqr": [float(q) for q in np.percentile(r, [25, 75])],
            "range": [float(r.min()), float(r.max())],
            "outliers_low": [geom.concepts[i] for i in np.argsort(r)[:3]],
            "outliers_high": [geom.concepts[i] for i in np.argsort(r)[-3:]]}
    # Gram conditioning + noise propagation per layer
    dz = geom.q_scale / geom.std2                # z-space step (≈ 4/127)
    per_layer = {}
    for i, L in enumerate(LAYERS):
        G = geom.G[i]
        ev = np.linalg.eigvalsh(G)
        Ginv = np.linalg.inv(G)
        iu = np.triu_indices(geom.K, 1)
        offv = np.abs(G[iu])
        top = np.argsort(offv)[::-1][:5]
        fam_ids = {}
        for c, f in geom.families.items():
            if c in geom.concepts:
                fam_ids.setdefault(f, []).append(geom.concepts.index(c))
        fam_stats = {}
        for f, ids in sorted(fam_ids.items()):
            if len(ids) < 2:
                continue
            sub = np.abs(G[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)])
            fam_stats[f] = {"n": len(ids), "mean_abs_cos": float(sub.mean()),
                            "max_abs_cos": float(sub.max())}
        # int8 noise: ε_c ~ U(±δ_p/2), δ_p,c = κ_c·dz_c; E‖x̂_err‖² = Σ (G⁻¹)_cc σ_c²
        sig_p = geom.kappa[i] * dz[i] / np.sqrt(12.0)
        e_err2 = float(np.sum(np.diag(Ginv) * sig_p ** 2))
        naive2 = float(np.sum(sig_p ** 2))       # if U were orthonormal
        typ_sig = loudness["subspace_total"]["ridge"][str(L)]["p50"] * rn[i]
        amp = np.sqrt((Ginv ** 2) @ (sig_p ** 2)) / sig_p   # coeff-noise blowup
        per_layer[str(L)] = {
            "cond_G": float(ev[-1] / ev[0]),
            "eig_min": float(ev[0]), "eig_max": float(ev[-1]),
            "effective_rank": float(ev.sum() ** 2 / np.sum(ev ** 2)),
            "top_collinear_pairs": [
                [geom.concepts[iu[0][t]], geom.concepts[iu[1][t]], float(offv[t])] for t in top],
            "family_collinearity": fam_stats,
            "noise": {
                "predicted_rms_recon_err": float(np.sqrt(e_err2)),
                "rms_if_orthonormal": float(np.sqrt(naive2)),
                "typical_signal_norm_p50": float(typ_sig),
                "predicted_noise_over_signal": float(np.sqrt(e_err2) / typ_sig),
                "predicted_R2_floor": float(1 - e_err2 / typ_sig ** 2),
                "coeff_amplification_median": float(np.median(amp)),
                "coeff_amplification_max": [float(amp.max()), geom.concepts[int(np.argmax(amp))]],
            },
        }
    out["per_layer"] = per_layer
    out["z_step_median"] = float(np.median(dz))
    return out


def loudness_audit_section(geom: Geometry, loudness: dict) -> dict:
    """Inequalities that MUST hold: active p-quantiles ≥ all p-quantiles per
    concept (active ⇒ |z−z̄| large, z̄≈0); ℓ_tot quantiles ≥ any single concept's
    (|⟨y,u_c⟩| ≤ ‖P_S y‖ pointwise ⇒ quantile dominance)."""
    checks = {}
    for L in map(str, LAYERS):
        r = loudness["ridge"]
        act, allq = r["active_loudness"][L], r["all_loudness"][L]
        st = loudness["subspace_total"]["ridge"][L]
        checks[L] = {
            "active_p50_ge_all_p50_violations": int(np.sum(np.array(act["p50"]) < np.array(allq["p50"]))),
            "active_p95_ge_all_p95_violations": int(np.sum(np.array(act["p95"]) < np.array(allq["p95"]))),
            "subspace_ge_concept": {q: bool(st[q] >= max(allq[q])) for q in ("p50", "p95", "p99")},
            "n_active_min": int(min(r["active_loudness"][L]["n_active"])),
            "active_rate_median": float(np.median(r["active_loudness"][L]["n_active"])
                                        / loudness["corpus"]["n_tokens"]),
        }
    return {"concepts_match_store_order": loudness["concepts"] == geom.concepts,
            "per_layer": checks}


def packet_section(geom: Geometry, loudness: dict, n_embd=1280, seed=1337,
                   n_tok=4096, rng_seed=0) -> dict:
    """Numerics for the injection-side sufficiency claim: the site map
    z-scores → packet is (i) loudness-exact (‖Δx‖/‖x‖ == rms(gate) per token)
    and (ii) invertible up to a per-token positive scale (the z/rms(z) renorm).
    Replicates nanochat injection/sites.py forward + donor calibration."""
    import torch
    K = geom.K
    g_t = torch.Generator().manual_seed(seed)                    # sites.orthonormal_direction
    a0 = torch.randn(n_embd, K, generator=g_t, dtype=torch.float64)
    D = torch.linalg.qr(a0, mode="reduced")[0].t().numpy()       # [K,n_embd], D Dᵀ = I

    rng = np.random.default_rng(rng_seed)
    donor = np.asarray(loudness["ridge"]["active_loudness"]["8"]["p50"], np.float64)
    target = float(loudness["subspace_total"]["ridge"]["8"]["p50"])
    rms_c = np.abs(rng.normal(1.0, 0.3, K)) + 0.2                # plausible channel rms
    w = donor / rms_c                                            # calibrate_donor_gate
    gate = w * (target / np.sqrt(np.mean(w ** 2)))
    rms_gate = float(np.sqrt(np.mean(gate ** 2)))

    a = rng.standard_normal((n_tok, K)) * rms_c                  # activations (z-scores)
    x = rng.standard_normal((n_tok, n_embd)) * 3.0               # host residual
    # site forward (sites.py InjectionSite.forward, vector gate)
    overall = rms_gate
    a_scaled = a * (gate / overall)
    z = a_scaled @ D
    z_hat = z / np.sqrt(np.mean(z ** 2, axis=1, keepdims=True))
    dx = overall * np.sqrt(np.mean(x ** 2, axis=1, keepdims=True)) * z_hat
    loud = np.linalg.norm(dx, axis=1) / np.linalg.norm(x, axis=1)
    # inversion: Δx Dᵀ ⊘ (gate/overall) recovers a up to per-token positive scale
    a_rec = (dx @ D.T) / (gate / overall)
    scale = a_rec[:, :1] / a[:, :1]
    rec_err = np.abs(a_rec / scale - a) / np.maximum(np.abs(a), 1e-9)
    return {
        "n_embd": n_embd, "direction_seed": seed, "target": target,
        "rms_gate_minus_target": float(abs(rms_gate - target)),
        "per_token_loudness_max_abs_err": float(np.max(np.abs(loud - target))),
        "recover_scores_up_to_scale_max_rel_err": float(np.max(rec_err)),
        "per_token_scale_is_positive": bool(np.all(scale > 0)),
    }


def cmd_cpu(args):
    geom = Geometry()
    corpus_stats = _fetch_store_json(args.store, "corpus_stats.json")
    quant = _fetch_store_json(args.store, "quant.json")
    cols = _fetch_store_json(args.store, "columns.json")
    assert cols["concepts"] == geom.concepts, "store column order != main_block_concepts"
    geom.attach_store(corpus_stats, quant)
    loudness = json.load(open(args.loudness))

    rep = load_report(Path(args.out))
    rep["conditioning"] = conditioning_section(geom, loudness)
    rep["loudness_audit"] = loudness_audit_section(geom, loudness)
    rep["injection_packet"] = packet_section(geom, loudness)
    rep.setdefault("provenance", {})["cpu"] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "store": args.store}
    write_report(Path(args.out), rep)

    c = rep["conditioning"]
    print(f"[cpu] kappa max rel diff vs published: {c['kappa_max_rel_diff_vs_published']:.2e}")
    print(f"[cpu] lambda L14/L6 per-concept: median {c['lambda_ratio_L14_over_L6']['median']:.3f} "
          f"iqr {c['lambda_ratio_L14_over_L6']['iqr']}")
    for L in map(str, LAYERS):
        p = c["per_layer"][L]
        print(f"[cpu] L{L}: cond(G)={p['cond_G']:.1f} effrank={p['effective_rank']:.1f} "
              f"predicted floor R2={p['noise']['predicted_R2_floor']:.4f} "
              f"noise/signal={p['noise']['predicted_noise_over_signal']:.4f}")
    la = rep["loudness_audit"]
    viol = sum(v["active_p50_ge_all_p50_violations"] + v["active_p95_ge_all_p95_violations"]
               for v in la["per_layer"].values())
    sub_ok = all(all(v["subspace_ge_concept"].values()) for v in la["per_layer"].values())
    print(f"[cpu] loudness audit: order match={la['concepts_match_store_order']} "
          f"quantile violations={viol} subspace-dominance={'OK' if sub_ok else 'FAIL'}")
    pk = rep["injection_packet"]
    print(f"[cpu] packet: |rms(gate)-target|={pk['rms_gate_minus_target']:.2e} "
          f"per-token loudness err={pk['per_token_loudness_max_abs_err']:.2e} "
          f"score recovery (up-to-scale) rel err={pk['recover_scores_up_to_scale_max_rel_err']:.2e}")


def _fetch_store_json(loc: str, name: str) -> dict:
    if os.path.isdir(loc):
        return json.load(open(os.path.join(loc, name)))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(loc, name, repo_type="dataset")))


# --------------------------------------------------------------------------
# pod: empirical reconstruction from STORED int8 scores on fresh activations
# --------------------------------------------------------------------------
def _coeff_metrics(a: np.ndarray, c_ref: np.ndarray, G: np.ndarray, sig2: np.ndarray):
    """x̂ = B a vs reference P_Sy = B c_ref, all in coefficient space:
    err² = (a−c)ᵀG(a−c); ⟨x̂,P_Sy⟩ = aᵀGc; ‖x̂‖² = aᵀGa; ‖P_Sy‖² = sig2."""
    d = a - c_ref
    err2 = np.einsum("nk,kj,nj->n", d, G, d)
    dot = np.einsum("nk,kj,nj->n", a, G, c_ref)
    n2 = np.einsum("nk,kj,nj->n", a, G, a)
    cos = dot / np.maximum(np.sqrt(n2 * sig2), 1e-12)
    return {"R2_pooled": float(1 - err2.sum() / sig2.sum()),
            "cos_median": float(np.median(cos)), "cos_p05": float(np.percentile(cos, 5)),
            "rms_err": float(np.sqrt(err2.mean())),
            "err_median": float(np.median(np.sqrt(err2)))}


def cmd_pod(args):
    # loading/model conventions come from measure_loudness (frozen, gated).
    from measure_loudness import (load_store_meta, load_gemma, forward_windows,
                                  _sample_docs, spearman, WINDOW)
    geom = Geometry()
    meta = load_store_meta(args.store)
    assert meta["columns"]["concepts"] == geom.concepts, "store column order != main_block"
    geom.attach_store(meta["corpus_stats"], meta["quant"])
    loudness = json.load(open(args.loudness))

    shards = [int(s) for s in args.shards.split(",") if s.strip()]
    per_shard = max(1, args.n_docs // len(shards))
    rng = np.random.default_rng(args.seed)
    tok, model = load_gemma(args.device)

    xs = {L: [] for L in LAYERS}          # fp32 activations
    qs = []                               # stored int8 [n,3,K]
    n_tokens = 0
    t0 = time.time()
    for sid in shards:
        if n_tokens >= args.max_tokens:
            break
        docs, total = _sample_docs(args.store, sid, args.rows_per_shard, per_shard, rng)
        print(f"[pod] shard {sid}: {len(docs)} docs (of {total:,} rows)")
        win_items, doc_map = [], {}
        for k, (ids, stored) in enumerate(docs):
            nw = (len(ids) + WINDOW - 1) // WINDOW
            doc_map[k] = {"stored": stored, "nw": nw, "x": {}}
            for w in range(nw):
                win_items.append((k, w, ids[w * WINDOW:(w + 1) * WINDOW]))
        win_items.sort(key=lambda it: len(it[2]))
        for i in range(0, len(win_items), args.batch):
            batch = win_items[i:i + args.batch]
            outs = forward_windows(model, tok, [b[2] for b in batch], args.device)
            for (k, w, _), xo in zip(batch, outs):
                doc_map[k]["x"][w] = xo
        for k, dm in doc_map.items():
            if n_tokens >= args.max_tokens:
                break
            for L in LAYERS:
                xs[L].append(np.concatenate([dm["x"][w][L] for w in range(dm["nw"])], axis=0))
            qs.append(np.asarray(dm["stored"], np.int8))
            n_tokens += dm["stored"].shape[0]
        print(f"[pod]  running: {n_tokens:,} tokens, {n_tokens/max(time.time()-t0,1e-9):.0f} tok/s")

    q_all = np.concatenate(qs, axis=0)                     # [N,3,K]
    rep = load_report(Path(args.out))
    emp = {}
    for i, L in enumerate(LAYERS):
        x = np.concatenate(xs[L], axis=0).astype(np.float64)   # [N,D]
        N = x.shape[0]
        U, G = geom.U[i], geom.G[i]
        xbar = x.mean(axis=0)
        comp_u = x @ U.T                                   # ⟨x,u_c⟩ absolute
        comp = comp_u - xbar @ U.T                         # Uᵀ(x−x̄)
        xnorm = np.linalg.norm(x, axis=1)
        ynorm2 = xnorm ** 2 - 2 * (x @ xbar) + xbar @ xbar

        # (1) affine identity on fresh activations: z_exact·κ + const == ⟨x,u⟩
        z_ex = geom.z_from_x(x, i)
        p_abs_ex = z_ex * geom.kappa[i] + geom.const[i]
        id_rel = np.abs(p_abs_ex - comp_u) / np.maximum(np.abs(comp_u), 1e-9)

        # reference coefficients of P_S(x−x̄) and P_S x
        c_ref = np.linalg.solve(G, comp.T).T
        c_ref_abs = np.linalg.solve(G, comp_u.T).T
        sig2 = np.einsum("nk,nk->n", comp, c_ref)          # ‖P_S(x−x̄)‖²
        sig2_abs = np.einsum("nk,nk->n", comp_u, c_ref_abs)

        # (2) stored int8 -> z -> reconstruction (centered + absolute)
        z_st = geom.z_from_int8(q_all[:, i, :], i)
        p_st = (z_st - z_st.mean(axis=0)) * geom.kappa[i]
        a_st = np.linalg.solve(G, p_st.T).T
        m_st = _coeff_metrics(a_st, c_ref, G, sig2)
        a_abs = np.linalg.solve(G, (z_st * geom.kappa[i] + geom.const[i]).T).T
        m_abs = _coeff_metrics(a_abs, c_ref_abs, G, sig2_abs)

        # (3) pure-quantization floor: round-trip my OWN exact z through int8
        z_rt = geom.quantize_roundtrip(z_ex, i)
        p_rt = (z_rt - z_rt.mean(axis=0)) * geom.kappa[i]
        m_rt = _coeff_metrics(np.linalg.solve(G, p_rt.T).T, c_ref, G, sig2)

        # (4) saturation (±4σ clip): stored |int8| == 127 anywhere on the token
        sat_tok = np.any(np.abs(q_all[:, i, :]) == 127, axis=1)
        sat = {"entry_frac": float(np.mean(np.abs(q_all[:, i, :]) == 127)),
               "token_frac": float(sat_tok.mean()), "n_tokens": int(sat_tok.sum())}
        if sat_tok.any():
            sat["saturated"] = _coeff_metrics(a_st[sat_tok], c_ref[sat_tok], G, sig2[sat_tok])
            sat["unsaturated"] = _coeff_metrics(a_st[~sat_tok], c_ref[~sat_tok], G, sig2[~sat_tok])

        # (5) captured fraction + ℓ_tot² check + stored-z crosscheck on new shards
        ell = np.sqrt(sig2) / xnorm
        sp = min(spearman(z_ex[:, c], z_st[:, c]) for c in range(geom.K))
        pred = rep.get("conditioning", {}).get("per_layer", {}).get(str(L), {}).get("noise", {})
        emp[str(L)] = {
            "n_tokens": N,
            "identity_rel_err_median": float(np.median(id_rel)),
            "stored_z": {"centered": m_st, "absolute": m_abs},
            "roundtrip_quant_floor": m_rt,
            "predicted_rms_err": pred.get("predicted_rms_recon_err"),
            "captured": {
                "pooled_frac_of_centered_var": float(sig2.sum() / ynorm2.sum()),
                "per_token_median_frac_ynorm": float(np.median(np.sqrt(sig2 / ynorm2))),
                "ell_tot_p50_here": float(np.median(ell)),
                "ell_tot_p50_loudness_json": loudness["subspace_total"]["ridge"][str(L)]["p50"],
            },
            "saturation": sat,
            "z_crosscheck": {"spearman_min": float(sp),
                             "median_abs_dz_max": float(max(np.median(np.abs(z_ex[:, c] - z_st[:, c]))
                                                            for c in range(geom.K)))},
        }
        print(f"[pod] L{L}: R2={m_st['R2_pooled']:.5f} cos_med={m_st['cos_median']:.5f} "
              f"rms_err={m_st['rms_err']:.3f} (pure-quant {m_rt['rms_err']:.3f}, "
              f"predicted {pred.get('predicted_rms_recon_err', float('nan')):.3f}) "
              f"sat_tok={sat['token_frac']:.3%}")
        del x

    rep["empirical"] = emp
    rep.setdefault("provenance", {})["pod"] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "store": args.store,
        "shards": shards, "n_tokens": n_tokens, "device": args.device,
        "seed": args.seed}
    write_report(Path(args.out), rep)
    print(f"[pod] report -> {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cpu", help="conditioning + noise propagation + audits ($0)")
    c.add_argument("--store", default="kaushikreddyxyz/climbmix-scored")
    c.add_argument("--loudness", default=str(OUT_DIR / "loudness.json"))
    c.add_argument("--out", default=str(OUT_DIR / "reconstruction_report.json"))
    c.set_defaults(fn=cmd_cpu)

    p = sub.add_parser("pod", help="GPU: empirical reconstruction from stored int8")
    p.add_argument("--store", default="kaushikreddyxyz/climbmix-scored")
    p.add_argument("--shards", default="3,13,23", help="fresh shards (loudness run used 2,12,22)")
    p.add_argument("--n-docs", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=120_000)
    p.add_argument("--rows-per-shard", type=int, default=500_000)
    p.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--loudness", default=str(OUT_DIR / "loudness.json"))
    p.add_argument("--out", default=str(OUT_DIR / "reconstruction_report.json"))
    p.set_defaults(fn=cmd_pod)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
