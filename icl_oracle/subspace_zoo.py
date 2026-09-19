"""Subspace zoo: which 2-D planes accept injected grid coordinates, and how do they relate to the model's own
(in-context) plane and to the QK-derived port?

Every candidate is a plane U (2 x d). Node j gets F_j = Z2_j @ U (Z2 = spectral grid coordinates), rescaled so the
mean node-vector norm equals norm_ref (same as the QK port), times beta, added at hidden state 26 at every occurrence
of node j. Readouts (paired over the same 13 contexts for every candidate):
  uptake   rule-following mass at tokens 100-300 (early) and 400-1200 (late), minus no-injection
  override coordinates of a PERMUTED map: mass on injected-map neighbors minus chance (tokens 200-600)
  unseen   node X absent from a 600-token context, then queried once: P(true neighbors of X) with X's true
           coordinate written, and P(neighbors of far node Y) when X is given Y's coordinate (claim-Y)
  self     mass on the current node's own token (direct-path / unembedding leakage diagnostic)
Geometry per candidate: cos of principal angles vs each context's own plane and vs the QK port; QK score
u^T S u / lambda_max(S) for the late-reader form S and the all-late-heads form; fast-RoPE leakage; alignment with the
context's 16 word embeddings; energy inside the top-64 vocabulary PCs / top-64 activation PCs.

Families: QK ports (late readers, anti, mid, early, all of L26, all late heads), token-embedding PCs of the 16 words,
vocabulary PCs (centered PCA and uncentered SVD, top to bottom of the spectrum), QK form restricted to the vocabulary /
16-word / activation subspaces, activation PCs at hs26 from a foreign word set, foreign-set own plane (E2 arm),
MLP-input and attention-output singular directions at layers 25/26, random plane, calibrated own plane.

stages: zoo (all candidates, beta 2 and 4), sweep (beta 1..8 for the best grid-agnostic candidates + references).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import (Lab, random_walk, cand_map, nanfill, fit_M, top_pcs, principal_angles, random_features,
                       far_node, T_LONG, T_UNSEEN, N, DEV)
from port_inject import forward_multi, delta_window, plane_of
from qk_port import build_port, port_features, rope_slow_dims, qk_bilinear, OUT as OUT_QK, NH, NKV, HD

OUT = os.environ.get("ICL_OUT_ZOO", os.path.join(HERE, "results_zoo")); os.makedirs(OUT, exist_ok=True)
H_MAIN = 26
NREF = [None]
EARLY, LATE = (100, 300), (400, 1200)
OV = (200, 600)


# ----------------------------------------------------------------------------- QK forms
@torch.no_grad()
def qk_forms(lab, heads, min_wavelength=5000, fast_penalty=1.0):
    """S = sum_h sym(B_h)/||sym(B_h)|| on RoPE-slow dims; Gn = normalized fast-RoPE Gram + I (same as build_port)."""
    dims, _ = rope_slow_dims(lab, min_wavelength); fast = [i for i in range(HD) if i not in set(dims)]
    S = torch.zeros((lab.d, lab.d), device=DEV); G = torch.zeros((lab.d, lab.d), device=DEV)
    for l, h in heads:
        B = qk_bilinear(lab, l, h, dims); Sh = (B + B.T) / 2; S += Sh / Sh.norm()
        attn = lab.model.model.layers[l].self_attn; ln = (1.0 + lab.model.model.layers[l].input_layernorm.weight.detach().float())
        Wqf = attn.q_proj.weight.detach().float()[h * HD:(h + 1) * HD][fast] * ln[None]
        Wkf = attn.k_proj.weight.detach().float()[(h // (NH // NKV)) * HD:(h // (NH // NKV) + 1) * HD][fast] * ln[None]
        Gh = Wqf.T @ Wqf + Wkf.T @ Wkf; G += Gh / Gh.norm()
    Gn = G * (lab.d / torch.trace(G)) * fast_penalty + torch.eye(lab.d, device=DEV)
    return S, Gn


@torch.no_grad()
def top_dirs(S, Gn, k=2, sign=+1, Q=None):
    """top-k (sign>0) or bottom-k (sign<0) generalized eigenvectors of (S, Gn); optionally restricted to the column
    space of an orthonormal Q [d,m]. Returns U [k,d] (unit rows) and the eigenvalues."""
    if Q is not None:
        S = Q.T @ S @ Q; Gn = Q.T @ Gn @ Q
    Lc = torch.linalg.cholesky(Gn); Li = torch.linalg.inv(Lc)
    evals, evecs = torch.linalg.eigh(Li @ S @ Li.T); V = Li.T @ evecs
    n = S.shape[0]; cols = list(range(n - k, n)) if sign > 0 else list(range(k))
    U = V[:, cols].T
    if Q is not None: U = U @ Q.T
    U = U / U.norm(dim=1, keepdim=True)
    return U.cpu().numpy().astype(np.float32), evals[cols].cpu().numpy().tolist()


_LAM = {}
def qk_score(S, U):
    """mean over the plane's orthonormal basis of u^T S u, divided by lambda_max(S) (1 = the best single direction)."""
    Q, _ = np.linalg.qr(np.asarray(U, np.float64).T); Qt = torch.tensor(Q, device=DEV, dtype=S.dtype)
    if id(S) not in _LAM: _LAM[id(S)] = torch.linalg.eigvalsh(S)[-1].item()
    lam = _LAM[id(S)]
    return float(torch.diag(Qt.T @ S @ Qt).mean().item() / lam)


@torch.no_grad()
def leak_ratio(lab, heads, U, min_wavelength=5000):
    dims, _ = rope_slow_dims(lab, min_wavelength); fast = [i for i in range(HD) if i not in set(dims)]
    Ut = torch.tensor(U, device=DEV); r = []
    for l, h in heads:
        attn = lab.model.model.layers[l].self_attn; ln = (1.0 + lab.model.model.layers[l].input_layernorm.weight.detach().float())
        Wq = attn.q_proj.weight.detach().float()[h * HD:(h + 1) * HD] * ln[None]
        r.append(((Wq[fast] @ Ut.T).norm(dim=0) / (Wq[dims] @ Ut.T).norm(dim=0)).mean().item())
    return float(np.mean(r))


# ----------------------------------------------------------------------------- other subspaces
@torch.no_grad()
def vocab_pcs(lab):
    """centered PCA of the whole input-embedding table (rows = PCs, descending) + uncentered top SVD directions."""
    W = lab.model.model.embed_tokens.weight.detach(); V_, d = W.shape
    mu = W.float().mean(0); C = torch.zeros((d, d), device=DEV)
    for i in range(0, V_, 16384):
        X = W[i:i + 16384].float() - mu; C += X.T @ X
    evals, evecs = torch.linalg.eigh(C); P = evecs.flip(1).T
    ev2, evec2 = torch.linalg.eigh(C + V_ * torch.outer(mu, mu)); P2 = evec2.flip(1).T
    return P.cpu().numpy().astype(np.float32), evals.flip(0).cpu().numpy(), P2[:4].cpu().numpy().astype(np.float32)


def act_pcs(lab, set_id=0, seed=0, h=H_MAIN):
    """PCs of hs-h activations over all node tokens of a 1200-token walk on a FOREIGN word set (BOS excluded)."""
    inst = lab.instance(set_id, seed); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(424242 + seed))
    ids, seg, node = lab.build_context([(inst, walk)]); cand, _ = cand_map([inst])
    _, hs = lab.forward(ids, cand, hs_layers=(h,)); H = hs[h][1:].astype(np.float64)
    Hc = H - H.mean(0); _, sv, Vt = np.linalg.svd(Hc, full_matrices=False)
    P, Hn = plane_of(lab, hs[h], seg, node, 0); M, _ = fit_M(Hn, lab.Z2)
    return Vt.astype(np.float32), (sv ** 2) / (sv ** 2).sum(), M.astype(np.float32)


@torch.no_grad()
def weight_dirs(lab):
    l = H_MAIN
    W = lab.model.model.layers[l].mlp.gate_proj.weight.detach().float() * (1.0 + lab.model.model.layers[l].pre_feedforward_layernorm.weight.detach().float())[None]
    Vt = torch.linalg.svd(W, full_matrices=False)[2]
    Wo = lab.model.model.layers[l - 1].self_attn.o_proj.weight.detach().float()
    Uo = torch.linalg.svd(Wo, full_matrices=False)[0]
    return Vt[:2].cpu().numpy(), Uo[:, :2].T.cpu().numpy()


def emb16(lab, inst):
    W = lab.model.model.embed_tokens.weight.detach().float()
    E = W[torch.tensor(inst["ids"], device=DEV)].cpu().numpy().astype(np.float64)
    return E


def orth(U):
    Q, _ = np.linalg.qr(np.asarray(U, np.float64).T); return Q.T.astype(np.float32)


def energy_in(U, P):
    """fraction of the plane's energy inside the row space of P (orthonormal rows)."""
    Q = orth(U); return float(np.sum((Q @ P.T) ** 2) / Q.shape[0])


# ----------------------------------------------------------------------------- candidates
def agnostic_candidates(lab, meta):
    """candidates that do not depend on the word set. Returns name -> dict(U, group, extra)."""
    late = [tuple(x) for x in meta["late"][:4]]; mid = [tuple(x) for x in meta["mid"][:4]]
    C = {}
    def add(name, U, group, **kw): C[name] = dict(U=np.asarray(U, np.float32)[:2], group=group, **kw)
    S_late, Gn_late = qk_forms(lab, late)
    U_late, ev = top_dirs(S_late, Gn_late, 2, +1); add("qk_late", U_late, "qk", eig=ev)
    U_chk, _, _ = build_port(lab, late, k=2, fast_penalty=1.0)
    print(f"[zoo] top_dirs vs build_port agreement (cos principal angles): {principal_angles(U_late, U_chk)}")
    U_anti, ev = top_dirs(S_late, Gn_late, 2, -1); add("qk_anti", U_anti, "qk", eig=ev)
    S_mid, Gn_mid = qk_forms(lab, mid); U_mid, ev = top_dirs(S_mid, Gn_mid, 2, +1); add("qk_mid4", U_mid, "qk", eig=ev)
    rng = np.random.default_rng(11); early = [(int(l), int(h)) for l, h in zip(rng.integers(2, 13, 4), rng.integers(0, NH, 4))]
    S_e, Gn_e = qk_forms(lab, early); U_e, ev = top_dirs(S_e, Gn_e, 2, +1); add("qk_early4", U_e, "qk", eig=ev, heads=early)
    S_26, Gn_26 = qk_forms(lab, [(H_MAIN, h) for h in range(NH)]); U_26, ev = top_dirs(S_26, Gn_26, 2, +1); add("qk_L26all", U_26, "qk", eig=ev)
    all_late = [(l, h) for l in range(H_MAIN, lab.n_layers) for h in range(NH)]
    S_all, Gn_all = qk_forms(lab, all_late); U_all, ev = top_dirs(S_all, Gn_all, 2, +1); add("qk_alllate", U_all, "qk", eig=ev)
    U_all_anti, ev = top_dirs(S_all, Gn_all, 2, -1); add("qk_alllate_anti", U_all_anti, "qk", eig=ev)
    # vocabulary
    P, evv, P_svd = vocab_pcs(lab)
    for a, b in ((1, 2), (2, 3), (3, 4), (5, 6), (10, 11), (20, 21), (50, 51), (100, 101), (300, 301), (1000, 1001), (3583, 3584)):
        add(f"vocab_pc{a}_{b}", P[[a - 1, b - 1]], "vocab", var=[float(evv[a - 1] / evv.sum()), float(evv[b - 1] / evv.sum())])
    add("vocab_svd1_2", P_svd[:2], "vocab")
    for k in (16, 64, 256, 1024):
        Q = torch.tensor(P[:k].T, device=DEV); U, ev = top_dirs(S_late, Gn_late, 2, +1, Q=Q); add(f"vocab_qk{k}", U, "vocab_qk", eig=ev)
    # activations on a foreign word set
    Pa, vara, M0 = act_pcs(lab)
    for a, b in ((1, 2), (2, 3), (3, 4), (5, 6), (10, 11), (20, 21), (50, 51)):
        add(f"act_pc{a}_{b}", Pa[[a - 1, b - 1]], "act", var=[float(vara[a - 1]), float(vara[b - 1])])
    Q = torch.tensor(Pa[:64].T, device=DEV); U, ev = top_dirs(S_late, Gn_late, 2, +1, Q=Q); add("act_qk64", U, "act", eig=ev)
    add("foreign_set0", orth(M0), "act", M=M0)                       # E2 arm: plane fitted on word set 0 (features use M0)
    # weights
    Vm, Uo = weight_dirs(lab); add("mlp_in26", Vm, "weight"); add("attn_out25", Uo, "weight")
    # random
    Q, _ = np.linalg.qr(np.random.default_rng(321).standard_normal((lab.d, 2))); add("rand", Q.T, "ref")
    Q, _ = np.linalg.qr(np.random.default_rng(322).standard_normal((lab.d, 2))); add("rand2", Q.T, "ref")
    forms = dict(S_late=S_late, Gn_late=Gn_late, S_all=S_all, P_vocab64=P[:64], P_act64=Pa[:64], late=late)
    return C, forms


def context_candidates(lab, inst, forms):
    """candidates that depend on the word set (but need no forward pass / calibration)."""
    E = emb16(lab, inst); Ec = E - E.mean(0); Vt = top_pcs(Ec, 15)
    C = {}
    for a, b in ((1, 2), (2, 3), (1, 3), (1, 4), (3, 4), (5, 6), (8, 9), (12, 13)):
        C[f"emb16_pc{a}_{b}"] = dict(U=Vt[[a - 1, b - 1]].astype(np.float32), group="emb16")
    Q = torch.tensor(Vt.T.astype(np.float32), device=DEV); U, ev = top_dirs(forms["S_late"], forms["Gn_late"], 2, +1, Q=Q)
    C["emb16_qk"] = dict(U=U, group="emb16", eig=ev)
    Q = torch.tensor(top_pcs(E, 16).T.astype(np.float32), device=DEV); U, ev = top_dirs(forms["S_late"], forms["Gn_late"], 2, +1, Q=Q)
    C["emb16_qk_raw"] = dict(U=U, group="emb16", eig=ev)
    return C, E


def features_of(lab, name, c, nref):
    if "M" in c:                                   # fitted planes keep their fitted scale (comparable with E2 / port3 own_b2)
        return (lab.Z2 @ c["M"].astype(np.float64)).astype(np.float32)
    return port_features(lab, c["U"], nref)


def geometry(lab, name, c, forms, P_own, E):
    U = c["U"]; g = dict(cos_own=float(np.mean(principal_angles(U, P_own))), cos_own_max=float(np.max(principal_angles(U, P_own))),
                         qk_late=qk_score(forms["S_late"], U), qk_all=qk_score(forms["S_all"], U), leak=leak_ratio(lab, forms["late"], U),
                         in_vocab64=energy_in(U, forms["P_vocab64"]), in_act64=energy_in(U, forms["P_act64"]))
    En = E / np.linalg.norm(E, axis=1, keepdims=True); Q = orth(U)
    g["emb_align"] = float(np.abs(Q @ En.T).mean())                     # mean |cos| between plane axes and the 16 word embeddings
    g["in_emb16"] = energy_in(U, top_pcs(E - E.mean(0), 15))
    if "M" in c: g["norm_ratio"] = float(np.linalg.norm(lab.Z2 @ c["M"].astype(np.float64), axis=1).mean() / NREF[0])
    return g


# ----------------------------------------------------------------------------- readouts
def run_context(lab, set_id, s, wseed, cands, forms, nref, betas, perm, res, n_unseen=2, tag="zoo", with_ctx=True):
    NREF[0] = nref
    inst = lab.instance(set_id, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(wseed))
    segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
    inv = np.argsort(perm)
    def inj_nbrs(sg, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    # own plane from a separate calibration walk (same as own_plane_features) + its PC basis
    walk_c = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s)); ids_c, seg_c, node_c = lab.build_context([(inst, walk_c)])
    _, hs_c = lab.forward(ids_c, cand, hs_layers=(H_MAIN,)); P_own, Hc = plane_of(lab, hs_c[H_MAIN], seg_c, node_c, 0); M_own, _ = fit_M(Hc, lab.Z2)
    ctx_c, E = context_candidates(lab, inst, forms)
    allc = dict(cands)
    if with_ctx: allc.update(ctx_c)
    allc["own"] = dict(U=orth(M_own), group="ref", M=M_own.astype(np.float32))
    key = f"set{set_id}_s{s}"
    res["geometry"][key] = {n: geometry(lab, n, c, forms, P_own, E) for n, c in allc.items()}
    res["geometry"][key]["_cos_pairs"] = {n: {m: float(np.mean(principal_angles(allc[n]["U"], allc[m]["U"]))) for m in ("qk_late", "qk_alllate", "own", "vocab_pc1_2", "act_pc1_2") if m in allc} for n in allc}
    def self_mass(probs, lo, hi):
        pos = np.nonzero(seg == 0)[0][lo:hi]; return float(np.mean([probs[p, col[int(inst["ids"][node[p]])]] for p in pos]))
    probs0, _ = forward_multi(lab, ids, cand, [], ()); m0 = lab.seg_masses(probs0, seg, node, segs, col)[0]; ch = lab.seg_masses(probs0, seg, node, segs, col, inj_nbrs)[0]
    res["curves"][f"{key}|none"] = m0.tolist()
    res["rows"].append(dict(ctx=key, set=set_id, seed=s, cand="none", beta=0.0, group="ref", up_early=float(m0[EARLY[0]:EARLY[1]].mean()), up_late=float(m0[LATE[0]:LATE[1]].mean()),
                            self_late=self_mass(probs0, *LATE), ov_true=float(m0[OV[0]:OV[1]].mean()), ov_inj=float(ch[OV[0]:OV[1]].mean()), ov_chance=float(ch[OV[0]:OV[1]].mean())))
    F_all = {n: features_of(lab, n, c, nref) for n, c in allc.items()}
    for n, c in allc.items():
        for b in betas:
            F = F_all[n] * b
            probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, F, 0, 0, T_LONG), H_MAIN)], ()); m = lab.seg_masses(probs, seg, node, segs, col)[0]
            probs_o, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, F[perm], 0, 0, T_LONG), H_MAIN)], ())
            mt = lab.seg_masses(probs_o, seg, node, segs, col)[0]; mi = lab.seg_masses(probs_o, seg, node, segs, col, inj_nbrs)[0]
            res["curves"][f"{key}|{n}|b{b:g}"] = m.tolist()
            res["rows"].append(dict(ctx=key, set=set_id, seed=s, cand=n, beta=b, group=c["group"], up_early=float(m[EARLY[0]:EARLY[1]].mean()), up_late=float(m[LATE[0]:LATE[1]].mean()),
                                    self_late=self_mass(probs, *LATE), ov_true=float(mt[OV[0]:OV[1]].mean()), ov_inj=float(mi[OV[0]:OV[1]].mean()), ov_chance=float(ch[OV[0]:OV[1]].mean())))
    rngu = np.random.default_rng(8000 + s + 100 * set_id)
    for X in rngu.choice(N, n_unseen, replace=False):
        X = int(X); Y = far_node(lab, X, rngu)
        walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rngu, avoid=X), [X]]); segs_u = [(inst, walk_q)]
        ids_u, seg_u, node_u = lab.build_context(segs_u); cand_u, col_u = cand_map([inst])
        def masses(probs):
            p = probs[-1]; return dict(P_nbrs_X=float(sum(p[col_u[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col_u[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
        res["unseen"].append(dict(ctx=key, set=set_id, seed=s, X=X, Y=Y, cand="none", beta=0.0, group="ref", **masses(forward_multi(lab, ids_u, cand_u, [], ())[0])))
        for n, c in allc.items():
            for b in betas:
                F = F_all[n] * b; Fy = F.copy(); Fy[X] = F[Y]
                tx = masses(forward_multi(lab, ids_u, cand_u, [(delta_window(lab, seg_u, node_u, F, 0, 0, T_UNSEEN + 1), H_MAIN)], ())[0])
                cy = masses(forward_multi(lab, ids_u, cand_u, [(delta_window(lab, seg_u, node_u, Fy, 0, 0, T_UNSEEN + 1), H_MAIN)], ())[0])
                res["unseen"].append(dict(ctx=key, set=set_id, seed=s, X=X, Y=Y, cand=n, beta=b, group=c["group"], trueX_X=tx["P_nbrs_X"], trueX_Y=tx["P_nbrs_Y"], claimY_X=cy["P_nbrs_X"], claimY_Y=cy["P_nbrs_Y"]))
    # console summary for this context
    r0 = [r for r in res["rows"] if r["ctx"] == key and r["cand"] == "none"][0]
    u0 = np.mean([u["P_nbrs_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == "none"])
    def summ(n, b):
        r = [r for r in res["rows"] if r["ctx"] == key and r["cand"] == n and r["beta"] == b][0]
        ux = np.mean([u["trueX_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == n and u["beta"] == b])
        return f"{n}@{b:g}: up{r['up_early'] - r0['up_early']:+.3f} ov{r['ov_inj'] - r['ov_chance']:+.3f} un{ux - u0:+.3f}"
    top = sorted(allc, key=lambda n: -[r for r in res["rows"] if r["ctx"] == key and r["cand"] == n and r["beta"] == betas[-1]][0]["ov_inj"])[:8]
    print(f"[{tag}] {key}: base up {r0['up_early']:.3f} unseenX {u0:.3f} | " + " ".join(summ(n, betas[-1]) for n in ["qk_late", "own", "rand"] + [t for t in top if t not in ("qk_late", "own", "rand")][:5]), flush=True)


def contexts(seeds_main=6, seeds_new=3):
    return [(4, s, 5000 + s) for s in range(seeds_main)] + [(k, 50 + s, 9100 + 10 * k + s) for k in (1, 2, 3) for s in range(seeds_new)]


def stage_zoo(lab, betas=(2.0, 4.0), smoke=False):
    meta = json.load(open(f"{OUT_QK}/port.json"))["meta"]; nref = meta["norm_ref"]
    cands, forms = agnostic_candidates(lab, meta)
    if smoke: cands = {k: cands[k] for k in ("qk_late", "vocab_pc1_2", "vocab_qk64", "rand")}
    perm = np.random.default_rng(99).permutation(N)
    res = dict(meta=dict(nref=nref, betas=list(betas), perm=perm.tolist(), late=forms["late"], groups={n: c["group"] for n, c in cands.items()},
                         eig={n: c.get("eig") for n, c in cands.items() if "eig" in c}, var={n: c.get("var") for n, c in cands.items() if "var" in c},
                         U={n: c["U"].tolist() for n, c in cands.items()}), geometry={}, rows=[], unseen=[], curves={})
    # pairwise overlap among grid-agnostic candidates
    names = list(cands); res["meta"]["cos_pairs"] = {a: {b: float(np.mean(principal_angles(cands[a]["U"], cands[b]["U"]))) for b in names} for a in names}
    t0 = time.time()
    for i, (set_id, s, wseed) in enumerate(contexts() if not smoke else contexts(1, 0)):
        run_context(lab, set_id, s, wseed, cands, forms, nref, betas, perm, res, n_unseen=3)
        json.dump(res, open(f"{OUT}/zoo{'_smoke' if smoke else ''}.json", "w"))
        print(f"[zoo] context {i + 1} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)


def stage_sweep(lab, betas=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0), n_top=5):
    meta = json.load(open(f"{OUT_QK}/port.json"))["meta"]; nref = meta["norm_ref"]
    zoo = json.load(open(f"{OUT}/zoo.json")); cands, forms = agnostic_candidates(lab, meta)
    # rank grid-agnostic, non-reference candidates by override + unseen at beta 4
    rows = zoo["rows"]; un = zoo["unseen"]; b = zoo["meta"]["betas"][-1]
    base_un = {(u["ctx"], u["X"]): u["P_nbrs_X"] for u in un if u["cand"] == "none"}
    score = {}
    for n, c in cands.items():
        if c["group"] == "ref": continue
        ov = np.mean([r["ov_inj"] - r["ov_chance"] for r in rows if r["cand"] == n and r["beta"] == b])
        ux = np.mean([u["trueX_X"] - base_un[(u["ctx"], u["X"])] for u in un if u["cand"] == n and u["beta"] == b])
        score[n] = ov + ux
    top = sorted(score, key=lambda n: -score[n])[:n_top]
    print(f"[sweep] top candidates by override+unseen at beta {b:g}: " + ", ".join(f"{n} ({score[n]:+.3f})" for n in top))
    keep = [n for n in ["qk_late", "rand"] + top if n in cands]
    cands = {n: cands[n] for n in keep}
    perm = np.array(zoo["meta"]["perm"])
    res = dict(meta=dict(nref=nref, betas=list(betas), perm=perm.tolist(), top=top, score=score, groups={n: c["group"] for n, c in cands.items()}), geometry={}, rows=[], unseen=[], curves={})
    t0 = time.time()
    for i, (set_id, s, wseed) in enumerate(contexts()):
        run_context(lab, set_id, s, wseed, cands, forms, nref, betas, perm, res, n_unseen=3, tag="sweep", with_ctx=False)
        json.dump(res, open(f"{OUT}/sweep.json", "w"))
        print(f"[sweep] context {i + 1} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="zoo", choices=["zoo", "sweep", "smoke"])
    args = ap.parse_args(); t0 = time.time(); lab = Lab()
    if args.stage == "smoke": stage_zoo(lab, smoke=True)
    if args.stage == "zoo": stage_zoo(lab)
    if args.stage == "sweep": stage_sweep(lab)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")
