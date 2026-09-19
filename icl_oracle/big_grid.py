"""Large-grid replication of the key injection experiments (16, 25, 64, 144, 216 nodes; gemma-2-9b).

Grids: 4x4, 5x5, 8x8, 12x12, 12x18 (216 = 12 x 18 rectangle so that the 2-D spectral coordinates remain the top pair).
Context length scales with the grid (1200, 2000, 4000, 7000, 8000 tokens). Words: single-token common English words drawn from
wordfreq (each context samples N fresh words). 6 contexts per grid (fresh words, assignment, walk), everything paired.

Strategies (node j gets a fixed vector at every occurrence):
  none          baseline
  own_b2        the context's OWN plane, calibrated on a separate walk of the same words (hs 26, beta 2)      [Part I]
  rand_b4       random plane, hs 32, beta 4                                                                    [null control]
  plane_*       top-2 plane of the all-late-heads QK form, hs 32 (beta 3 fixed; and beta matched, see below)   [Part IV plane port]
  adj_*         adjacency-kernel port: F S F^T ~ A_trunc (12 positive + 12 negative grid modes), hs 32
                (beta 6 fixed; and matched)                                                                    [Part IV best port]
"Matched" beta rescales each port so that the implied added attention logit contrast (neighbor minus distance-2 node) equals what
it was on the 4x4 grid — with more nodes the coordinates get closer together, so a fixed beta means a weaker kernel.

Readouts per context: dense rule-following curve, early/late windows, self-copy, override (permuted map; injected-map mass minus
chance), unseen node (2 X per context: P(true neighbors) with true coordinate; P(neighbors of far Y) with Y's coordinate).
Geometry per context: own-plane fit R^2, its energy inside the top-32 cone of S_all, cos to the port plane.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import Lab, principal_angles, top_pcs, DEV
from qk_port import OUT as OUT_QK, NH
from subspace_zoo import qk_forms, orth
from zoo_round2 import gen_eig, dirs_at

OUT = os.environ.get("ICL_OUT_BIG", os.path.join(HERE, "results_big")); os.makedirs(OUT, exist_ok=True)
GRIDS = {16: (4, 4), 25: (5, 5), 64: (8, 8), 144: (12, 12), 216: (12, 18)}
T_CTX = {16: 1200, 25: 2000, 64: 4000, 144: 7000, 216: 8000}
HS_OWN, HS_PORT = 26, 32
M_MODES = 12


# ----------------------------------------------------------------------------- grid
def make_grid(n1, n2):
    coords = [(r, c) for r in range(n1) for c in range(n2)]; idx = {rc: i for i, rc in enumerate(coords)}; n = len(coords)
    A = np.zeros((n, n), np.float32)
    for i, (r, c) in enumerate(coords):
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if (r + dr, c + dc) in idx: A[i, idx[(r + dr, c + dc)]] = 1
    nbrs = [np.nonzero(A[i])[0].tolist() for i in range(n)]
    v1 = np.cos(np.pi * (np.arange(n1) + 0.5) / n1); v2 = np.cos(np.pi * (np.arange(n2) + 0.5) / n2)
    Z = np.array([[v1[r], v2[c]] for r, c in coords], np.float64); Z -= Z.mean(0); Z /= np.linalg.norm(Z, axis=0, keepdims=True)
    dist = np.abs(np.array(coords)[:, None, :] - np.array(coords)[None, :, :]).sum(2)
    return dict(n=n, n1=n1, n2=n2, coords=coords, A=A, nbrs=nbrs, Z2=Z, dist=dist)


def random_walk(nbrs, T, rng, avoid=None):
    allowed = [i for i in range(len(nbrs)) if i != avoid]; cur = int(rng.choice(allowed)); walk = [cur]
    for _ in range(T - 1):
        opts = [j for j in nbrs[cur] if j != avoid]; cur = int(rng.choice(opts)); walk.append(cur)
    return np.array(walk)


# ----------------------------------------------------------------------------- words
def build_pool(tok, skip=300, top=30000):
    from wordfreq import top_n_list
    pool, seen = [], set()
    for w in top_n_list("en", top)[skip:]:
        if not (w.isalpha() and w.islower() and w.isascii() and 3 <= len(w) <= 9): continue
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1 and t[0] not in seen and tok.decode(t).strip() == w:
            pool.append((w, int(t[0]))); seen.add(t[0])
    return pool


# ----------------------------------------------------------------------------- model helpers (N-generic)
@torch.inference_mode()
def forward_probs(lab, ids, cand_ids, hooks, hs_layers=(), chunk=1024):
    """probs over cand_ids [T, n_cand], normalized over the full vocabulary; optional hidden states."""
    handles = []
    for delta, h in hooks:
        def fn(mod, inp, out, delta=delta):
            if isinstance(out, tuple): out[0].add_(delta[None]); return out
            out.add_(delta[None]); return out
        handles.append(lab.model.model.layers[h - 1].register_forward_hook(fn))
    try:
        out = lab.model(torch.tensor(ids, device=DEV)[None], output_hidden_states=bool(hs_layers))
    finally:
        for hd in handles: hd.remove()
    logits = out.logits[0]; cand = torch.tensor(cand_ids, device=DEV); res = []
    for i in range(0, logits.shape[0], chunk):
        L = logits[i:i + chunk].float(); res.append(torch.exp(L[:, cand] - torch.logsumexp(L, -1, keepdim=True)).cpu())
    hs = {h: out.hidden_states[h][0].float().cpu().numpy() for h in hs_layers}
    return torch.cat(res).numpy(), hs


def delta_for(lab, walk, F):
    """[T+1, d] bf16: F[node] at each walk position (position 0 = BOS gets nothing)."""
    delta = np.zeros((len(walk) + 1, lab.d), np.float32); delta[1:] = F[walk]
    return torch.tensor(delta, dtype=torch.bfloat16, device=DEV)


def masses(probs, walk, ids_of, col, nbr_of):
    """rule-following mass at each walk position with a successor: sum of probs on the tokens of nbr_of(node)."""
    return np.array([sum(probs[t + 1, col[int(ids_of[q])]] for q in nbr_of(int(walk[t]))) for t in range(len(walk) - 1)])


def node_means(hs, walk, n, lo):
    """mean hidden state per node over walk positions >= lo (hs row t+1 corresponds to walk[t])."""
    H = np.full((n, hs.shape[1]), np.nan, np.float32)
    for j in range(n):
        pj = np.nonzero(walk[lo:] == j)[0] + lo + 1
        if len(pj): H[j] = hs[pj].mean(0)
    return np.where(np.isnan(H), np.nanmean(H, 0, keepdims=True), H)


# ----------------------------------------------------------------------------- ports (N-generic)
def plane_features(Z2, U, nref):
    F = Z2 @ U[:2]; return (F * nref / np.linalg.norm(F, axis=1).mean()).astype(np.float32)


def adjacency_features(A, V, evals, nref, m=M_MODES):
    """F [n,d] with F S F^T ~ A truncated to its m largest-positive and m most-negative eigenmodes, mapped onto the m top /
    m bottom generalized eigen-directions of S (u normalized so u^T S u = +-1). For the 4x4 grid (6+6 modes) this is the
    Part III/IV construction exactly."""
    mu, z = np.linalg.eigh(A.astype(np.float64)); ev = evals.cpu().numpy(); Vn = V.cpu().numpy(); n = A.shape[0]
    pos = [i for i in np.argsort(mu)[::-1] if mu[i] > 1e-6][:m]; neg = [i for i in np.argsort(mu) if mu[i] < -1e-6][:m]
    top = np.argsort(ev)[::-1][:len(pos)]; bot = np.argsort(ev)[:len(neg)]
    F = np.zeros((n, Vn.shape[0]))
    for i, c in zip(pos, top): F += np.sqrt(mu[i]) * np.outer(z[:, i], Vn[:, c] / np.sqrt(abs(ev[c])))
    for i, c in zip(neg, bot): F += np.sqrt(-mu[i]) * np.outer(z[:, i], Vn[:, c] / np.sqrt(abs(ev[c])))
    F -= F.mean(0)
    return (F * nref / np.linalg.norm(F, axis=1).mean()).astype(np.float32), dict(n_pos=len(pos), n_neg=len(neg), ev_used=[float(ev[c]) for c in list(top) + list(bot)])


@torch.no_grad()
def implied_kernel(S, F, g):
    """added attention logit (per unit beta^2, before the RMSNorm factor) between node i (query) and j (key): F S F^T."""
    Ft = torch.tensor(F, device=DEV); K = (Ft @ S @ Ft.T).cpu().numpy(); d = g["dist"]; eye = np.eye(g["n"], dtype=bool)
    stats = dict(self=float(K[eye].mean()), nbr=float(K[d == 1].mean()), d2=float(K[d == 2].mean()), d3=float(K[d == 3].mean()), other=float(K[(d >= 2)].mean()))
    stats["contrast"] = stats["nbr"] - stats["d2"]                       # discriminative contrast: neighbor vs distance-2 node
    return stats


# ----------------------------------------------------------------------------- main
def main(sizes, n_ctx=6, smoke=False):
    lab = Lab(); meta_qk = json.load(open(f"{OUT_QK}/port.json"))["meta"]; nref = meta_qk["norm_ref"]
    pool = build_pool(lab.tok); print(f"[big] word pool: {len(pool)} single-token words, e.g. {[w for w, _ in pool[:12]]}", flush=True)
    all_late = [(l, h) for l in range(HS_OWN, lab.n_layers) for h in range(NH)]; S_all, Gn_all = qk_forms(lab, all_late); V_all, ev_all = gen_eig(S_all, Gn_all)
    U_plane = dirs_at(V_all, [1, 2]); Qc, _ = torch.linalg.qr(V_all[:, -32:]); P_cone32 = Qc.cpu().numpy().T
    Qr, _ = np.linalg.qr(np.random.default_rng(321).standard_normal((lab.d, 2))); U_rand = Qr.T.astype(np.float32)
    # reference kernel contrasts on the 4x4 grid (beta 3 plane, beta 6 adjacency)
    g16 = make_grid(4, 4); k16_plane = implied_kernel(S_all, plane_features(g16["Z2"], U_plane, nref), g16); k16_adj = implied_kernel(S_all, adjacency_features(g16["A"], V_all, ev_all, nref)[0], g16)
    ref_plane, ref_adj = 9.0 * k16_plane["contrast"], 36.0 * k16_adj["contrast"]
    res = dict(meta=dict(nref=nref, grids={str(n): GRIDS[n] for n in sizes}, T={str(n): T_CTX[n] for n in sizes}, m_modes=M_MODES, k16=dict(plane=k16_plane, adj=k16_adj), betas={}, kernels={}, pool_size=len(pool)),
               rows=[], unseen=[], curves={}, geometry=[])
    t0 = time.time()
    for n in sizes:
        g = make_grid(*GRIDS[n]); T = T_CTX[n] if not smoke else min(T_CTX[n], 1500); Z2 = g["Z2"]
        F_plane = plane_features(Z2, U_plane, nref); F_adj, adj_info = adjacency_features(g["A"], V_all, ev_all, nref); F_rand = plane_features(Z2, U_rand, nref)
        kp, ka = implied_kernel(S_all, F_plane, g), implied_kernel(S_all, F_adj, g)
        b_plane = float(np.clip(np.sqrt(ref_plane / max(kp["contrast"], 1e-9)), 1, 12)); b_adj = float(np.clip(np.sqrt(ref_adj / max(ka["contrast"], 1e-9)), 1, 12))
        res["meta"]["betas"][str(n)] = dict(plane_matched=b_plane, adj_matched=b_adj); res["meta"]["kernels"][str(n)] = dict(plane=kp, adj=ka, adj_info=adj_info)
        print(f"[big] N={n} ({g['n1']}x{g['n2']}), T={T}: matched beta plane {b_plane:.2f} adj {b_adj:.2f}; kernel plane self/nbr/d2/other {kp['self']:.2f}/{kp['nbr']:.2f}/{kp['d2']:.2f}/{kp['other']:.2f}; adj {ka['self']:.2f}/{ka['nbr']:.2f}/{ka['d2']:.2f}/{ka['other']:.2f}", flush=True)
        early, late, ov_win = (T // 12, T // 4), (T // 3, T), (T // 6, T // 2)
        for c in range(n_ctx if not smoke else 1):
            rng = np.random.default_rng(100_000 * n + 17 * c + 5); key = f"N{n}_c{c}"
            pick = rng.choice(len(pool), n, replace=False); ids_of = np.array([pool[i][1] for i in pick]); words = [pool[i][0] for i in pick]
            cand = sorted(set(int(t) for t in ids_of)); col = {t: i for i, t in enumerate(cand)}
            walk = random_walk(g["nbrs"], T, rng); ids = np.concatenate([[lab.bos_id], ids_of[walk]])
            # own plane: calibration walk (same words), node means over the second half
            walk_c = random_walk(g["nbrs"], T, np.random.default_rng(rng.integers(1 << 30))); ids_c = np.concatenate([[lab.bos_id], ids_of[walk_c]])
            probs_c, hs_c = forward_probs(lab, ids_c, cand, [], hs_layers=(HS_OWN,)); H = node_means(hs_c[HS_OWN], walk_c, n, T // 2).astype(np.float64); Hc = H - H.mean(0)
            M_own, *_ = np.linalg.lstsq(Z2, Hc, rcond=None); r2 = float(1 - np.sum((Hc - Z2 @ M_own) ** 2) / np.sum(Hc ** 2)); P_own = top_pcs(Hc, 2)
            F_own = (Z2 @ M_own).astype(np.float32); acc_cal = masses(probs_c, walk_c, ids_of, col, lambda j: g["nbrs"][j])
            res["geometry"].append(dict(ctx=key, N=n, r2_own=r2, r2_in_pcs2=float(1 - np.sum(((Hc - Z2 @ M_own) @ P_own.T) ** 2) / np.sum((Hc @ P_own.T) ** 2)),
                                        own_in_cone32=float(np.sum((orth(P_own) @ P_cone32.T) ** 2) / 2), own_cos_plane=float(np.mean(principal_angles(P_own, U_plane))),
                                        own_norm=float(np.linalg.norm(F_own, axis=1).mean()), pc_var2=float(np.sum((Hc @ P_own.T) ** 2) / np.sum(Hc ** 2)), acc_cal_late=float(acc_cal[late[0]:].mean())))
            perm = rng.permutation(n); inv = np.argsort(perm); inj_nbr = lambda j: [int(inv[q]) for q in g["nbrs"][perm[j]]]; true_nbr = lambda j: g["nbrs"][j]
            conds = {"none": [], "own_b2": [(F_own * 2, HS_OWN)], "rand_b4": [(F_rand * 4, HS_PORT)], "plane_b3": [(F_plane * 3, HS_PORT)], "plane_matched": [(F_plane * b_plane, HS_PORT)],
                     "adj_b6": [(F_adj * 6, HS_PORT)], "adj_matched": [(F_adj * b_adj, HS_PORT)]}
            ch = None
            for cname, hk in conds.items():
                probs, _ = forward_probs(lab, ids, cand, [(delta_for(lab, walk, F), hs) for F, hs in hk]); m = masses(probs, walk, ids_of, col, true_nbr)
                self_m = float(np.mean([probs[t + 1, col[int(ids_of[walk[t]])]] for t in range(late[0], len(walk) - 1)]))
                if cname == "none": ch = masses(probs, walk, ids_of, col, inj_nbr); mt, mi = m, ch
                else:
                    probs_o, _ = forward_probs(lab, ids, cand, [(delta_for(lab, walk, F[perm]), hs) for F, hs in hk]); mt = masses(probs_o, walk, ids_of, col, true_nbr); mi = masses(probs_o, walk, ids_of, col, inj_nbr)
                res["curves"][f"{key}|{cname}"] = m.astype(np.float32).tolist()
                res["rows"].append(dict(ctx=key, N=n, cand=cname, up_early=float(m[early[0]:early[1]].mean()), up_late=float(m[late[0]:late[1]].mean()), self_late=self_m,
                                        ov_true=float(mt[ov_win[0]:ov_win[1]].mean()), ov_inj=float(mi[ov_win[0]:ov_win[1]].mean()), ov_chance=float(ch[ov_win[0]:ov_win[1]].mean())))
            # unseen node
            for X in rng.choice(n, 2, replace=False):
                X = int(X); far = np.nonzero(g["dist"][X] >= 3)[0]; Y = int(rng.choice(far))
                walk_u = np.concatenate([random_walk(g["nbrs"], T // 2, rng, avoid=X), [X]]); ids_u = np.concatenate([[lab.bos_id], ids_of[walk_u]])
                def mx(probs, node): p = probs[-1]; return float(sum(p[col[int(ids_of[q])]] for q in g["nbrs"][node]))
                p0, _ = forward_probs(lab, ids_u, cand, []); res["unseen"].append(dict(ctx=key, N=n, X=X, Y=Y, cand="none", P_nbrs_X=mx(p0, X), P_nbrs_Y=mx(p0, Y)))
                for cname, hk in conds.items():
                    if cname == "none": continue
                    tx, _ = forward_probs(lab, ids_u, cand, [(delta_for(lab, walk_u, F), hs) for F, hs in hk])
                    hk_y = []
                    for F, hs in hk:
                        Fy = F.copy(); Fy[X] = F[Y]; hk_y.append((delta_for(lab, walk_u, Fy), hs))
                    cy, _ = forward_probs(lab, ids_u, cand, hk_y)
                    res["unseen"].append(dict(ctx=key, N=n, X=X, Y=Y, cand=cname, trueX_X=mx(tx, X), trueX_Y=mx(tx, Y), claimY_X=mx(cy, X), claimY_Y=mx(cy, Y)))
            r0 = [r for r in res["rows"] if r["ctx"] == key and r["cand"] == "none"][0]; u0 = np.mean([u["P_nbrs_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == "none"])
            def summ(cn):
                r = [r for r in res["rows"] if r["ctx"] == key and r["cand"] == cn][0]; ux = np.mean([u["trueX_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == cn])
                return f"{cn}: e{r['up_early'] - r0['up_early']:+.2f} l{r['up_late'] - r0['up_late']:+.2f} ov{r['ov_inj'] - r['ov_chance']:+.2f} un{ux - u0:+.2f}"
            print(f"[big] {key} ({(time.time() - t0) / 60:.1f} min) base early {r0['up_early']:.2f} late {r0['up_late']:.2f} unseen {u0:.2f} | own R2 {r2:.2f} cone32 {res['geometry'][-1]['own_in_cone32']:.2f} | " + " | ".join(summ(cn) for cn in conds if cn != "none"), flush=True)
            json.dump(res, open(f"{OUT}/big{'_smoke' if smoke else ''}.json", "w"))
    print(f"[big] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="16,25,64,144,216"); ap.add_argument("--n_ctx", type=int, default=6); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); main([int(s) for s in a.sizes.split(",")], a.n_ctx, a.smoke)
