"""Zoo round 2: how far does the QK-positive cone extend, and how well can a weight-defined port do?

  spectrum  walk down the eigen-spectrum of the all-late-heads QK form S_all (generalized, fast-RoPE penalized): planes
            from eigen-rank pairs (1,2), (3,4), (5,6), (9,10), (17,18), ..., (2049,2050) and the bottom pairs. Effect vs
            eigen-rank tells whether ONLY the top directions work or the whole positive part of the spectrum.
  layer     the top-2 S_all plane injected at a single hidden state in {8, 12, 16, 20, 26, 30, 34}, and at several at once.
  dims      3 coordinates onto the top-3 directions; two planes (ranks 1-2 and 3-4) carrying the same coordinates.
  geometry  energy of each context's OWN plane (and of reference planes) inside the top-k eigen-directions of S_all and of the
            late-reader form, k = 2..512: is the native plane inside the QK cone?
All readouts as in subspace_zoo (uptake early/late, self, override, unseen 3 X), 15 contexts, beta 4.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import Lab, random_walk, cand_map, fit_M, top_pcs, principal_angles, far_node, T_LONG, T_UNSEEN, N, DEV
from port_inject import forward_multi, delta_window, plane_of
from qk_port import port_features, OUT as OUT_QK, NH
from subspace_zoo import qk_forms, top_dirs, qk_score, orth, contexts, H_MAIN, EARLY, LATE, OV

OUT = os.environ.get("ICL_OUT_ZOO", os.path.join(HERE, "results_zoo")); os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def gen_eig(S, Gn):
    Lc = torch.linalg.cholesky(Gn); Li = torch.linalg.inv(Lc)
    evals, evecs = torch.linalg.eigh(Li @ S @ Li.T); V = Li.T @ evecs
    return V, evals                                   # ascending; V^T S V = diag(evals), V^T Gn V = I


def dirs_at(V, ranks_from_top):
    """unit-norm directions at the given 1-based ranks counted from the TOP of the spectrum (negative = from the bottom)."""
    n = V.shape[1]; cols = [n - r if r > 0 else -r - 1 for r in ranks_from_top]
    U = V[:, cols].T; U = U / U.norm(dim=1, keepdim=True)
    return U.cpu().numpy().astype(np.float32)


def scaled(lab, F, nref):
    return (F * nref / np.linalg.norm(F, axis=1).mean()).astype(np.float32)


def main():
    lab = Lab(); meta = json.load(open(f"{OUT_QK}/port.json"))["meta"]; nref = meta["norm_ref"]
    late = [tuple(x) for x in meta["late"][:4]]
    S_late, Gn_late = qk_forms(lab, late); V_late, ev_late = gen_eig(S_late, Gn_late)
    all_late = [(l, h) for l in range(H_MAIN, lab.n_layers) for h in range(NH)]
    S_all, Gn_all = qk_forms(lab, all_late); V_all, ev_all = gen_eig(S_all, Gn_all)
    U_late = dirs_at(V_late, [1, 2]); U_all = dirs_at(V_all, [1, 2])
    print(f"[r2] S_all eigenvalues: top {ev_all[-6:].cpu().numpy().round(3)} bottom {ev_all[:4].cpu().numpy().round(3)}; #positive {(ev_all > 0).sum().item()} of {lab.d}")
    print(f"[r2] cos(qk_late plane, S_all top-2 plane) = {np.mean(principal_angles(U_late, U_all)):.2f}")
    # ---- conditions: name -> list of (F [16,d], hs)
    pairs = [(1, 2), (3, 4), (5, 6), (9, 10), (17, 18), (33, 34), (65, 66), (129, 130), (257, 258), (513, 514), (1025, 1026), (2049, 2050), (-4, -3), (-2, -1)]
    conds = {}
    for a, b in pairs:
        U = dirs_at(V_all, [a, b]); tag = f"spec_top{a}_{b}" if a > 0 else f"spec_bot{-b}_{-a}"
        conds[tag] = dict(hooks=[(port_features(lab, U, nref) * 4, H_MAIN)], U=U, eig=[float(ev_all[-a].item()) if a > 0 else float(ev_all[-a - 1].item()), float(ev_all[-b].item()) if b > 0 else float(ev_all[-b - 1].item())])
    conds["qk_late_b4"] = dict(hooks=[(port_features(lab, U_late, nref) * 4, H_MAIN)], U=U_late)
    for hs in (8, 12, 16, 20, 30, 34):
        conds[f"all_top2_hs{hs}"] = dict(hooks=[(port_features(lab, U_all, nref) * 4, hs)], U=U_all)
    conds["all_top2_hs16+26"] = dict(hooks=[(port_features(lab, U_all, nref) * 4, 16), (port_features(lab, U_all, nref) * 4, H_MAIN)], U=U_all)
    conds["all_top2_hs20+26"] = dict(hooks=[(port_features(lab, U_all, nref) * 4, 20), (port_features(lab, U_all, nref) * 4, H_MAIN)], U=U_all)
    conds["all_top2_hs12+16+20+26_b2"] = dict(hooks=[(port_features(lab, U_all, nref) * 2, hs) for hs in (12, 16, 20, H_MAIN)], U=U_all)
    U3 = dirs_at(V_all, [1, 2, 3]); F3 = scaled(lab, lab.Z3 @ U3, nref); conds["all_top3_Z3"] = dict(hooks=[(F3 * 4, H_MAIN)], U=U3)
    U34 = dirs_at(V_all, [3, 4]); F2p = scaled(lab, lab.Z2 @ U_all + lab.Z2 @ U34, nref); conds["all_top12+34"] = dict(hooks=[(F2p * 4, H_MAIN)], U=np.concatenate([U_all, U34]))
    Ftop6 = scaled(lab, sum(lab.Z2 @ dirs_at(V_all, [2 * i + 1, 2 * i + 2]) for i in range(3)), nref); conds["all_top6_3planes"] = dict(hooks=[(Ftop6 * 4, H_MAIN)])
    # ---- geometry: energy of planes inside the top-k eigen-directions
    ks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    def topk_basis(V, k):
        Q, _ = torch.linalg.qr(V[:, -k:]); return Q.cpu().numpy().T
    bases_all = {k: topk_basis(V_all, k) for k in ks}; bases_late = {k: topk_basis(V_late, k) for k in ks}
    def energy(U, P): Q = orth(U); return float(np.sum((Q @ P.T) ** 2) / Q.shape[0])
    rngp = np.random.default_rng(321); Qr, _ = np.linalg.qr(rngp.standard_normal((lab.d, 2))); U_rand = Qr.T.astype(np.float32)
    res = dict(meta=dict(nref=nref, late=late, ks=ks, ev_all_top=ev_all[-64:].cpu().numpy().tolist(), ev_all_bottom=ev_all[:8].cpu().numpy().tolist(),
                         n_pos=int((ev_all > 0).sum().item()), cond_eig={n: c.get("eig") for n, c in conds.items() if "eig" in c},
                         cos_late_all=float(np.mean(principal_angles(U_late, U_all))),
                         energy_ref={n: dict(all={k: energy(U, bases_all[k]) for k in ks}, late={k: energy(U, bases_late[k]) for k in ks}) for n, U in (("qk_late", U_late), ("rand", U_rand))}),
               rows=[], unseen=[], curves={}, own_energy=[])
    perm = np.random.default_rng(99).permutation(N); inv = np.argsort(perm)
    def inj_nbrs(sg, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    t0 = time.time()
    for ci, (set_id, s, wseed) in enumerate(contexts()):
        inst = lab.instance(set_id, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(wseed))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst]); key = f"set{set_id}_s{s}"
        walk_c = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s)); ids_c, seg_c, node_c = lab.build_context([(inst, walk_c)])
        _, hs_c = lab.forward(ids_c, cand, hs_layers=(H_MAIN,)); P_own, Hc = plane_of(lab, hs_c[H_MAIN], seg_c, node_c, 0); M_own, _ = fit_M(Hc, lab.Z2)
        res["own_energy"].append(dict(ctx=key, all={k: energy(P_own, bases_all[k]) for k in ks}, late={k: energy(P_own, bases_late[k]) for k in ks},
                                      qk_all=qk_score(S_all, P_own), qk_late=qk_score(S_late, P_own), cos_qk_late=float(np.mean(principal_angles(P_own, U_late))), cos_all_top2=float(np.mean(principal_angles(P_own, U_all)))))
        cc = dict(conds); cc["own_b2"] = dict(hooks=[((lab.Z2 @ M_own).astype(np.float32) * 2, H_MAIN)])
        def self_mass(probs, lo, hi):
            pos = np.nonzero(seg == 0)[0][lo:hi]; return float(np.mean([probs[p, col[int(inst["ids"][node[p]])]] for p in pos]))
        probs0, _ = forward_multi(lab, ids, cand, [], ()); m0 = lab.seg_masses(probs0, seg, node, segs, col)[0]; ch = lab.seg_masses(probs0, seg, node, segs, col, inj_nbrs)[0]
        res["curves"][f"{key}|none"] = m0.tolist()
        res["rows"].append(dict(ctx=key, set=set_id, seed=s, cand="none", up_early=float(m0[EARLY[0]:EARLY[1]].mean()), up_late=float(m0[LATE[0]:LATE[1]].mean()), self_late=self_mass(probs0, *LATE),
                                ov_true=float(m0[OV[0]:OV[1]].mean()), ov_inj=float(ch[OV[0]:OV[1]].mean()), ov_chance=float(ch[OV[0]:OV[1]].mean())))
        for n, c in cc.items():
            hooks = [(delta_window(lab, seg, node, F, 0, 0, T_LONG), hs) for F, hs in c["hooks"]]
            probs, _ = forward_multi(lab, ids, cand, hooks, ()); m = lab.seg_masses(probs, seg, node, segs, col)[0]
            hooks_o = [(delta_window(lab, seg, node, F[perm], 0, 0, T_LONG), hs) for F, hs in c["hooks"]]
            probs_o, _ = forward_multi(lab, ids, cand, hooks_o, ()); mt = lab.seg_masses(probs_o, seg, node, segs, col)[0]; mi = lab.seg_masses(probs_o, seg, node, segs, col, inj_nbrs)[0]
            res["curves"][f"{key}|{n}"] = m.tolist()
            res["rows"].append(dict(ctx=key, set=set_id, seed=s, cand=n, up_early=float(m[EARLY[0]:EARLY[1]].mean()), up_late=float(m[LATE[0]:LATE[1]].mean()), self_late=self_mass(probs, *LATE),
                                    ov_true=float(mt[OV[0]:OV[1]].mean()), ov_inj=float(mi[OV[0]:OV[1]].mean()), ov_chance=float(ch[OV[0]:OV[1]].mean())))
        rngu = np.random.default_rng(8000 + s + 100 * set_id)
        for X in rngu.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rngu)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rngu, avoid=X), [X]]); segs_u = [(inst, walk_q)]
            ids_u, seg_u, node_u = lab.build_context(segs_u); cand_u, col_u = cand_map([inst])
            def masses(probs):
                p = probs[-1]; return dict(P_nbrs_X=float(sum(p[col_u[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col_u[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            res["unseen"].append(dict(ctx=key, set=set_id, seed=s, X=X, Y=Y, cand="none", **masses(forward_multi(lab, ids_u, cand_u, [], ())[0])))
            for n, c in cc.items():
                hk = [(delta_window(lab, seg_u, node_u, F, 0, 0, T_UNSEEN + 1), hs) for F, hs in c["hooks"]]
                Fys = []
                for F, hs in c["hooks"]:
                    Fy = F.copy(); Fy[X] = F[Y]; Fys.append((delta_window(lab, seg_u, node_u, Fy, 0, 0, T_UNSEEN + 1), hs))
                tx = masses(forward_multi(lab, ids_u, cand_u, hk, ())[0]); cy = masses(forward_multi(lab, ids_u, cand_u, Fys, ())[0])
                res["unseen"].append(dict(ctx=key, set=set_id, seed=s, X=X, Y=Y, cand=n, trueX_X=tx["P_nbrs_X"], trueX_Y=tx["P_nbrs_Y"], claimY_X=cy["P_nbrs_X"], claimY_Y=cy["P_nbrs_Y"]))
        r0 = res["rows"][-len(cc) - 1]; u0 = np.mean([u["P_nbrs_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == "none"])
        def summ(n):
            r = [r for r in res["rows"] if r["ctx"] == key and r["cand"] == n][0]; ux = np.mean([u["trueX_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == n])
            return f"{n}: up{r['up_early'] - r0['up_early']:+.3f} ov{r['ov_inj'] - r['ov_chance']:+.3f} un{ux - u0:+.3f}"
        print(f"[r2] {key} ({(time.time() - t0) / 60:.1f} min): " + " | ".join(summ(n) for n in ("qk_late_b4", "spec_top1_2", "spec_top5_6", "spec_top33_34", "spec_top257_258", "spec_bot1_2", "all_top2_hs16", "all_top2_hs16+26", "all_top3_Z3", "own_b2")), flush=True)
        json.dump(res, open(f"{OUT}/round2.json", "w"))
    print(f"[r2] own plane energy in top-k of S_all: " + " ".join(f"k{k}:{np.mean([o['all'][k] for o in res['own_energy']]):.2f}" for k in ks))
    print(f"[r2] rand plane energy in top-k of S_all: " + " ".join(f"k{k}:{res['meta']['energy_ref']['rand']['all'][k]:.2f}" for k in ks))


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"DONE in {(time.time() - t0) / 60:.1f} min")
