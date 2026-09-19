"""Zoo round 5: the adjacency-kernel port at late layers, and the unseen-EDGE test with it.

  A. adjacency-kernel port (F S_all F^T ~ grid adjacency; self ~ 0) at hs 30 / 32 / 34 / 36, beta 4 / 6 / 8 / 10 (+ plane port
     hs32 b3, qk_late_b4, own_b2 as references). Round 4 found adj@hs34 b6 gives override +0.64 with IMPROVED late accuracy.
  B. unseen edge (never-traversed true edge X-Y, both nodes present with their other neighbors): does the strong late port make
     the model predict Y after X? Conditions: none, adj@34 b6 with the edge IN the kernel, the same with the edge DELETED from
     the kernel (A'), plane port hs32 b3, 4-reader port hs26 b4, own plane b2. 15 contexts x 2 directions.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import Lab, random_walk, cand_map, fit_M, far_node, T_LONG, N, DEV
from port_inject import forward_multi, delta_window, plane_of
from qk_port import port_features, adjacency_port_features, OUT as OUT_QK, NH
from subspace_zoo import qk_forms, contexts, H_MAIN
from zoo_round2 import gen_eig, dirs_at
from zoo_round3 import run
from unseen_edge import walk_avoid_edge, Q_MIN

OUT = os.environ.get("ICL_OUT_ZOO", os.path.join(HERE, "results_zoo")); os.makedirs(OUT, exist_ok=True)


def adj_features(lab, V, ev, nref, A):
    A0 = lab.A; lab.A = A
    try: F, _ = adjacency_port_features(lab, V, ev, nref)
    finally: lab.A = A0
    return F


def main():
    lab = Lab(); meta = json.load(open(f"{OUT_QK}/port.json"))["meta"]; nref = meta["norm_ref"]; late = [tuple(x) for x in meta["late"][:4]]
    S_late, Gn_late = qk_forms(lab, late); V_late, _ = gen_eig(S_late, Gn_late); U_late = dirs_at(V_late, [1, 2])
    all_late = [(l, h) for l in range(H_MAIN, lab.n_layers) for h in range(NH)]; S_all, Gn_all = qk_forms(lab, all_late); V_all, ev_all = gen_eig(S_all, Gn_all)
    U_all = dirs_at(V_all, [1, 2]); F_plane = port_features(lab, U_all, nref); F_adj = adj_features(lab, V_all, ev_all, nref, lab.A)
    # ---------------- A
    conds = {}
    for hs in (30, 32, 34, 36):
        for b in (4.0, 6.0, 8.0, 10.0): conds[f"adj_hs{hs}_b{b:g}"] = dict(hooks=[(F_adj * b, hs)])
    conds["plane_hs32_b3"] = dict(hooks=[(F_plane * 3, 32)]); conds["qk_late_b4"] = dict(hooks=[(port_features(lab, U_late, nref) * 4, H_MAIN)])
    def ctx_conds(inst, M_own, P_own): return {"own_b2": dict(hooks=[((lab.Z2 @ M_own).astype(np.float32) * 2, H_MAIN)])}, {}
    run(lab, conds, ctx_conds, "round5", extra_meta=dict(nref=nref))
    # ---------------- B: unseen edge
    res = {"meta": dict(nref=nref), "rows": []}; dist = np.abs(np.array(lab.coords)[:, None, :] - np.array(lab.coords)[None, :, :]).sum(2); t0 = time.time()
    for set_id, s, wseed in contexts():
        inst = lab.instance(set_id, s); rng = np.random.default_rng(wseed + 777)
        X = int(rng.integers(N)); Y = int(rng.choice(lab.nbrs[X])); edge = (X, Y)
        walk = walk_avoid_edge(lab.nbrs, T_LONG, np.random.default_rng(wseed), edge)
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst]); key = f"set{set_id}_s{s}"
        A_minus = lab.A.copy(); A_minus[X, Y] = A_minus[Y, X] = 0; F_adj_minus = adj_features(lab, V_all, ev_all, nref, A_minus)
        walk_c = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s)); ids_c, seg_c, node_c = lab.build_context([(inst, walk_c)])
        _, hs_c = lab.forward(ids_c, cand, hs_layers=(H_MAIN,)); _, Hc = plane_of(lab, hs_c[H_MAIN], seg_c, node_c, 0); M_own, _ = fit_M(Hc, lab.Z2)
        cc = {"none": [], "adj_hs34_b6": [(F_adj * 6, 34)], "adj_hs34_b6_minus": [(F_adj_minus * 6, 34)], "adj_hs34_b4": [(F_adj * 4, 34)], "adj_hs34_b4_minus": [(F_adj_minus * 4, 34)],
              "plane_hs32_b3": [(F_plane * 3, 32)], "qk_late_b4": [(port_features(lab, U_late, nref) * 4, H_MAIN)], "own_b2": [((lab.Z2 @ M_own).astype(np.float32) * 2, H_MAIN)]}
        for cname, hk in cc.items():
            probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, F, 0, 0, T_LONG), hs) for F, hs in hk], ())
            mass = lab.seg_masses(probs, seg, node, segs, col)[0]
            for src, tgt in ((X, Y), (Y, X)):
                qs = [t for t in range(Q_MIN, len(walk) - 1) if walk[t] == src]
                if not qs: continue
                P = np.array([probs[t + 1] for t in qs]); seen_nb = [q for q in lab.nbrs[src] if q != tgt]
                d2 = [q for q in range(N) if dist[src, q] == 2]; d3 = [q for q in range(N) if dist[src, q] == 3]
                res["rows"].append(dict(ctx=key, set=set_id, seed=s, X=X, Y=Y, src=src, tgt=tgt, cond=cname, n_q=len(qs),
                                        P_heldout=float(P[:, col[int(inst["ids"][tgt])]].mean()), P_seen_nbr=float(np.mean([P[:, col[int(inst["ids"][q])]].mean() for q in seen_nb])),
                                        P_d2=float(np.mean([P[:, col[int(inst["ids"][q])]].mean() for q in d2])), P_d3=float(np.mean([P[:, col[int(inst["ids"][q])]].mean() for q in d3])) if d3 else float("nan"),
                                        acc=float(mass[Q_MIN:].mean())))
        g = lambda c, k: np.mean([r[k] for r in res["rows"] if r["ctx"] == key and r["cond"] == c])
        print(f"[edge5] {key} ({(time.time() - t0) / 60:.1f} min) P(held-out)/P(seen nbr)/P(dist-2): " + " ".join(f"{c}={g(c, 'P_heldout'):.3f}/{g(c, 'P_seen_nbr'):.3f}/{g(c, 'P_d2'):.3f}" for c in cc), flush=True)
        json.dump(res, open(f"{OUT}/edge5.json", "w"))


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"DONE in {(time.time() - t0) / 60:.1f} min")
