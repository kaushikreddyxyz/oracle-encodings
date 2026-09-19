"""Unseen-EDGE test: the walk never traverses one true grid edge (X,Y); both nodes appear with their other
neighbors. Tokens place X and Y at path-distance 3, the true geometry at distance 1. Does each channel make the
model predict Y after X (and X after Y) above matched non-neighbors?

Channels (hs26 unless noted): none, calibrated own plane, random plane, QK late port b3/b4, anti-port,
non-reader port, late adjacency port (kernel = A) and the same with the held-out edge DELETED (kernel = A'),
plus the hs16 spectral port and hs16 adjacency port. 15 contexts, paired.
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import Lab, cand_map, fit_M, random_features, T_LONG, N, DEV
from port_inject import forward_multi, delta_window, plane_of
from qk_port import (build_port, build_port_general, adjacency_port_features, port_features, own_plane_features, OUT, NH)

OUT_E = os.environ.get("ICL_OUT_EDGE", os.path.join(HERE, "results_edge")); os.makedirs(OUT_E, exist_ok=True)
Q_MIN = 300


def walk_avoid_edge(nbrs, T, rng, edge):
    X, Y = edge
    cur = int(rng.integers(len(nbrs))); walk = [cur]
    for _ in range(T - 1):
        opts = [j for j in nbrs[cur] if not ((cur == X and j == Y) or (cur == Y and j == X))]
        cur = int(rng.choice(opts)); walk.append(cur)
    return np.array(walk)


def adjacency_features_for(lab, V, evals, norm, A):
    """adjacency_port_features with an arbitrary adjacency matrix A (temporarily swapped into lab)."""
    A0 = lab.A; lab.A = A
    try:
        F, info = adjacency_port_features(lab, V, evals, norm)
    finally:
        lab.A = A0
    return F


def main(seeds_main=6, seeds_new=3):
    lab = Lab(); h_main = 26
    pr = json.load(open(f"{OUT}/port.json")); meta = pr["meta"]; nref = meta["norm_ref"]
    late = [tuple(x) for x in meta["late"][:4]]; readers = [tuple(x) for x in meta["readers"]]; hs_late = min(l for l, _ in late)
    U_late, _, _ = build_port(lab, late, k=2, fast_penalty=1.0); F_late = port_features(lab, U_late, nref)
    U_anti, _, _ = build_port(lab, late, k=2, sign=-1, fast_penalty=1.0); F_anti = port_features(lab, U_anti, nref)
    rng0 = np.random.default_rng(3); used = set(late)
    nonlate = [t for t in [(int(l), int(h)) for l, h in zip(rng0.integers(h_main, lab.n_layers, 300), rng0.integers(0, NH, 300))] if t not in used][:4]
    U_non, _, _ = build_port(lab, nonlate, k=2, fast_penalty=1.0); F_non = port_features(lab, U_non, nref)
    V_late, evl = build_port_general(lab, late)
    U_z2 = np.array(meta["U"], np.float32); F_z2 = port_features(lab, U_z2, nref)
    V_all, ev_all = build_port_general(lab, readers)
    F_rand = random_features(lab, nref, seed=321)
    dist = np.abs(np.array(lab.coords)[:, None, :] - np.array(lab.coords)[None, :, :]).sum(2)
    res = {"meta": dict(late=late, nonlate=nonlate, hs_late=hs_late, readers=readers), "rows": []}
    contexts = [(4, s, 5000 + s) for s in range(seeds_main)] + [(k, 50 + s, 9100 + 10 * k + s) for k in (1, 2, 3) for s in range(seeds_new)]
    for set_id, s, wseed in contexts:
        inst = lab.instance(set_id, s); rng = np.random.default_rng(wseed + 777)
        X = int(rng.integers(N)); Y = int(rng.choice(lab.nbrs[X])); edge = (X, Y)
        walk = walk_avoid_edge(lab.nbrs, T_LONG, np.random.default_rng(wseed), edge)
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        A_minus = lab.A.copy(); A_minus[X, Y] = A_minus[Y, X] = 0
        F_adj_full = adjacency_features_for(lab, V_late, evl, nref, lab.A); F_adj_minus = adjacency_features_for(lab, V_late, evl, nref, A_minus)
        F_adj16_full = adjacency_features_for(lab, V_all, ev_all, nref, lab.A)
        F_own = own_plane_features(lab, inst, s, h_main)     # calibrated on a separate FULL walk (edge present)
        conds = {"none": [], "own_b2": [(delta_window(lab, seg, node, F_own * 2, 0, 0, T_LONG), h_main)],
                 "rand26_b4": [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), hs_late)],
                 "late_b3": [(delta_window(lab, seg, node, F_late * 3, 0, 0, T_LONG), hs_late)], "late_b4": [(delta_window(lab, seg, node, F_late * 4, 0, 0, T_LONG), hs_late)],
                 "lateanti_b4": [(delta_window(lab, seg, node, F_anti * 4, 0, 0, T_LONG), hs_late)], "latenon_b4": [(delta_window(lab, seg, node, F_non * 4, 0, 0, T_LONG), hs_late)],
                 "lateadj_b4": [(delta_window(lab, seg, node, F_adj_full * 4, 0, 0, T_LONG), hs_late)], "lateadj_minus_b4": [(delta_window(lab, seg, node, F_adj_minus * 4, 0, 0, T_LONG), hs_late)],
                 "z2_hs16_b4": [(delta_window(lab, seg, node, F_z2 * 4, 0, 0, T_LONG), 16)], "adj_hs16_b6": [(delta_window(lab, seg, node, F_adj16_full * 6, 0, 0, T_LONG), 16)]}
        for cname, hooks in conds.items():
            probs, _ = forward_multi(lab, ids, cand, hooks, ())
            mass = lab.seg_masses(probs, seg, node, segs, col)[0]
            for src, tgt in ((X, Y), (Y, X)):
                qs = [t for t in range(Q_MIN, len(walk) - 1) if walk[t] == src]
                if not qs: continue
                P = np.array([probs[t + 1] for t in qs])                          # [nq, 16] node probs
                seen_nb = [q for q in lab.nbrs[src] if q != tgt]
                d2 = [q for q in range(N) if dist[src, q] == 2]; d3 = [q for q in range(N) if dist[src, q] == 3]
                res["rows"].append(dict(set=set_id, seed=s, X=X, Y=Y, src=src, tgt=tgt, cond=cname, n_q=len(qs),
                                        P_heldout=float(P[:, col[int(inst["ids"][tgt])]].mean()),
                                        P_seen_nbr=float(np.mean([P[:, col[int(inst["ids"][q])]].mean() for q in seen_nb])),
                                        P_d2=float(np.mean([P[:, col[int(inst["ids"][q])]].mean() for q in d2])),
                                        P_d3=float(np.mean([P[:, col[int(inst["ids"][q])]].mean() for q in d3])) if d3 else float("nan"),
                                        acc=float(mass[Q_MIN:].mean())))
        g = lambda c: np.mean([r["P_heldout"] for r in res["rows"] if r["set"] == set_id and r["seed"] == s and r["cond"] == c])
        g2 = lambda c: np.mean([r["P_d2"] for r in res["rows"] if r["set"] == set_id and r["seed"] == s and r["cond"] == c])
        print(f"[edge] set{set_id} seed{s} edge={lab.words[X] if False else X}-{Y}: P(held-out)/P(dist-2): " + " ".join(f"{c}={g(c):.3f}/{g2(c):.3f}" for c in conds))
        json.dump(res, open(f"{OUT_E}/edge.json", "w"))


if __name__ == "__main__":
    main()
