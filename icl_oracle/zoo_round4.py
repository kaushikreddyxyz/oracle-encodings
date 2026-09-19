"""Zoo round 4: the late-layer port. Round 2 found the all-late-heads plane works better injected at hs 30-34 than at hs 26.

  layer fine sweep   S_all top-2 plane at hs 28, 30, 32, 34, 36, 38, 40 (beta 4)
  matched forms      plane from the heads that can actually read an injection at hs h (layers >= h): S_ge30@hs30, S_ge34@hs34, S_ge38@hs38
  strength           beta 2, 3, 6, 8 at hs 32 and hs 34
  cone at late hs    eigen-rank 3-4 and 9-10 planes of S_all at hs 34 (is the whole cone usable there too?)
  adjacency kernel   F S F^T ~ A (self ~ 0, neighbor > 0) from S_all at hs 26 / 30 / 34 (does removing the self-term keep the
                     late-context accuracy while retaining control?)
  references         qk_late_b4 @26, own_b2 @26, random plane @34
15 contexts; readouts as before.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import Lab, principal_angles, N, DEV
from qk_port import port_features, adjacency_port_features, OUT as OUT_QK, NH
from subspace_zoo import qk_forms, top_dirs, qk_score, H_MAIN
from zoo_round2 import gen_eig, dirs_at
from zoo_round3 import run

OUT = os.environ.get("ICL_OUT_ZOO", os.path.join(HERE, "results_zoo")); os.makedirs(OUT, exist_ok=True)


def main():
    lab = Lab(); meta = json.load(open(f"{OUT_QK}/port.json"))["meta"]; nref = meta["norm_ref"]; late = [tuple(x) for x in meta["late"][:4]]
    S_late, Gn_late = qk_forms(lab, late); V_late, _ = gen_eig(S_late, Gn_late); U_late = dirs_at(V_late, [1, 2])
    all_late = [(l, h) for l in range(H_MAIN, lab.n_layers) for h in range(NH)]; S_all, Gn_all = qk_forms(lab, all_late); V_all, ev_all = gen_eig(S_all, Gn_all)
    U_all = dirs_at(V_all, [1, 2]); F_all = port_features(lab, U_all, nref)
    conds = {}; geo = {}
    def add(name, F, hs, beta):
        conds[name] = dict(hooks=[(F * beta, hs)])
    for hs in (28, 30, 32, 34, 36, 38, 40): add(f"all_top2_hs{hs}", F_all, hs, 4.0)
    for hs in (30, 34, 38):
        heads = [(l, h) for l in range(hs, lab.n_layers) for h in range(NH)]; S, Gn = qk_forms(lab, heads); U, ev = top_dirs(S, Gn, 2, +1)
        add(f"ge{hs}_top2_hs{hs}", port_features(lab, U, nref), hs, 4.0); geo[f"ge{hs}"] = dict(eig=ev, cos_vs_all=float(np.mean(principal_angles(U, U_all))), cos_vs_late=float(np.mean(principal_angles(U, U_late))), qk_all=qk_score(S_all, U))
    for hs in (32, 34):
        for b in (2.0, 3.0, 6.0, 8.0): add(f"all_top2_hs{hs}_b{b:g}", F_all, hs, b)
    for a, b in ((3, 4), (9, 10)): add(f"spec_top{a}_{b}_hs34", port_features(lab, dirs_at(V_all, [a, b]), nref), 34, 4.0)
    F_adj, info = adjacency_port_features(lab, V_all, ev_all, nref); geo["adj"] = info
    for hs in (26, 30, 34): add(f"adj_all_hs{hs}", F_adj, hs, 4.0)
    add("adj_all_hs34_b6", F_adj, 34, 6.0)
    add("qk_late_b4", port_features(lab, U_late, nref), H_MAIN, 4.0)
    Qr, _ = np.linalg.qr(np.random.default_rng(321).standard_normal((lab.d, 2))); add("rand_hs34", port_features(lab, Qr.T.astype(np.float32), nref), 34, 4.0)
    def ctx_conds(inst, M_own, P_own):
        return {"own_b2": dict(hooks=[((lab.Z2 @ M_own).astype(np.float32) * 2, H_MAIN)])}, {}
    run(lab, conds, ctx_conds, "round4", extra_meta=dict(nref=nref, geo=geo))


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"DONE in {(time.time() - t0) / 60:.1f} min")
