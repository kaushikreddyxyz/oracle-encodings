"""Zoo round 3: dissecting the cone.

  own split    each context's OWN plane features split into the part inside the top-k QK cone of S_all (k = 16, 64, 256)
               and the part outside it (F_in + F_out = F_own, same fitted scale, beta 2): does the native plane work
               BECAUSE of its QK-cone component?
  layer bands  QK ports from all heads of layers 12-25, 26-33, 34-41, 12-41, and all 672 heads (which layers carry the cone?)
  act_qk       the S_late-best plane inside the top-k activation PCs (k = 8, 16, 32, 64, 128) of a foreign walk on word set 0,
               and k = 64 from word set 1 (is it set-specific?), and S_all-best inside act64
  vocab_qk     top-2048 / 3072 vocabulary PCs (how much of the vocabulary subspace is needed to recover the port?)
All at hs 26, beta 4 unless noted; 15 contexts; readouts as before.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import Lab, random_walk, cand_map, fit_M, principal_angles, far_node, T_LONG, T_UNSEEN, N, DEV
from port_inject import forward_multi, delta_window, plane_of
from qk_port import port_features, OUT as OUT_QK, NH
from subspace_zoo import qk_forms, top_dirs, qk_score, orth, contexts, vocab_pcs, act_pcs, H_MAIN, EARLY, LATE, OV
from zoo_round2 import gen_eig, dirs_at

OUT = os.environ.get("ICL_OUT_ZOO", os.path.join(HERE, "results_zoo")); os.makedirs(OUT, exist_ok=True)


def run(lab, conds_static, ctx_conds_fn, out_name, extra_meta=None):
    perm = np.random.default_rng(99).permutation(N); inv = np.argsort(perm)
    def inj_nbrs(sg, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    res = dict(meta=dict(perm=perm.tolist(), **(extra_meta or {})), rows=[], unseen=[], curves={}, geometry={}); t0 = time.time()
    for ci, (set_id, s, wseed) in enumerate(contexts()):
        inst = lab.instance(set_id, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(wseed))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst]); key = f"set{set_id}_s{s}"
        walk_c = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s)); ids_c, seg_c, node_c = lab.build_context([(inst, walk_c)])
        _, hs_c = lab.forward(ids_c, cand, hs_layers=(H_MAIN,)); P_own, Hc = plane_of(lab, hs_c[H_MAIN], seg_c, node_c, 0); M_own, _ = fit_M(Hc, lab.Z2)
        cc = dict(conds_static); ctx_c, geo = ctx_conds_fn(inst, M_own, P_own); cc.update(ctx_c); res["geometry"][key] = geo
        def self_mass(probs, lo, hi):
            pos = np.nonzero(seg == 0)[0][lo:hi]; return float(np.mean([probs[p, col[int(inst["ids"][node[p]])]] for p in pos]))
        probs0, _ = forward_multi(lab, ids, cand, [], ()); m0 = lab.seg_masses(probs0, seg, node, segs, col)[0]; ch = lab.seg_masses(probs0, seg, node, segs, col, inj_nbrs)[0]
        res["curves"][f"{key}|none"] = m0.tolist()
        res["rows"].append(dict(ctx=key, set=set_id, seed=s, cand="none", up_early=float(m0[EARLY[0]:EARLY[1]].mean()), up_late=float(m0[LATE[0]:LATE[1]].mean()), self_late=self_mass(probs0, *LATE),
                                ov_true=float(m0[OV[0]:OV[1]].mean()), ov_inj=float(ch[OV[0]:OV[1]].mean()), ov_chance=float(ch[OV[0]:OV[1]].mean())))
        for n, c in cc.items():
            probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, F, 0, 0, T_LONG), hs) for F, hs in c["hooks"]], ()); m = lab.seg_masses(probs, seg, node, segs, col)[0]
            probs_o, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, F[perm], 0, 0, T_LONG), hs) for F, hs in c["hooks"]], ())
            mt = lab.seg_masses(probs_o, seg, node, segs, col)[0]; mi = lab.seg_masses(probs_o, seg, node, segs, col, inj_nbrs)[0]
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
                hk = [(delta_window(lab, seg_u, node_u, F, 0, 0, T_UNSEEN + 1), hs) for F, hs in c["hooks"]]; Fys = []
                for F, hs in c["hooks"]:
                    Fy = F.copy(); Fy[X] = F[Y]; Fys.append((delta_window(lab, seg_u, node_u, Fy, 0, 0, T_UNSEEN + 1), hs))
                tx = masses(forward_multi(lab, ids_u, cand_u, hk, ())[0]); cy = masses(forward_multi(lab, ids_u, cand_u, Fys, ())[0])
                res["unseen"].append(dict(ctx=key, set=set_id, seed=s, X=X, Y=Y, cand=n, trueX_X=tx["P_nbrs_X"], trueX_Y=tx["P_nbrs_Y"], claimY_X=cy["P_nbrs_X"], claimY_Y=cy["P_nbrs_Y"]))
        r0 = [r for r in res["rows"] if r["ctx"] == key and r["cand"] == "none"][0]; u0 = np.mean([u["P_nbrs_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == "none"])
        def summ(n):
            r = [r for r in res["rows"] if r["ctx"] == key and r["cand"] == n][0]; ux = np.mean([u["trueX_X"] for u in res["unseen"] if u["ctx"] == key and u["cand"] == n])
            return f"{n}: up{r['up_early'] - r0['up_early']:+.3f} ov{r['ov_inj'] - r['ov_chance']:+.3f} un{ux - u0:+.3f}"
        print(f"[{out_name}] {key} ({(time.time() - t0) / 60:.1f} min): " + " | ".join(summ(n) for n in list(cc)[:10]), flush=True)
        json.dump(res, open(f"{OUT}/{out_name}.json", "w"))
    return res


def main():
    lab = Lab(); meta = json.load(open(f"{OUT_QK}/port.json"))["meta"]; nref = meta["norm_ref"]; late = [tuple(x) for x in meta["late"][:4]]
    S_late, Gn_late = qk_forms(lab, late)
    all_late = [(l, h) for l in range(H_MAIN, lab.n_layers) for h in range(NH)]; S_all, Gn_all = qk_forms(lab, all_late); V_all, ev_all = gen_eig(S_all, Gn_all)
    conds = {}; geo_static = {}
    def add(name, U, hs=H_MAIN, beta=4.0):
        conds[name] = dict(hooks=[(port_features(lab, U, nref) * beta, hs)]); geo_static[name] = dict(qk_late=qk_score(S_late, U), qk_all=qk_score(S_all, U), cos_qk_late=float(np.mean(principal_angles(U, dirs_at(V_late, [1, 2])))), cos_all_top2=float(np.mean(principal_angles(U, dirs_at(V_all, [1, 2])))))
    V_late, _ = gen_eig(S_late, Gn_late)
    add("qk_late", dirs_at(V_late, [1, 2])); add("qk_alllate", dirs_at(V_all, [1, 2]))
    for name, layers in (("qk_L12_25", range(12, 26)), ("qk_L26_33", range(26, 34)), ("qk_L34_41", range(34, 42)), ("qk_L12_41", range(12, 42)), ("qk_all672", range(0, 42))):
        S, Gn = qk_forms(lab, [(l, h) for l in layers for h in range(NH)]); U, ev = top_dirs(S, Gn, 2, +1); add(name, U); geo_static[name]["eig"] = ev
    P, evv, _ = vocab_pcs(lab)
    for k in (2048, 3072):
        U, ev = top_dirs(S_late, Gn_late, 2, +1, Q=torch.tensor(P[:k].T, device=DEV)); add(f"vocab_qk{k}", U); geo_static[f"vocab_qk{k}"]["eig"] = ev
    Pa, _, _ = act_pcs(lab, set_id=0, seed=0)
    U64_set0 = None
    for k in (8, 16, 32, 64, 128):
        U, ev = top_dirs(S_late, Gn_late, 2, +1, Q=torch.tensor(Pa[:k].T, device=DEV)); add(f"act_qk{k}", U); geo_static[f"act_qk{k}"]["eig"] = ev
        if k == 64: U64_set0 = U
    U, ev = top_dirs(S_all, Gn_all, 2, +1, Q=torch.tensor(Pa[:64].T, device=DEV)); add("act_qk64_Sall", U); geo_static["act_qk64_Sall"]["eig"] = ev
    Pa1, _, _ = act_pcs(lab, set_id=1, seed=0)
    U, ev = top_dirs(S_late, Gn_late, 2, +1, Q=torch.tensor(Pa1[:64].T, device=DEV)); add("act_qk64_set1", U); geo_static["act_qk64_set1"]["eig"] = ev
    geo_static["cos_act64_set0_set1"] = float(np.mean(principal_angles(U64_set0, U)))
    geo_static["cos_actpcs_set0_set1_top64"] = float(np.mean(principal_angles(Pa[:64], Pa1[:64])))
    # cone bases for the own-plane split
    ks = (16, 64, 256); bases = {}
    for k in ks:
        Q, _ = torch.linalg.qr(V_all[:, -k:]); bases[k] = Q.cpu().numpy().T.astype(np.float64)      # [k,d] orthonormal
    def ctx_conds(inst, M_own, P_own):
        F_own = (lab.Z2 @ M_own).astype(np.float64); cc = {"own_b2": dict(hooks=[(F_own.astype(np.float32) * 2, H_MAIN)])}; geo = {}
        for k in ks:
            B = bases[k]; F_in = (F_own @ B.T) @ B; F_out = F_own - F_in
            cc[f"own_in{k}_b2"] = dict(hooks=[(F_in.astype(np.float32) * 2, H_MAIN)]); cc[f"own_out{k}_b2"] = dict(hooks=[(F_out.astype(np.float32) * 2, H_MAIN)])
            geo[f"frac_in{k}"] = float(np.sum(F_in ** 2) / np.sum(F_own ** 2))
            # in-cone part rescaled to the full own norm (is the in-cone part alone, at full strength, enough?)
            cc[f"own_in{k}_full_b2"] = dict(hooks=[((F_in * np.linalg.norm(F_own, axis=1).mean() / np.linalg.norm(F_in, axis=1).mean()).astype(np.float32) * 2, H_MAIN)])
        geo["own_qk_all"] = qk_score(S_all, P_own); geo["own_qk_late"] = qk_score(S_late, P_own)
        return cc, geo
    run(lab, conds, ctx_conds, "round3", extra_meta=dict(nref=nref, ks=list(ks), geo_static=geo_static))


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"DONE in {(time.time() - t0) / 60:.1f} min")
