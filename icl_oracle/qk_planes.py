"""Are ports built from different late heads the same plane? Principal angles between 2-plane ports from
reader heads, random late heads, random mid heads, random early heads; plus a quick uptake/unseen test of
ports derived from each family, all injected at hs26 (beta 4)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gt_inject import Lab, random_walk, cand_map, principal_angles, far_node, T_LONG, T_UNSEEN, N
from port_inject import forward_multi, delta_window
from qk_port import build_port, port_features, OUT, NH

lab = Lab(); h_main = 26
pr = json.load(open(f"{OUT}/port.json")); nref = pr["meta"]["norm_ref"]; late = [tuple(x) for x in pr["meta"]["late"][:4]]
rng = np.random.default_rng(11)
sets = {"readers_late": late}
for i in range(4): sets[f"rand_late_{i}"] = [(int(l), int(h)) for l, h in zip(rng.integers(26, 42, 4), rng.integers(0, NH, 4))]
for i in range(3): sets[f"rand_mid_{i}"] = [(int(l), int(h)) for l, h in zip(rng.integers(16, 26, 4), rng.integers(0, NH, 4))]
for i in range(3): sets[f"rand_early_{i}"] = [(int(l), int(h)) for l, h in zip(rng.integers(2, 13, 4), rng.integers(0, NH, 4))]
U = {k: build_port(lab, v, k=2, fast_penalty=1.0)[0] for k, v in sets.items()}
names = list(sets)
print("cos(principal angles) between port planes:")
for a in names:
    print(f"{a:14s} " + " ".join(f"{np.mean(principal_angles(U[a], U[b])):.2f}" for b in names))
res = {"planes": {a: {b: float(np.mean(principal_angles(U[a], U[b]))) for b in names} for a in names}, "sets": {k: v for k, v in sets.items()}, "func": []}
F = {k: port_features(lab, U[k], nref) for k in ("readers_late", "rand_late_0", "rand_mid_0", "rand_early_0")}
for s in range(3):
    inst = lab.instance(4, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s)); segs = [(inst, walk)]
    ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
    probs, _ = forward_multi(lab, ids, cand, [], ()); base = lab.seg_masses(probs, seg, node, segs, col)[0]
    row = dict(seed=s, uptake={"none": float(base[100:300].mean())}, unseen={})
    for k, Fk in F.items():
        probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, Fk * 4, 0, 0, T_LONG), h_main)], ())
        row["uptake"][k] = float(lab.seg_masses(probs, seg, node, segs, col)[0][100:300].mean())
    rngu = np.random.default_rng(8000 + s)
    for X in rngu.choice(N, 3, replace=False):
        X = int(X); Y = far_node(lab, X, rngu)
        walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rngu, avoid=X), [X]]); segs_u = [(inst, walk_q)]
        ids_u, seg_u, node_u = lab.build_context(segs_u); cand_u, col_u = cand_map([inst])
        def mx(probs):
            p = probs[-1]; return float(sum(p[col_u[int(inst["ids"][q])]] for q in lab.nbrs[X]))
        row["unseen"].setdefault("none", []).append(mx(forward_multi(lab, ids_u, cand_u, [], ())[0]))
        for k, Fk in F.items():
            row["unseen"].setdefault(k, []).append(mx(forward_multi(lab, ids_u, cand_u, [(delta_window(lab, seg_u, node_u, Fk * 4, 0, 0, T_UNSEEN + 1), h_main)], ())[0]))
    res["func"].append(row)
    print(f"[planes] seed {s}: uptake " + " ".join(f"{k}={v:.3f}" for k, v in row["uptake"].items()) + " | unseen P(nbrs X): " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in row["unseen"].items()))
json.dump(res, open(f"{OUT}/planes.json", "w"))
