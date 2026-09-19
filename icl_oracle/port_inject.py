"""Can a stable injection PORT be established in-context?

P1  Priming alignment (no injection): does grid B's position plane align with grid A's when A precedes
    B in the same context? If so, calibrate on A in-context and inject GT into B through that plane.
P2  Seeding: inject a chosen plane P during the first N tokens only, then stop. Does the model's OWN
    plane (measured later, uninjected) align with P? If so, P is now a port: test late injections via P
    (uptake on tokens 600-1200, unseen-node) with vs. without the seed.
P4  Withdrawal: inject the context's own-plane GT for the first N tokens, then stop; does the model's
    subsequent learning proceed normally, stall, or benefit?

Reuses gt_inject.Lab and helpers; needs results_gt/e1.json + e1_means.npz (run gt_inject --stage e1).
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gt_inject import (Lab, random_walk, cand_map, nanfill, fit_M, top_pcs, principal_angles,
                       calibrate_M, gt_features, random_features, shared_subspace, far_node,
                       T_LONG, T_DEMO, T_QUERY, T_UNSEEN, N, DEV)

OUT = os.environ.get("ICL_OUT_PORT", os.path.join(HERE, "results_port"))
os.makedirs(OUT, exist_ok=True)
LAYERS = (16, 20, 26, 30)


def plane_of(lab, hs, seg, node, s, win=200):
    H = nanfill(lab.node_means(hs, seg, node, s, win).astype(np.float64)); Hc = H - H.mean(0)
    return top_pcs(Hc, 2), Hc


def cosP(A, B):
    return float(np.mean(principal_angles(A, B)))


def delta_window(lab, seg, node, F, s, lo, hi):
    """F [16,d] on steps [lo,hi) of segment s (positions = steps within that segment)."""
    delta = np.zeros((len(seg), lab.d), np.float32)
    pos = np.nonzero(seg == s)[0]
    for i, p in enumerate(pos):
        if lo <= i < hi:
            delta[p] = F[node[p]]
    return torch.tensor(delta, dtype=torch.bfloat16, device=DEV)


# ============================================================================= P1
def stage_p1(lab, h_main, seeds=6):
    res = {"align": [], "align4": [], "same_words": [], "dense": {"none": [], "none_ctx": [], "M_A_ctx": [], "M_A_alone": [], "M_B_self": []}}
    for s in range(seeds):
        instA, instB = lab.instance(1, s), lab.instance(4, s)
        walkA = random_walk(lab.nbrs, T_DEMO, np.random.default_rng(6000 + 100 * s))
        walkB = random_walk(lab.nbrs, T_QUERY, np.random.default_rng(7000 + s))
        segA, segB, segAB = [(instA, walkA)], [(instB, walkB)], [(instA, walkA), (instB, walkB)]
        # --- alone
        ids, seg, node = lab.build_context(segA); cand, col = cand_map([instA])
        _, hsA = lab.forward(ids, cand, hs_layers=LAYERS)
        PA_alone = {l: plane_of(lab, hsA[l], seg, node, 0)[0] for l in LAYERS}
        HA_alone = plane_of(lab, hsA[h_main], seg, node, 0)[1]
        ids, seg, node = lab.build_context(segB); cand, col = cand_map([instB])
        probsB, hsB = lab.forward(ids, cand, hs_layers=LAYERS)
        PB_alone = {l: plane_of(lab, hsB[l], seg, node, 0)[0] for l in LAYERS}
        res["dense"]["none"].append(lab.seg_masses(probsB, seg, node, segB, col)[0].tolist())
        # --- together
        ids, seg, node = lab.build_context(segAB); cand, col = cand_map([instA, instB])
        probs, hs = lab.forward(ids, cand, hs_layers=LAYERS)
        PA_ctx = {l: plane_of(lab, hs[l], seg, node, 0)[0] for l in LAYERS}
        PB_ctx = {l: plane_of(lab, hs[l], seg, node, 1)[0] for l in LAYERS}
        HA_ctx = plane_of(lab, hs[h_main], seg, node, 0)[1]
        res["dense"]["none_ctx"].append(lab.seg_masses(probs, seg, node, segAB, col)[1].tolist())
        res["align"].append({str(l): dict(B_ctx_vs_A_ctx=cosP(PB_ctx[l], PA_ctx[l]), B_alone_vs_A_alone=cosP(PB_alone[l], PA_alone[l]),
                                          B_ctx_vs_B_alone=cosP(PB_ctx[l], PB_alone[l]), A_ctx_vs_A_alone=cosP(PA_ctx[l], PA_alone[l]))
                             for l in LAYERS})
        # --- inject GT into B through A's in-context plane / A's alone plane / B's own calibrated plane
        M_ctx, _ = fit_M(HA_ctx, lab.Z2); M_alone, _ = fit_M(HA_alone, lab.Z2)
        walk_cal = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s))
        idc, sgc, ndc = lab.build_context([(instB, walk_cal)]); cc, _ = cand_map([instB])
        _, hsc = lab.forward(idc, cc, hs_layers=(h_main,))
        M_self, _ = fit_M(plane_of(lab, hsc[h_main], sgc, ndc, 0)[1], lab.Z2)
        nrm = np.linalg.norm(lab.Z2 @ M_self, axis=1).mean()
        for name, M in (("M_A_ctx", M_ctx), ("M_A_alone", M_alone), ("M_B_self", M_self)):
            F = (lab.Z2 @ M); F = (F * nrm / np.linalg.norm(F, axis=1).mean() * 2).astype(np.float32)
            ids, seg, node = lab.build_context(segAB); cand, col = cand_map([instA, instB])
            delta = delta_window(lab, seg, node, F, 1, 0, T_QUERY)
            probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=h_main)
            res["dense"][name].append(lab.seg_masses(probs, seg, node, segAB, col)[1].tolist())
        # --- 4 grids in one context: pairwise plane alignment at each layer
        segs = [(lab.instance(k, s), random_walk(lab.nbrs, T_DEMO, np.random.default_rng(6000 + 100 * s + k))) for k in (1, 2, 3)]
        segs.append((instB, walkB))
        ids, seg, node = lab.build_context(segs); cand, col = cand_map([i for i, _ in segs])
        _, hs = lab.forward(ids, cand, hs_layers=LAYERS)
        P4 = {l: [plane_of(lab, hs[l], seg, node, k)[0] for k in range(4)] for l in LAYERS}
        res["align4"].append({str(l): [[cosP(P4[l][i], P4[l][j]) for j in range(4)] for i in range(4)] for l in LAYERS})
        # --- same words, different placement, in one context
        instA2 = lab.instance(4, 100 + s)   # same word set as B, different assignment
        walkA2 = random_walk(lab.nbrs, T_DEMO, np.random.default_rng(6500 + s))
        ids, seg, node = lab.build_context([(instA2, walkA2), (instB, walkB)]); cand, col = cand_map([instA2, instB])
        _, hs = lab.forward(ids, cand, hs_layers=LAYERS)
        res["same_words"].append({str(l): dict(B_ctx_vs_A2_ctx=cosP(plane_of(lab, hs[l], seg, node, 1)[0], plane_of(lab, hs[l], seg, node, 0)[0]),
                                               B_ctx_vs_B_alone=cosP(plane_of(lab, hs[l], seg, node, 1)[0], PB_alone[l])) for l in LAYERS})
        a = res["align"][-1][str(h_main)]
        print(f"[P1] seed {s} hs{h_main}: B|A vs A: {a['B_ctx_vs_A_ctx']:.2f} (alone {a['B_alone_vs_A_alone']:.2f}); B|A vs B alone {a['B_ctx_vs_B_alone']:.2f}; "
              f"same-words diff-place {res['same_words'][-1][str(h_main)]['B_ctx_vs_A2_ctx']:.2f} | uptake@200: " +
              " ".join(f"{k}={np.mean(v[-1][150:250]):.2f}" for k, v in res["dense"].items()))
    json.dump(res, open(f"{OUT}/p1.json", "w"))


# ============================================================================= P2
def stage_p2(lab, h_main, seeds=6, query_set=4):
    M0, _ = calibrate_M(lab, h_main, (0,))
    F_for = gt_features(lab, M0); norm_gt = float(np.linalg.norm(F_for, axis=1).mean())
    Qsh, _ = shared_subspace(lab, h_main, exclude_set=query_set)
    R2, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((Qsh.shape[1], 2)))
    F_sh = lab.Z2 @ (Qsh @ R2).T; F_sh = (F_sh * norm_gt / np.linalg.norm(F_sh, axis=1).mean()).astype(np.float32)
    F_rd = random_features(lab, norm_gt, seed=321)
    planes = {"foreign": F_for.astype(np.float32), "rand_shared": F_sh, "random": F_rd}
    plane_rows = {k: top_pcs(v.astype(np.float64) - v.astype(np.float64).mean(0), 2) for k, v in planes.items()}
    res = {"meta": dict(h_main=h_main, norm_gt=norm_gt), "runs": []}
    LATE = (600, T_LONG)
    for s in range(seeds):
        inst = lab.instance(query_set, s)
        walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))   # same walks as E2/E2b
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        def run(delta, want_planes=True):
            probs, hs = lab.forward(ids, cand, hs_layers=LAYERS if want_planes else (), delta=delta, inject_hs=h_main)
            mass = lab.seg_masses(probs, seg, node, segs, col)[0]
            late = {str(l): plane_of(lab, hs[l], seg, node, 0, win=200)[0] for l in LAYERS} if want_planes else {}
            return mass, late
        mass0, late0 = run(None)
        base = dict(cond="none", plane=None, seed_beta=0, n_seed=0, late_inject=False, mass=mass0.tolist(),
                    cos_plane_vs_P={}, cos_plane_vs_none={str(l): 1.0 for l in LAYERS}, seed=s)
        res["runs"].append(base)
        for pname, F in planes.items():
            P = plane_rows[pname]
            base["cos_plane_vs_P"][pname] = {str(l): cosP(late0[str(l)], P) for l in LAYERS}
            conds = []
            for sb in (2.0, 4.0):
                for ns in (200, 400):
                    conds.append(dict(seed_beta=sb, n_seed=ns, late_inject=False, seed_hs=h_main))
                conds.append(dict(seed_beta=sb, n_seed=200, late_inject=True, seed_hs=h_main))
            conds.append(dict(seed_beta=0.0, n_seed=0, late_inject=True, seed_hs=h_main))          # late injection only
            conds.append(dict(seed_beta=4.0, n_seed=200, late_inject=False, seed_hs=16))          # seed at an earlier layer
            for c in conds:
                delta = torch.zeros((len(seg), lab.d), dtype=torch.bfloat16, device=DEV)
                if c["n_seed"]:
                    delta += delta_window(lab, seg, node, F * c["seed_beta"], 0, 0, c["n_seed"])
                if c["late_inject"]:
                    delta += delta_window(lab, seg, node, F * 2.0, 0, LATE[0], LATE[1])
                if c["seed_hs"] != h_main and c["n_seed"]:
                    # seed at a different layer: two hooks needed -> run with seed hook only (no late injection)
                    probs, hs = lab.forward(ids, cand, hs_layers=LAYERS, delta=delta, inject_hs=c["seed_hs"])
                    mass = lab.seg_masses(probs, seg, node, segs, col)[0]
                    late = {str(l): plane_of(lab, hs[l], seg, node, 0, win=200)[0] for l in LAYERS}
                else:
                    mass, late = run(delta)
                res["runs"].append(dict(cond=f"{pname}|sb{c['seed_beta']}|n{c['n_seed']}|late{int(c['late_inject'])}|hs{c['seed_hs']}",
                                        plane=pname, seed_beta=c["seed_beta"], n_seed=c["n_seed"], late_inject=c["late_inject"], seed_hs=c["seed_hs"],
                                        mass=mass.tolist(), cos_plane_vs_P={str(l): cosP(late[str(l)], P) for l in LAYERS},
                                        cos_plane_vs_none={str(l): cosP(late[str(l)], late0[str(l)]) for l in LAYERS}, seed=s))
        r = [x for x in res["runs"] if x["seed"] == s]
        def g(cond): return [x for x in r if x["cond"] == cond][0]
        print(f"[P2] seed {s}: cos(late plane, P) @hs{h_main}: " + " | ".join(
            f"{p}: none {base['cos_plane_vs_P'][p][str(h_main)]:.2f} seed2/200 {g(f'{p}|sb2.0|n200|late0|hs{h_main}')['cos_plane_vs_P'][str(h_main)]:.2f} "
            f"seed4/400 {g(f'{p}|sb4.0|n400|late0|hs{h_main}')['cos_plane_vs_P'][str(h_main)]:.2f}" for p in planes) +
            f" || late mass none {np.mean(mass0[LATE[0]:]):.2f} " + " ".join(
            f"{p}: lateInj {np.mean(g(f'{p}|sb0.0|n0|late1|hs{h_main}')['mass'][LATE[0]:]):.2f} seed+lateInj {np.mean(g(f'{p}|sb2.0|n200|late1|hs{h_main}')['mass'][LATE[0]:]):.2f}" for p in planes))
        json.dump(res, open(f"{OUT}/p2.json", "w"))
    # --- unseen node after seeding (foreign and random planes)
    rows = []
    for s in range(seeds):
        inst = lab.instance(query_set, s); rng = np.random.default_rng(8000 + s)
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
            segs = [(inst, walk_q)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])),
                            P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            rows.append(dict(seed=s, X=X, Y=Y, plane="-", cond="none", **masses(lab.forward(ids, cand)[0])))
            for pname in ("foreign", "random"):
                F = planes[pname]
                for cname, sb, xf in (("seed_only", 4.0, False), ("Xfeat_only", 0.0, True), ("seed+Xfeat", 4.0, True)):
                    delta = torch.zeros((len(seg), lab.d), dtype=torch.bfloat16, device=DEV)
                    if sb: delta += delta_window(lab, seg, node, F * sb, 0, 0, 200)
                    if xf: delta[-1] += torch.tensor(F[X] * 2.0, dtype=torch.bfloat16, device=DEV)
                    rows.append(dict(seed=s, X=X, Y=Y, plane=pname, cond=cname, **masses(lab.forward(ids, cand, delta=delta, inject_hs=h_main)[0])))
        print(f"[P2-unseen] seed {s}: " + " | ".join(f"{p}/{c}: {np.mean([r['P_nbrs_X'] for r in rows if r['seed']==s and r['plane']==p and r['cond']==c]):.3f}"
                                                  for p, c in (("-", "none"), ("foreign", "Xfeat_only"), ("foreign", "seed+Xfeat"), ("random", "seed+Xfeat"))))
    res["unseen"] = rows
    json.dump(res, open(f"{OUT}/p2.json", "w"))


# ============================================================================= P4
def stage_p4(lab, h_main, seeds=6, query_set=4):
    res = {"dense": {}}
    for s in range(seeds):
        inst = lab.instance(query_set, s)
        walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        walk_cal = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s))
        idc, sgc, ndc = lab.build_context([(inst, walk_cal)])
        _, hsc = lab.forward(idc, cand, hs_layers=(h_main,))
        M_self, _ = fit_M(plane_of(lab, hsc[h_main], sgc, ndc, 0)[1], lab.Z2)
        F = (lab.Z2 @ M_self * 2.0).astype(np.float32)
        for name, lo, hi in (("none", 0, 0), ("self_all", 0, T_LONG), ("self_0_200", 0, 200), ("self_0_400", 0, 400), ("self_0_800", 0, 800)):
            delta = delta_window(lab, seg, node, F, 0, lo, hi) if hi else None
            probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=h_main)
            res["dense"].setdefault(name, []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
        print(f"[P4] seed {s}: " + " ".join(f"{k}: [400-600]={np.mean(v[-1][400:600]):.2f} [800-1200]={np.mean(v[-1][800:]):.2f}" for k, v in res["dense"].items()))
    json.dump(res, open(f"{OUT}/p4.json", "w"))



# ============================================================================= multi-hook forward
@torch.inference_mode()
def forward_multi(lab, ids, cand_ids, hooks, hs_layers=()):
    """hooks: list of (delta [T,d] bf16, inject_hs). Same semantics as Lab.forward but several layers."""
    handles = []
    for delta, h in hooks:
        def fn(mod, inp, out, delta=delta):
            if isinstance(out, tuple):
                out[0].add_(delta[None]); return out
            out.add_(delta[None]); return out
        handles.append(lab.model.model.layers[h - 1].register_forward_hook(fn))
    try:
        out = lab.model(torch.tensor(ids, device=DEV)[None], output_hidden_states=bool(hs_layers))
    finally:
        for hd in handles: hd.remove()
    probs = torch.softmax(out.logits[0].float(), -1)[:, torch.tensor(cand_ids, device=DEV)]
    hs = {h: out.hidden_states[h][0].float().cpu().numpy() for h in hs_layers}
    return probs.cpu().numpy(), hs


def make_planes(lab, h_main, query_set=4):
    M0, _ = calibrate_M(lab, h_main, (0,))
    F_for = gt_features(lab, M0); norm_gt = float(np.linalg.norm(F_for, axis=1).mean())
    Qsh, _ = shared_subspace(lab, h_main, exclude_set=query_set)
    R2, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((Qsh.shape[1], 2)))
    F_sh = lab.Z2 @ (Qsh @ R2).T; F_sh = (F_sh * norm_gt / np.linalg.norm(F_sh, axis=1).mean()).astype(np.float32)
    F_rd = random_features(lab, norm_gt, seed=321)
    planes = {"foreign": F_for.astype(np.float32), "rand_shared": F_sh, "random": F_rd}
    rows = {k: top_pcs(v.astype(np.float64) - v.astype(np.float64).mean(0), 2) for k, v in planes.items()}
    return planes, rows, norm_gt


# ============================================================================= P2b: seed at EARLY layers
def stage_p2b(lab, h_main, seeds=6, query_set=4):
    planes, rows, norm_gt = make_planes(lab, h_main, query_set)
    RL = (20, 26, 30); LATE = (600, T_LONG); NS = 300
    res = {"meta": dict(h_main=h_main, n_seed=NS), "runs": [], "unseen": []}
    for s in range(seeds):
        inst = lab.instance(query_set, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        probs, hs = forward_multi(lab, ids, cand, [], RL)
        mass0 = lab.seg_masses(probs, seg, node, segs, col)[0]
        late0 = {l: plane_of(lab, hs[l], seg, node, 0, 200)[0] for l in RL}
        res["runs"].append(dict(seed=s, cond="none", plane=None, seed_hs=None, seed_beta=0, late_inject=False, mass=mass0.tolist(),
                                cos_P={p: {str(l): cosP(late0[l], rows[p]) for l in RL} for p in planes}, cos_none={str(l): 1.0 for l in RL}))
        for pname, F in planes.items():
            for seed_hs in (4, 8, 16):
                for sb in (2.0, 4.0, 8.0):
                    for late_inject in ((False, True) if (seed_hs == 8 and sb == 4.0) else (False,)):
                        hooks = [(delta_window(lab, seg, node, F * sb, 0, 0, NS), seed_hs)]
                        if late_inject: hooks.append((delta_window(lab, seg, node, F * 2.0, 0, LATE[0], LATE[1]), h_main))
                        probs, hs = forward_multi(lab, ids, cand, hooks, RL)
                        mass = lab.seg_masses(probs, seg, node, segs, col)[0]
                        late = {l: plane_of(lab, hs[l], seg, node, 0, 200)[0] for l in RL}
                        res["runs"].append(dict(seed=s, cond=f"{pname}|hs{seed_hs}|sb{sb}|late{int(late_inject)}", plane=pname, seed_hs=seed_hs, seed_beta=sb,
                                                late_inject=late_inject, mass=mass.tolist(), cos_P={str(l): cosP(late[l], rows[pname]) for l in RL},
                                                cos_none={str(l): cosP(late[l], late0[l]) for l in RL}))
            # late injection only (reference)
            hooks = [(delta_window(lab, seg, node, F * 2.0, 0, LATE[0], LATE[1]), h_main)]
            probs, _ = forward_multi(lab, ids, cand, hooks, ())
            res["runs"].append(dict(seed=s, cond=f"{pname}|lateonly", plane=pname, seed_hs=None, seed_beta=0, late_inject=True,
                                    mass=lab.seg_masses(probs, seg, node, segs, col)[0].tolist(), cos_P={}, cos_none={}))
        rs = [r for r in res["runs"] if r["seed"] == s]
        g = lambda c: [r for r in rs if r["cond"] == c][0]
        print(f"[P2b] seed {s}: cos(late plane@26, P): " + " | ".join(
            f"{p}: none {rs[0]['cos_P'][p]['26']:.2f} " + " ".join(f"hs{h}/b{b:g}={g(f'{p}|hs{h}|sb{b}|late0')['cos_P']['26']:.2f}" for h in (4, 8, 16) for b in (4.0, 8.0)) for p in planes) +
            f" || late mass: none {np.mean(mass0[600:]):.2f} " + " ".join(f"{p}: lateonly {np.mean(g(f'{p}|lateonly')['mass'][600:]):.2f} seed(hs8,b4)+late {np.mean(g(f'{p}|hs8|sb4.0|late1')['mass'][600:]):.2f}" for p in planes))
        json.dump(res, open(f"{OUT}/p2b.json", "w"))
    # --- unseen node with early seeding (hs8, beta 4, 300 tok) + X feature via P at hs26
    rows_u = []
    for s in range(seeds):
        inst = lab.instance(query_set, s); rng = np.random.default_rng(8000 + s)
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
            segs = [(inst, walk_q)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            rows_u.append(dict(seed=s, X=X, Y=Y, plane="-", cond="none", **masses(forward_multi(lab, ids, cand, [], ())[0])))
            for pname in ("foreign", "random"):
                F = planes[pname]
                xf = torch.zeros((len(seg), lab.d), dtype=torch.bfloat16, device=DEV); xf[-1] = torch.tensor(F[X] * 2.0, dtype=torch.bfloat16, device=DEV)
                for cname, hooks in (("seed_only", [(delta_window(lab, seg, node, F * 4.0, 0, 0, NS), 8)]), ("Xfeat_only", [(xf, h_main)]),
                                     ("seed+Xfeat", [(delta_window(lab, seg, node, F * 4.0, 0, 0, NS), 8), (xf, h_main)]),
                                     ("seed+Xfeat@hs8", [(delta_window(lab, seg, node, F * 4.0, 0, 0, NS), 8), (xf * 2, 8)])):
                    rows_u.append(dict(seed=s, X=X, Y=Y, plane=pname, cond=cname, **masses(forward_multi(lab, ids, cand, hooks, ())[0])))
        print(f"[P2b-unseen] seed {s}: " + " | ".join(f"{p}/{c}: {np.mean([r['P_nbrs_X'] for r in rows_u if r['seed']==s and r['plane']==p and r['cond']==c]):.3f}"
                                                   for p, c in (("-", "none"), ("foreign", "Xfeat_only"), ("foreign", "seed+Xfeat"), ("foreign", "seed+Xfeat@hs8"), ("random", "seed+Xfeat"))))
    res["unseen"] = rows_u
    json.dump(res, open(f"{OUT}/p2b.json", "w"))


# ============================================================================= P3: demos with EARLY-layer injection
def stage_p3(lab, h_main, seeds=6, query_set=4, demo_sets=(1, 2, 3)):
    planes, rows, norm_gt = make_planes(lab, h_main, query_set)
    res = {"meta": dict(h_main=h_main), "curves": {}}
    def build(s, n_demos):
        segs = [(lab.instance(demo_sets[k], s), random_walk(lab.nbrs, T_DEMO, np.random.default_rng(6000 + 100 * s + k))) for k in range(n_demos)]
        segs.append((lab.instance(query_set, s), random_walk(lab.nbrs, T_QUERY, np.random.default_rng(7000 + s))))
        return segs
    def run(name, n_demos, demo_inj, query_inj, F, inj_hs, beta):
        per = []
        for s in range(seeds):
            segs = build(s, n_demos); ids, seg, node = lab.build_context(segs); cand, col = cand_map([i for i, _ in segs])
            hooks = []
            if F is not None:
                delta = torch.zeros((len(seg), lab.d), dtype=torch.bfloat16, device=DEV)
                for k in range(n_demos):
                    if demo_inj: delta += delta_window(lab, seg, node, F * beta, k, 0, T_DEMO)
                if query_inj: delta += delta_window(lab, seg, node, F * beta, n_demos, 0, T_QUERY)
                hooks = [(delta, inj_hs)]
            probs, _ = forward_multi(lab, ids, cand, hooks, ())
            per.append([m.tolist() for m in lab.seg_masses(probs, seg, node, segs, col)])
        q = np.array([p[-1] for p in per]).mean(0)
        print(f"[P3] {name}: query mass@[100,200,400,600]=" + " ".join(f"{q[min(t, len(q))-1]:.2f}" for t in (100, 200, 400, 600)))
        res["curves"][name] = per
    run("none_demos3", 3, False, False, None, h_main, 0)
    run("none_demos0", 0, False, False, None, h_main, 0)
    for pname in ("foreign", "random"):
        F = planes[pname]
        for inj_hs in (8, 16):
            for beta in (2.0, 4.0):
                run(f"{pname}|hs{inj_hs}|b{beta}|demos3_all", 3, True, True, F, inj_hs, beta)
                run(f"{pname}|hs{inj_hs}|b{beta}|demos3_queryOnly", 3, False, True, F, inj_hs, beta)
                run(f"{pname}|hs{inj_hs}|b{beta}|demos0", 0, False, True, F, inj_hs, beta)
        json.dump(res, open(f"{OUT}/p3.json", "w"))
    json.dump(res, open(f"{OUT}/p3.json", "w"))



# ============================================================================= P5: random-plane port sweep (calibration-free)
def stage_p5(lab, h_main, seeds=6, query_set=4):
    planes, rows, norm_gt = make_planes(lab, h_main, query_set)
    F_r2 = random_features(lab, norm_gt, seed=654)
    cands = {"random": planes["random"], "random2": F_r2}
    d_e2 = json.load(open(f"{HERE}/results_gt/e2.json")) if os.path.exists(f"{HERE}/results_gt/e2.json") else None
    perm = np.array(d_e2["override"]["b2.0"]["perm"]) if d_e2 else np.random.default_rng(99).permutation(N)
    inv = np.argsort(perm)
    def inj_nbrs(s, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    res = {"meta": dict(h_main=h_main, norm_gt=norm_gt, perm=perm.tolist()), "dense": {}, "override": {}, "unseen": []}
    grid = [(h, b) for h in (12, 16, 20) for b in (4.0, 8.0, 16.0)]
    for s in range(seeds):
        inst = lab.instance(query_set, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        probs, _ = forward_multi(lab, ids, cand, [], ())
        res["dense"].setdefault("none", []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
        res["override"].setdefault("none_injmap", []).append(lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist())
        for pname, F in cands.items():
            for h, b in grid:
                probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, F * b, 0, 0, T_LONG), h)], ())
                res["dense"].setdefault(f"{pname}|hs{h}|b{b:g}", []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
                if pname == "random":   # override: permuted GT through the random plane
                    Fp = (lab.Z2[perm] @ rows["random"]) ; Fp = (Fp * norm_gt / np.linalg.norm(Fp, axis=1).mean()).astype(np.float32)
                    probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, Fp * b, 0, 0, T_LONG), h)], ())
                    res["override"].setdefault(f"random|hs{h}|b{b:g}", []).append(dict(
                        true_map=lab.seg_masses(probs, seg, node, segs, col)[0].tolist(), injected_map=lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist()))
        print(f"[P5] seed {s}: none [200-600]={np.mean(res['dense']['none'][-1][200:600]):.3f} | " + " ".join(
            f"{k.split('|',1)[1]}={np.mean(v[-1][200:600]):.3f}" for k, v in res["dense"].items() if k.startswith("random|")))
        json.dump(res, open(f"{OUT}/p5.json", "w"))
    # --- unseen node through the random plane at (hs16, b8) and (hs16, b4), (hs12, b8)
    rows_u = []
    for s in range(seeds):
        inst = lab.instance(query_set, s); rng = np.random.default_rng(8000 + s); F = cands["random"]
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
            segs = [(inst, walk_q)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            rows_u.append(dict(seed=s, X=X, Y=Y, cond="none", **masses(forward_multi(lab, ids, cand, [], ())[0])))
            for h, b in ((16, 4.0), (16, 8.0), (12, 8.0), (20, 8.0)):
                full = delta_window(lab, seg, node, F * b, 0, 0, T_UNSEEN + 1)
                Fy = F.copy(); Fy[X] = F[Y]; fully = delta_window(lab, seg, node, Fy * b, 0, 0, T_UNSEEN + 1)
                rows_u.append(dict(seed=s, X=X, Y=Y, cond=f"trueX|hs{h}|b{b:g}", **masses(forward_multi(lab, ids, cand, [(full, h)], ())[0])))
                rows_u.append(dict(seed=s, X=X, Y=Y, cond=f"claimY|hs{h}|b{b:g}", **masses(forward_multi(lab, ids, cand, [(fully, h)], ())[0])))
        print(f"[P5-unseen] seed {s}: " + " | ".join(f"{c}: X={np.mean([r['P_nbrs_X'] for r in rows_u if r['seed']==s and r['cond']==c]):.3f} Y={np.mean([r['P_nbrs_Y'] for r in rows_u if r['seed']==s and r['cond']==c]):.3f}"
                                                 for c in ("none", "trueX|hs16|b8", "claimY|hs16|b8")))
    res["unseen"] = rows_u
    json.dump(res, open(f"{OUT}/p5.json", "w"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["all", "p1", "p2", "p4", "p2b", "p3", "p5"])
    ap.add_argument("--seeds", type=int, default=6); ap.add_argument("--h_main", type=int, default=None)
    args = ap.parse_args(); t0 = time.time()
    lab = Lab()
    h_main = args.h_main or json.load(open(f"{HERE}/results_gt/e1.json"))["h_energy"]
    print(f"[main] h_main={h_main}")
    if args.stage in ("all", "p1"): stage_p1(lab, h_main, args.seeds)
    if args.stage in ("all", "p2"): stage_p2(lab, h_main, args.seeds)
    if args.stage in ("all", "p4"): stage_p4(lab, h_main, args.seeds)
    if args.stage == "p2b": stage_p2b(lab, h_main, args.seeds)
    if args.stage == "p3": stage_p3(lab, h_main, args.seeds)
    if args.stage == "p5": stage_p5(lab, h_main, args.seeds)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")
