"""QK-space port: a universal injection channel defined by the attention weights that READ position.

screen    per-head attention from node tokens (grid learned) to: same-node occurrences, neighbor occurrences,
          other nodes, successor-of-same (induction targets), successor-of-neighbor, BOS. Enrichment over base
          rate. Baseline vs own-plane injection (hs26, b2) vs random plane (hs16, b4): heads whose
          neighbor-attention rises under injection are the position READERS.
ablate    zero candidate reader heads (and control groups); does the injection gain disappear?
port      build directions from the readers' W_Q/W_K (RoPE-slow dims, layernorm folded in) so that injected
          coordinates add adjacency-structured attention logits in every reader head at once. Compare with
          random plane / own plane / anti-port / non-reader port on uptake, override, unseen-node, and on
          the readers' attention itself.
universal same port, new word sets and assignments, no recalibration.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gt_inject import (Lab, random_walk, cand_map, nanfill, fit_M, top_pcs, calibrate_M, gt_features,
                       random_features, far_node, T_LONG, T_UNSEEN, N, DEV)
from port_inject import forward_multi, delta_window, plane_of

OUT = os.environ.get("ICL_OUT_QK", os.path.join(HERE, "results_qk"))
os.makedirs(OUT, exist_ok=True)
NH, NKV, HD = 16, 8, 256
CATS = ["same", "nbr", "other", "succ_same", "succ_nbr", "bos"]


# ----------------------------------------------------------------------------- features
def own_plane_features(lab, inst, s, h_main):
    walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s))
    ids, seg, node = lab.build_context([(inst, walk)]); cand, _ = cand_map([inst])
    _, hs = lab.forward(ids, cand, hs_layers=(h_main,))
    M, _ = fit_M(plane_of(lab, hs[h_main], seg, node, 0)[1], lab.Z2)
    return (lab.Z2 @ M).astype(np.float32)


def norm_ref(lab, h_main):
    M0, _ = calibrate_M(lab, h_main, (0,))
    return float(np.linalg.norm(gt_features(lab, M0), axis=1).mean())


# ----------------------------------------------------------------------------- attention capture
class AttnStats:
    """hooks every self_attn; reduces attention weights from query set Q into category masses."""
    def __init__(self, lab, masks, Qidx, base):
        self.lab, self.masks, self.Qidx, self.base = lab, masks, Qidx, base   # masks [C,T,T] bf16 on DEV
        self.mass = torch.zeros((lab.n_layers, NH, len(CATS)), device=DEV)
        self.handles = []
        for l, layer in enumerate(lab.model.model.layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._make(l)))

    def _make(self, l):
        def fn(mod, inp, out):
            w = out[1]
            if w is None: return
            aq = w[0][:, self.Qidx, :].float()                       # [NH, |Q|, T]
            for c in range(len(CATS)):
                self.mass[l, :, c] = (aq * self.masks[c][self.Qidx][None]).sum((1, 2)) / len(self.Qidx)
        return fn

    def remove(self):
        for h in self.handles: h.remove()

    def result(self):
        m = self.mass.cpu().numpy()
        return m, m / np.maximum(self.base[None, None, :], 1e-9)


def category_masks(lab, ids, node, walk_steps_min=400):
    T = len(ids); tok = torch.tensor(ids, device=DEV); nd = torch.tensor(node, device=DEV)
    is_node = nd >= 0
    adj = torch.tensor(lab.A, device=DEV) > 0                                          # [N,N]
    same = (tok[:, None] == tok[None, :])                                                # [T,T] q,k
    nbr = torch.zeros((T, T), dtype=torch.bool, device=DEV)
    nbr[is_node[:, None] & is_node[None, :]] = adj[nd[is_node][:, None], nd[is_node][None, :]].flatten()
    prev_nd = torch.full((T,), -1, device=DEV); prev_nd[1:] = nd[:-1]                   # node at k-1
    prev_node = prev_nd >= 0
    succ_same = torch.zeros((T, T), dtype=torch.bool, device=DEV)
    succ_same[is_node[:, None] & prev_node[None, :]] = (nd[is_node][:, None] == prev_nd[prev_node][None, :]).flatten()
    succ_nbr = torch.zeros((T, T), dtype=torch.bool, device=DEV)
    succ_nbr[is_node[:, None] & prev_node[None, :]] = adj[nd[is_node][:, None], prev_nd[prev_node][None, :]].flatten()
    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=DEV), diagonal=-1)  # k < q
    keynode = is_node[None, :].expand(T, T)
    cats = [same & keynode & causal, nbr & keynode & causal, (~same) & (~nbr) & keynode & causal,
            succ_same & causal, succ_nbr & causal, torch.zeros((T, T), dtype=torch.bool, device=DEV)]
    cats[5][:, 0] = True; cats[5] &= causal
    masks = torch.stack([c.to(torch.bfloat16) for c in cats])
    Qidx = torch.tensor([p for p in range(T) if node[p] >= 0 and p - 1 >= walk_steps_min], device=DEV)
    navail = torch.arange(T, device=DEV).float().clamp(min=1)                            # keys available = q
    base = np.array([float((masks[c][Qidx].float().sum(1) / navail[Qidx]).mean()) for c in range(len(CATS))])
    return masks, Qidx, base


# ----------------------------------------------------------------------------- ablation hooks
def ablation_hooks(lab, heads):
    handles = []
    by_layer = {}
    for l, h in heads: by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        def pre(mod, args, hs=hs):
            x = args[0].clone()
            for h in hs: x[..., h * HD:(h + 1) * HD] = 0
            return (x,)
        handles.append(lab.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
    return handles


# ============================================================================= screen
def stage_screen(lab, h_main, seeds=3, query_set=4):
    nref = norm_ref(lab, h_main)
    F_rand = random_features(lab, nref, seed=321)
    res = {"meta": dict(h_main=h_main, norm_ref=nref, cats=CATS), "conds": {}}
    for s in range(seeds):
        inst = lab.instance(query_set, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        masks, Qidx, base = category_masks(lab, ids, node)
        F_own = own_plane_features(lab, inst, s, h_main)
        for cname, hooks in (("none", []), ("own_hs26_b2", [(delta_window(lab, seg, node, F_own * 2, 0, 0, T_LONG), h_main)]),
                             ("rand_hs16_b4", [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), 16)]),
                             ("rand_hs20_b4", [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), 20)])):
            st = AttnStats(lab, masks, Qidx, base)
            probs, _ = forward_multi(lab, ids, cand, hooks, ())
            st.remove(); mass, enr = st.result()
            r = res["conds"].setdefault(cname, {"mass": [], "enr": [], "base": [], "acc": []})
            r["mass"].append(mass.tolist()); r["enr"].append(enr.tolist()); r["base"].append(base.tolist())
            r["acc"].append(float(lab.seg_masses(probs, seg, node, segs, col)[0][400:].mean()))
        print(f"[screen] seed {s} done; base rates " + " ".join(f"{c}={b:.3f}" for c, b in zip(CATS, base)))
    # summary tables
    E = {c: np.mean(res["conds"][c]["enr"], 0) for c in res["conds"]}       # [L,H,C]
    def top(arr, k=12, label=""):
        idx = np.argsort(arr.ravel())[::-1][:k]
        print(f"[screen] top {label}: " + ", ".join(f"L{i // NH}H{i % NH}={arr.ravel()[i]:.2f}" for i in idx))
        return [(int(i // NH), int(i % NH)) for i in idx]
    ci = {c: i for i, c in enumerate(CATS)}
    top(E["none"][:, :, ci["nbr"]], label="neighbor enrichment (baseline)")
    top(E["none"][:, :, ci["same"]], label="same-node enrichment (baseline)")
    top(E["none"][:, :, ci["succ_same"]], label="successor-of-same enrichment = induction (baseline)")
    d_own = E["own_hs26_b2"][:, :, ci["nbr"]] - E["none"][:, :, ci["nbr"]]
    d_r16 = E["rand_hs16_b4"][:, :, ci["nbr"]] - E["none"][:, :, ci["nbr"]]
    d_r20 = E["rand_hs20_b4"][:, :, ci["nbr"]] - E["none"][:, :, ci["nbr"]]
    readers_own = top(d_own, label="Δ neighbor enrichment under OWN-plane injection (readers)")
    readers_r16 = top(d_r16, label="Δ neighbor enrichment under random plane @hs16")
    readers_r20 = top(d_r20, label="Δ neighbor enrichment under random plane @hs20")
    res["summary"] = dict(readers_own=readers_own, readers_r16=readers_r16, readers_r20=readers_r20,
                          acc={c: float(np.mean(res["conds"][c]["acc"])) for c in res["conds"]})
    print("[screen] acc[400:]: " + " ".join(f"{c}={v:.3f}" for c, v in res["summary"]["acc"].items()))
    json.dump(res, open(f"{OUT}/screen.json", "w"))
    return res


# ============================================================================= ablate
def stage_ablate(lab, h_main, seeds=3, query_set=4):
    sc = json.load(open(f"{OUT}/screen.json")); ci = {c: i for i, c in enumerate(CATS)}
    E = {c: np.mean(sc["conds"][c]["enr"], 0) for c in sc["conds"]}
    d_own = E["own_hs26_b2"][:, :, ci["nbr"]] - E["none"][:, :, ci["nbr"]]
    order = lambda arr: [(int(i // NH), int(i % NH)) for i in np.argsort(arr.ravel())[::-1]]
    G = {"readers_own8": order(d_own)[:8], "nbr_base8": order(E["none"][:, :, ci["nbr"]])[:8],
         "induction8": order(E["none"][:, :, ci["succ_same"]])[:8]}
    used = set(sum(G.values(), [])); rng = np.random.default_rng(0)
    G["random8"] = [t for t in [(int(l), int(h)) for l, h in zip(rng.integers(8, lab.n_layers, 200), rng.integers(0, NH, 200))] if t not in used][:8]
    nref = norm_ref(lab, h_main); F_rand = random_features(lab, nref, seed=321)
    res = {"groups": {k: v for k, v in G.items()}, "runs": []}
    for s in range(seeds):
        inst = lab.instance(query_set, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        F_own = own_plane_features(lab, inst, s, h_main)
        conds = {"none": [], "own_hs26_b2": [(delta_window(lab, seg, node, F_own * 2, 0, 0, T_LONG), h_main)],
                 "rand_hs16_b4": [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), 16)]}
        for gname, heads in [("no_ablation", [])] + list(G.items()) + [(f"single_{l}_{h}", [(l, h)]) for l, h in G["readers_own8"]]:
            hd = ablation_hooks(lab, heads)
            try:
                for cname, hooks in conds.items():
                    probs, _ = forward_multi(lab, ids, cand, hooks, ())
                    m = lab.seg_masses(probs, seg, node, segs, col)[0]
                    res["runs"].append(dict(seed=s, group=gname, cond=cname, acc=float(m[400:].mean()), acc_early=float(m[100:400].mean())))
            finally:
                for h in hd: h.remove()
        g = lambda gn, cn: [r["acc"] for r in res["runs"] if r["seed"] == s and r["group"] == gn and r["cond"] == cn][0]
        print(f"[ablate] seed {s}: " + " | ".join(f"{gn}: none {g(gn, 'none'):.3f} own {g(gn, 'own_hs26_b2'):.3f} rand {g(gn, 'rand_hs16_b4'):.3f}" for gn in ["no_ablation"] + list(G)))
    json.dump(res, open(f"{OUT}/ablate.json", "w"))


# ============================================================================= port construction
def rope_slow_dims(lab, min_wavelength=5000):
    inv = lab.model.model.rotary_emb.inv_freq.detach().float().cpu().numpy()          # [HD/2]
    wl = 2 * np.pi / inv
    slow = [i for i in range(HD // 2) if wl[i] > min_wavelength]
    return slow + [i + HD // 2 for i in slow], int(len(slow))


@torch.no_grad()
def qk_bilinear(lab, l, h, dims):
    attn = lab.model.model.layers[l].self_attn
    ln = (1.0 + lab.model.model.layers[l].input_layernorm.weight.detach().float())  # gemma RMSNorm scale
    Wq = attn.q_proj.weight.detach().float()[h * HD:(h + 1) * HD][dims] * ln[None]  # [nd, d]
    Wk = attn.k_proj.weight.detach().float()[(h // (NH // NKV)) * HD:(h // (NH // NKV) + 1) * HD][dims] * ln[None]
    return (Wq.T @ Wk) * attn.scaling                                               # [d,d]; logits = x_q^T B x_k


@torch.no_grad()
def build_port(lab, heads, k=2, min_wavelength=5000, sign=+1, fast_penalty=1.0):
    """Directions U [k,d] maximizing the shared slow-RoPE QK self-similarity of the reader heads, penalizing
    projection through the fast-RoPE rows (fast_penalty=0 -> plain eigenproblem)."""
    dims, npairs = rope_slow_dims(lab, min_wavelength)
    fast = [i for i in range(HD) if i not in set(dims)]
    S = torch.zeros((lab.d, lab.d), device=DEV); G = torch.zeros((lab.d, lab.d), device=DEV)
    for l, h in heads:
        B = qk_bilinear(lab, l, h, dims); Sh = (B + B.T) / 2
        S += Sh / Sh.norm()
        attn = lab.model.model.layers[l].self_attn; ln = (1.0 + lab.model.model.layers[l].input_layernorm.weight.detach().float())
        Wqf = attn.q_proj.weight.detach().float()[h * HD:(h + 1) * HD][fast] * ln[None]
        Wkf = attn.k_proj.weight.detach().float()[(h // (NH // NKV)) * HD:(h // (NH // NKV) + 1) * HD][fast] * ln[None]
        Gh = Wqf.T @ Wqf + Wkf.T @ Wkf
        G += Gh / Gh.norm()
    if fast_penalty > 0:
        Gn = G * (lab.d / torch.trace(G)) * fast_penalty + torch.eye(lab.d, device=DEV)
        Lc = torch.linalg.cholesky(Gn); Li = torch.linalg.inv(Lc)
        St = Li @ S @ Li.T
        evals, evecs = torch.linalg.eigh(St)
        V = Li.T @ evecs                                       # generalized eigenvectors
    else:
        evals, evecs = torch.linalg.eigh(S); V = evecs
    cols = list(range(lab.d - k, lab.d)) if sign > 0 else list(range(k))
    U = V[:, cols].T
    U = U / U.norm(dim=1, keepdim=True)
    ev = evals[cols].cpu().numpy().tolist()
    return U.cpu().numpy().astype(np.float32), ev, npairs


@torch.no_grad()
def leakage(lab, heads, U, min_wavelength=5000):
    """per reader head: ||Wq_fast u|| / ||Wq_slow u|| and same for Wk (mean over the k directions)."""
    dims, _ = rope_slow_dims(lab, min_wavelength); fast = [i for i in range(HD) if i not in set(dims)]
    Ut = torch.tensor(U, device=DEV); out = {}
    for l, h in heads:
        attn = lab.model.model.layers[l].self_attn; ln = (1.0 + lab.model.model.layers[l].input_layernorm.weight.detach().float())
        Wq = attn.q_proj.weight.detach().float()[h * HD:(h + 1) * HD] * ln[None]; Wk = attn.k_proj.weight.detach().float()[(h // 2) * HD:(h // 2 + 1) * HD] * ln[None]
        rq = ((Wq[fast] @ Ut.T).norm(dim=0) / (Wq[dims] @ Ut.T).norm(dim=0)).mean().item()
        rk = ((Wk[fast] @ Ut.T).norm(dim=0) / (Wk[dims] @ Ut.T).norm(dim=0)).mean().item()
        out[f"L{l}H{h}"] = dict(q=rq, k=rk)
    return out


def port_features(lab, U, norm, perm=None):
    Z = lab.Z2 if perm is None else lab.Z2[perm]
    F = Z @ U[:2]
    return (F * norm / np.linalg.norm(F, axis=1).mean()).astype(np.float32)


@torch.no_grad()
def kernel_report(lab, heads, F, dims, rms2=1.0):
    """implied ADDED attention logit (after RMSNorm scaling) between node j (query) and j' (key) per reader head."""
    Ft = torch.tensor(F, device=DEV)
    out = {}
    for l, h in heads:
        K = (Ft @ qk_bilinear(lab, l, h, dims) @ Ft.T).cpu().numpy() / rms2      # [N,N]
        A = lab.A > 0; eye = np.eye(N, dtype=bool); other = ~A & ~eye
        out[f"L{l}H{h}"] = dict(self=float(K[eye].mean()), nbr=float(K[A].mean()), other=float(K[other].mean()),
                                asym=float(np.linalg.norm(K - K.T) / (np.linalg.norm(K) + 1e-9)))
    return out


# ============================================================================= port
def stage_port(lab, h_main, seeds=6, query_set=4, n_readers=8):
    sc = json.load(open(f"{OUT}/screen.json")); ci = {c: i for i, c in enumerate(CATS)}
    E = {c: np.mean(sc["conds"][c]["enr"], 0) for c in sc["conds"]}
    Mm = {c: np.mean(sc["conds"][c]["mass"], 0) for c in sc["conds"]}; Bb = {c: np.mean(sc["conds"][c]["base"], 0) for c in sc["conds"]}
    pure = lambda c: (Mm[c][:, :, ci["nbr"]] - Mm[c][:, :, ci["succ_same"]]) / max(Bb[c][ci["nbr"]] - Bb[c][ci["succ_same"]], 1e-9)
    base_nbr = pure("none")                      # PURE adjacency: neighbor occurrences that are not induction targets
    d_own = pure("own_hs26_b2") - base_nbr
    d_r = pure("rand_hs16_b4") - base_nbr
    def ranked(arr, lmin, lmax, inj_pure):
        cand = [(float(arr[l, h]), l, h) for l in range(lmin, lmax) for h in range(NH) if inj_pure[l, h] > 1.0]
        return [(l, h) for _, l, h in sorted(cand, reverse=True)]
    late = ranked(d_own, h_main, lab.n_layers, pure("own_hs26_b2")); mid = ranked(d_r, 16, h_main, pure("rand_hs16_b4"))
    readers = []
    for a, b in zip(late[:n_readers // 2], mid[:n_readers // 2]): readers += [a, b]
    readers = readers[:n_readers]
    print(f"[port] late readers (d_own, L>={h_main}): {late[:6]}  mid readers (d_rand16, 16..{h_main-1}): {mid[:6]}")
    used = set(readers); rng = np.random.default_rng(1)
    nonreaders = [t for t in [(int(l), int(h)) for l, h in zip(rng.integers(16, lab.n_layers, 300), rng.integers(0, NH, 300))] if t not in used][:n_readers]
    dims, npairs = rope_slow_dims(lab)
    U, ev, _ = build_port(lab, readers, k=2, fast_penalty=1.0)
    U0, ev0, _ = build_port(lab, readers, k=2, fast_penalty=0.0)
    Ua, eva, _ = build_port(lab, readers, k=2, sign=-1, fast_penalty=1.0)
    Un, evn, _ = build_port(lab, nonreaders, k=2, fast_penalty=1.0)
    hs_inj = max(1, min(l for l, _ in readers))                                       # present at every reader's input
    nref = norm_ref(lab, h_main)
    # residual RMS^2 at the injection layer (for the RMSNorm scaling of the implied logits)
    inst0 = lab.instance(query_set, 0); walk0 = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000))
    ids0, seg0, node0 = lab.build_context([(inst0, walk0)]); cand0, _ = cand_map([inst0])
    _, hs0 = lab.forward(ids0, cand0, hs_layers=(hs_inj,))
    rms2 = float((np.linalg.norm(hs0[hs_inj][node0 >= 0], axis=1) ** 2 / lab.d).mean())
    F_port = port_features(lab, U, nref); F_plain = port_features(lab, U0, nref)
    leak, leak0 = leakage(lab, readers, U), leakage(lab, readers, U0)
    print(f"[port] readers={readers} inject@hs{hs_inj}; slow RoPE pairs={npairs}; eig pen={ev} plain={ev0} anti={eva} nonreader={evn}; rms2@hs{hs_inj}={rms2:.1f}")
    print("[port] fast/slow leakage (penalized): " + " ".join(f"{k}: q{v['q']:.2f} k{v['k']:.2f}" for k, v in leak.items()))
    print("[port] fast/slow leakage (plain):     " + " ".join(f"{k}: q{v['q']:.2f} k{v['k']:.2f}" for k, v in leak0.items()))
    kern = kernel_report(lab, readers, F_port, dims, rms2)
    print("[port] implied ADDED logit per reader head at β=1 (scale by β²): " + " ".join(f"{k}: self {v['self']:.3f} nbr {v['nbr']:.3f} other {v['other']:.3f} asym {v['asym']:.2f}" for k, v in kern.items()))
    res = {"meta": dict(readers=readers, nonreaders=nonreaders, late=late[:12], mid=mid[:12], hs_inj=hs_inj, norm_ref=nref, rms2=rms2,
                        port_eig=ev, plain_eig=ev0, anti_eig=eva, nonreader_eig=evn, leakage=leak, leakage_plain=leak0,
                        kernel=kern, kernel_plain=kernel_report(lab, readers, F_plain, dims, rms2),
                        kernel_anti=kernel_report(lab, readers, port_features(lab, Ua, nref), dims, rms2),
                        U=U.tolist(), U_plain=U0.tolist()),
           "dense": {}, "override": {}, "unseen": [], "attn": {}}
    F_rand = random_features(lab, nref, seed=321)
    perm = np.random.default_rng(99).permutation(N); inv = np.argsort(perm)
    def inj_nbrs(s, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    betas = (1.0, 2.0, 4.0, 8.0)
    for s in range(seeds):
        inst = lab.instance(query_set, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        F_own = own_plane_features(lab, inst, s, h_main)
        conds = {"none": [], "own_hs26_b2": [(delta_window(lab, seg, node, F_own * 2, 0, 0, T_LONG), h_main)],
                 "rand_hs16_b4": [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), 16)],
                 "rand_hs20_b4": [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), 20)]}
        for b in betas:
            conds[f"port_b{b:g}"] = [(delta_window(lab, seg, node, F_port * b, 0, 0, T_LONG), hs_inj)]
            conds[f"port_b{b:g}_hs16"] = [(delta_window(lab, seg, node, F_port * b, 0, 0, T_LONG), 16)]
        for b in (2.0, 4.0):
            conds[f"anti_b{b:g}"] = [(delta_window(lab, seg, node, port_features(lab, Ua, nref) * b, 0, 0, T_LONG), hs_inj)]
            conds[f"nonreader_b{b:g}"] = [(delta_window(lab, seg, node, port_features(lab, Un, nref) * b, 0, 0, T_LONG), hs_inj)]
            conds[f"plain_b{b:g}"] = [(delta_window(lab, seg, node, F_plain * b, 0, 0, T_LONG), hs_inj)]
        masks, Qidx, base = category_masks(lab, ids, node)
        for cname, hooks in conds.items():
            st = AttnStats(lab, masks, Qidx, base)
            probs, _ = forward_multi(lab, ids, cand, hooks, ())
            st.remove(); _, enr = st.result()
            res["dense"].setdefault(cname, []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
            res["attn"].setdefault(cname, []).append({f"L{l}H{h}": dict(nbr=float(enr[l, h, ci["nbr"]]), same=float(enr[l, h, ci["same"]]), succ_same=float(enr[l, h, ci["succ_same"]])) for l, h in readers})
            if cname == "none":
                res["override"].setdefault("none_injmap", []).append(lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist())
        for b in (2.0, 4.0):   # override through the port
            Fp = port_features(lab, U, nref, perm=perm) * b
            probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, Fp, 0, 0, T_LONG), hs_inj)], ())
            res["override"].setdefault(f"port_b{b:g}", []).append(dict(true_map=lab.seg_masses(probs, seg, node, segs, col)[0].tolist(),
                                                                        injected_map=lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist()))
        a = lambda c: np.mean(res["dense"][c][-1][200:600])
        print(f"[port] seed {s}: none {a('none'):.3f} own {a('own_hs26_b2'):.3f} rand16 {a('rand_hs16_b4'):.3f} rand20 {a('rand_hs20_b4'):.3f} | " +
              " ".join(f"port b{b:g}={a(f'port_b{b:g}'):.3f}" for b in betas) + " | " + " ".join(f"port@16 b{b:g}={a(f'port_b{b:g}_hs16'):.3f}" for b in betas) +
              f" | anti b2 {a('anti_b2'):.3f} b4 {a('anti_b4'):.3f} nonreader b2 {a('nonreader_b2'):.3f} b4 {a('nonreader_b4'):.3f} plain b2 {a('plain_b2'):.3f} b4 {a('plain_b4'):.3f}")
        json.dump(res, open(f"{OUT}/port.json", "w"))
    # --- unseen node through the port (and references)
    rows = []
    for s in range(seeds):
        inst = lab.instance(query_set, s); rng = np.random.default_rng(8000 + s); F_own = own_plane_features(lab, inst, s, h_main)
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
            segs = [(inst, walk_q)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            rows.append(dict(seed=s, X=X, Y=Y, cond="none", **masses(forward_multi(lab, ids, cand, [], ())[0])))
            for cname, F, hs, b in (("own", F_own, h_main, 2.0), ("rand16", F_rand, 16, 4.0), ("port_b2", F_port, hs_inj, 2.0), ("port_b4", F_port, hs_inj, 4.0), ("port_b8", F_port, hs_inj, 8.0)):
                full = delta_window(lab, seg, node, F * b, 0, 0, T_UNSEEN + 1)
                Fy = F.copy(); Fy[X] = F[Y]; fully = delta_window(lab, seg, node, Fy * b, 0, 0, T_UNSEEN + 1)
                rows.append(dict(seed=s, X=X, Y=Y, cond=f"{cname}|trueX", **masses(forward_multi(lab, ids, cand, [(full, hs)], ())[0])))
                rows.append(dict(seed=s, X=X, Y=Y, cond=f"{cname}|claimY", **masses(forward_multi(lab, ids, cand, [(fully, hs)], ())[0])))
        print(f"[port-unseen] seed {s}: " + " | ".join(f"{c}: X={np.mean([r['P_nbrs_X'] for r in rows if r['seed']==s and r['cond']==c]):.3f} Y={np.mean([r['P_nbrs_Y'] for r in rows if r['seed']==s and r['cond']==c]):.3f}"
                                                   for c in ("none", "own|trueX", "own|claimY", "port_b4|trueX", "port_b4|claimY", "rand16|trueX")))
    res["unseen"] = rows
    json.dump(res, open(f"{OUT}/port.json", "w"))


# ============================================================================= universal
def stage_universal(lab, h_main, seeds=3, beta=4.0):
    pr = json.load(open(f"{OUT}/port.json")); hs_inj = pr["meta"]["hs_inj"]; nref = pr["meta"]["norm_ref"]
    U = np.array(pr["meta"]["U"], np.float32); F_port = port_features(lab, U, nref)
    F_rand = random_features(lab, nref, seed=321)
    res = {"dense": {}, "unseen": []}
    for set_id in (1, 2, 3):
        for s in range(seeds):
            inst = lab.instance(set_id, 50 + s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(9100 + 10 * set_id + s))
            segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            for cname, hooks in (("none", []), (f"port_b{beta:g}", [(delta_window(lab, seg, node, F_port * beta, 0, 0, T_LONG), hs_inj)]),
                                 ("rand_hs16_b4", [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), 16)])):
                probs, _ = forward_multi(lab, ids, cand, hooks, ())
                res["dense"].setdefault(f"set{set_id}|{cname}", []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
            rng = np.random.default_rng(9200 + 10 * set_id + s)
            for X in rng.choice(N, 2, replace=False):
                X = int(X); Y = far_node(lab, X, rng)
                walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
                segs = [(inst, walk_q)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
                def masses(probs):
                    p = probs[-1]
                    return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
                res["unseen"].append(dict(set=set_id, seed=s, X=X, Y=Y, cond="none", **masses(forward_multi(lab, ids, cand, [], ())[0])))
                full = delta_window(lab, seg, node, F_port * beta, 0, 0, T_UNSEEN + 1); Fy = F_port.copy(); Fy[X] = F_port[Y]
                res["unseen"].append(dict(set=set_id, seed=s, X=X, Y=Y, cond="port|trueX", **masses(forward_multi(lab, ids, cand, [(full, hs_inj)], ())[0])))
                res["unseen"].append(dict(set=set_id, seed=s, X=X, Y=Y, cond="port|claimY", **masses(forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, Fy * beta, 0, 0, T_UNSEEN + 1), hs_inj)], ())[0])))
        a = lambda c: np.mean([np.mean(x[200:600]) for x in res["dense"][f"set{set_id}|{c}"]])
        print(f"[universal] set {set_id}: none {a('none'):.3f} port {a(f'port_b{beta:g}'):.3f} rand16 {a('rand_hs16_b4'):.3f} | unseen: " +
              " ".join(f"{c}: X={np.mean([r['P_nbrs_X'] for r in res['unseen'] if r['set']==set_id and r['cond']==c]):.3f} Y={np.mean([r['P_nbrs_Y'] for r in res['unseen'] if r['set']==set_id and r['cond']==c]):.3f}" for c in ("none", "port|trueX", "port|claimY")))
    json.dump(res, open(f"{OUT}/universal.json", "w"))


# ============================================================================= port2 (round two)
@torch.no_grad()
def build_port_general(lab, heads, min_wavelength=5000, fast_penalty=1.0):
    """generalized eigen-decomposition of the readers' shared symmetric QK form: returns (V [d,d] columns = directions
    with V^T S V = diag(evals)), evals ascending."""
    dims, npairs = rope_slow_dims(lab, min_wavelength); fast = [i for i in range(HD) if i not in set(dims)]
    S = torch.zeros((lab.d, lab.d), device=DEV); G = torch.zeros((lab.d, lab.d), device=DEV)
    for l, h in heads:
        B = qk_bilinear(lab, l, h, dims); Sh = (B + B.T) / 2; S += Sh / Sh.norm()
        attn = lab.model.model.layers[l].self_attn; ln = (1.0 + lab.model.model.layers[l].input_layernorm.weight.detach().float())
        Wqf = attn.q_proj.weight.detach().float()[h * HD:(h + 1) * HD][fast] * ln[None]
        Wkf = attn.k_proj.weight.detach().float()[(h // 2) * HD:(h // 2 + 1) * HD][fast] * ln[None]
        Gh = Wqf.T @ Wqf + Wkf.T @ Wkf; G += Gh / Gh.norm()
    Gn = G * (lab.d / torch.trace(G)) * fast_penalty + torch.eye(lab.d, device=DEV)
    Lc = torch.linalg.cholesky(Gn); Li = torch.linalg.inv(Lc)
    evals, evecs = torch.linalg.eigh(Li @ S @ Li.T)
    V = Li.T @ evecs
    return V, evals


def adjacency_port_features(lab, V, evals, norm):
    """F [16,d] such that F S F^T ~ A (grid adjacency): map A's positive/negative eigen-modes to the readers'
    positive/negative generalized eigen-directions (u normalized so u^T S u = +-1)."""
    mu, z = np.linalg.eigh(lab.A.astype(np.float64))                       # ascending
    pos = [i for i in range(N) if mu[i] > 1e-6]; neg = [i for i in range(N) if mu[i] < -1e-6]
    ev = evals.cpu().numpy(); Vn = V.cpu().numpy()
    top = np.argsort(ev)[::-1][:len(pos)]; bot = np.argsort(ev)[:len(neg)]
    F = np.zeros((N, lab.d))
    for i, c in zip(sorted(pos, key=lambda i: -mu[i]), top):               # largest mu -> largest eigenvalue direction
        F += np.sqrt(mu[i]) * np.outer(z[:, i], Vn[:, c] / np.sqrt(abs(ev[c])))
    for i, c in zip(sorted(neg, key=lambda i: mu[i]), bot):                # most negative mu -> most negative direction
        F += np.sqrt(-mu[i]) * np.outer(z[:, i], Vn[:, c] / np.sqrt(abs(ev[c])))
    F = F - F.mean(0)
    return (F * norm / np.linalg.norm(F, axis=1).mean()).astype(np.float32), dict(n_pos=len(pos), n_neg=len(neg))


def stage_port2(lab, h_main, seeds=6, query_set=4):
    pr = json.load(open(f"{OUT}/port.json")); meta = pr["meta"]
    readers = [tuple(x) for x in meta["readers"]]; late = [tuple(x) for x in meta["late"][:4]]; mid = [tuple(x) for x in meta["mid"][:4]]
    nref = meta["norm_ref"]; dims, _ = rope_slow_dims(lab)
    U_z2 = np.array(meta["U"], np.float32); F_z2 = port_features(lab, U_z2, nref)
    V_all, ev_all = build_port_general(lab, readers); F_adj, info_adj = adjacency_port_features(lab, V_all, ev_all, nref)
    U_late, ev_late, _ = build_port(lab, late, k=2, fast_penalty=1.0); F_late = port_features(lab, U_late, nref)
    V_late, evl = build_port_general(lab, late); F_late_adj, _ = adjacency_port_features(lab, V_late, evl, nref)
    F_rand = random_features(lab, nref, seed=321)
    hs_mid, hs_late = 16, min(l for l, _ in late)
    # rms^2 at hs_late for kernel scaling
    inst0 = lab.instance(query_set, 0); walk0 = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000))
    ids0, seg0, node0 = lab.build_context([(inst0, walk0)]); cand0, _ = cand_map([inst0])
    _, hs0 = lab.forward(ids0, cand0, hs_layers=(hs_late,)); rms2_late = float((np.linalg.norm(hs0[hs_late][node0 >= 0], axis=1) ** 2 / lab.d).mean())
    kern = dict(z2_mid=kernel_report(lab, mid, F_z2, dims, meta["rms2"]), adj_mid=kernel_report(lab, mid, F_adj, dims, meta["rms2"]),
                adj_late=kernel_report(lab, late, F_adj, dims, rms2_late), late_late=kernel_report(lab, late, F_late, dims, rms2_late),
                late_adj_late=kernel_report(lab, late, F_late_adj, dims, rms2_late))
    for k, v in kern.items():
        print(f"[port2] kernel {k} (β=1): " + " ".join(f"{hh}: self {x['self']:.2f} nbr {x['nbr']:.2f} other {x['other']:.2f}" for hh, x in v.items()))
    print(f"[port2] adjacency port uses {info_adj}; late readers {late} inject@hs{hs_late}; rms2_late={rms2_late:.1f}")
    res = {"meta": dict(readers=readers, late=late, mid=mid, hs_mid=hs_mid, hs_late=hs_late, norm_ref=nref, rms2_late=rms2_late, kernels=kern,
                        F_adj_norm=float(np.linalg.norm(F_adj, axis=1).mean())), "dense": {}, "override": {}, "unseen": [], "attn": {}}
    perm = np.random.default_rng(99).permutation(N); inv = np.argsort(perm)
    def inj_nbrs(s, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    def feats_perm(Fbase, kind):
        if kind == "z2": return port_features(lab, U_z2, nref, perm=perm)
        if kind == "late": return port_features(lab, U_late, nref, perm=perm)
        return Fbase[perm]                                                  # adjacency port: permute node rows
    for s in range(seeds):
        inst = lab.instance(query_set, s); walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        conds = {"none": []}
        for b in (3.0, 4.0, 5.0): conds[f"rand20_b{b:g}"] = [(delta_window(lab, seg, node, F_rand * b, 0, 0, T_LONG), 20)]
        for b in (3.0, 4.0, 5.0): conds[f"z2_b{b:g}"] = [(delta_window(lab, seg, node, F_z2 * b, 0, 0, T_LONG), hs_mid)]
        for b in (2.0, 3.0, 4.0, 6.0, 8.0): conds[f"adj_b{b:g}"] = [(delta_window(lab, seg, node, F_adj * b, 0, 0, T_LONG), hs_mid)]
        for b in (2.0, 3.0, 4.0, 6.0): conds[f"late_b{b:g}"] = [(delta_window(lab, seg, node, F_late * b, 0, 0, T_LONG), hs_late)]
        for b in (2.0, 4.0, 6.0): conds[f"lateadj_b{b:g}"] = [(delta_window(lab, seg, node, F_late_adj * b, 0, 0, T_LONG), hs_late)]
        masks, Qidx, base = category_masks(lab, ids, node)
        for cname, hooks in conds.items():
            st = AttnStats(lab, masks, Qidx, base); probs, _ = forward_multi(lab, ids, cand, hooks, ()); st.remove(); _, enr = st.result()
            res["dense"].setdefault(cname, []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
            res["attn"].setdefault(cname, []).append({f"L{l}H{h}": float(enr[l, h, CATS.index("nbr")]) for l, h in readers})
            if cname == "none": res["override"].setdefault("none_injmap", []).append(lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist())
        # reader-ablation specificity (readers = mid + late)
        hd = ablation_hooks(lab, readers)
        try:
            for cname, hooks in (("abl|none", []), ("abl|z2_b4", conds["z2_b4"]), ("abl|rand20_b4", conds["rand20_b4"]), ("abl|adj_b4", conds["adj_b4"]), ("abl|late_b4", conds["late_b4"])):
                probs, _ = forward_multi(lab, ids, cand, hooks, ()); res["dense"].setdefault(cname, []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
        finally:
            for h in hd: h.remove()
        # override through adj / late ports
        for cname, Fb, kind, hs, b in (("adj_b4", F_adj, "adj", hs_mid, 4.0), ("late_b4", F_late, "late", hs_late, 4.0), ("z2_b4", F_z2, "z2", hs_mid, 4.0)):
            Fp = feats_perm(Fb, kind) * b
            probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, Fp, 0, 0, T_LONG), hs)], ())
            res["override"].setdefault(cname, []).append(dict(true_map=lab.seg_masses(probs, seg, node, segs, col)[0].tolist(), injected_map=lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist()))
        a = lambda c: np.mean(res["dense"][c][-1][200:600])
        print(f"[port2] seed {s}: none {a('none'):.3f} | rand20 " + " ".join(f"b{b:g}={a(f'rand20_b{b:g}'):.3f}" for b in (3, 4, 5)) + " | z2 " + " ".join(f"b{b:g}={a(f'z2_b{b:g}'):.3f}" for b in (3, 4, 5)) +
              " | adj " + " ".join(f"b{b:g}={a(f'adj_b{b:g}'):.3f}" for b in (2, 3, 4, 6, 8)) + " | late " + " ".join(f"b{b:g}={a(f'late_b{b:g}'):.3f}" for b in (2, 3, 4, 6)) +
              " | lateadj " + " ".join(f"b{b:g}={a(f'lateadj_b{b:g}'):.3f}" for b in (2, 4, 6)) + f" | ablated: none {a('abl|none'):.3f} z2 {a('abl|z2_b4'):.3f} rand20 {a('abl|rand20_b4'):.3f} adj {a('abl|adj_b4'):.3f} late {a('abl|late_b4'):.3f}")
        json.dump(res, open(f"{OUT}/port2.json", "w"))
    rows = []
    for s in range(seeds):
        inst = lab.instance(query_set, s); rng = np.random.default_rng(8000 + s)
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
            segs = [(inst, walk_q)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            rows.append(dict(seed=s, X=X, Y=Y, cond="none", **masses(forward_multi(lab, ids, cand, [], ())[0])))
            for cname, F, hs, b in (("rand20_b4", F_rand, 20, 4.0), ("z2_b4", F_z2, hs_mid, 4.0), ("adj_b3", F_adj, hs_mid, 3.0), ("adj_b4", F_adj, hs_mid, 4.0), ("adj_b6", F_adj, hs_mid, 6.0),
                                    ("late_b3", F_late, hs_late, 3.0), ("late_b4", F_late, hs_late, 4.0), ("lateadj_b4", F_late_adj, hs_late, 4.0)):
                full = delta_window(lab, seg, node, F * b, 0, 0, T_UNSEEN + 1); Fy = F.copy(); Fy[X] = F[Y]
                rows.append(dict(seed=s, X=X, Y=Y, cond=f"{cname}|trueX", **masses(forward_multi(lab, ids, cand, [(full, hs)], ())[0])))
                rows.append(dict(seed=s, X=X, Y=Y, cond=f"{cname}|claimY", **masses(forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, Fy * b, 0, 0, T_UNSEEN + 1), hs)], ())[0])))
        print(f"[port2-unseen] seed {s}: " + " | ".join(f"{c}: X={np.mean([r['P_nbrs_X'] for r in rows if r['seed']==s and r['cond']==c]):.3f} Y={np.mean([r['P_nbrs_Y'] for r in rows if r['seed']==s and r['cond']==c]):.3f}"
                                                   for c in ("none", "adj_b4|trueX", "adj_b4|claimY", "late_b4|trueX", "late_b4|claimY")))
    res["unseen"] = rows
    json.dump(res, open(f"{OUT}/port2.json", "w"))


# ============================================================================= port3 (late port: universality + specificity controls at hs26)
def stage_port3(lab, h_main, seeds_main=6, seeds_new=3):
    pr = json.load(open(f"{OUT}/port.json")); meta = pr["meta"]; nref = meta["norm_ref"]
    late = [tuple(x) for x in meta["late"][:4]]; hs_late = min(l for l, _ in late)
    U_late, _, _ = build_port(lab, late, k=2, fast_penalty=1.0); F_late = port_features(lab, U_late, nref)
    V_late, evl = build_port_general(lab, late); F_lateadj, _ = adjacency_port_features(lab, V_late, evl, nref)
    U_anti, _, _ = build_port(lab, late, k=2, sign=-1, fast_penalty=1.0); F_anti = port_features(lab, U_anti, nref)
    rng = np.random.default_rng(3); used = set(late)
    nonlate = [t for t in [(int(l), int(h)) for l, h in zip(rng.integers(h_main, lab.n_layers, 300), rng.integers(0, NH, 300))] if t not in used][:4]
    U_non, _, _ = build_port(lab, nonlate, k=2, fast_penalty=1.0); F_non = port_features(lab, U_non, nref)
    F_rand = random_features(lab, nref, seed=321)
    perm = np.random.default_rng(99).permutation(N); inv = np.argsort(perm)
    def inj_nbrs(s, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    res = {"meta": dict(late=late, nonlate=nonlate, hs_late=hs_late, U_late=U_late.tolist()), "dense": {}, "override": {}, "unseen": [], "attn": {}}
    contexts = [(4, s, np.random.default_rng(5000 + s)) for s in range(seeds_main)] + [(k, 50 + s, np.random.default_rng(9100 + 10 * k + s)) for k in (1, 2, 3) for s in range(seeds_new)]
    for set_id, s, rngw in contexts:
        inst = lab.instance(set_id, s); walk = random_walk(lab.nbrs, T_LONG, rngw)
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        F_own = own_plane_features(lab, inst, s, h_main)
        conds = {"none": [], "own_b2": [(delta_window(lab, seg, node, F_own * 2, 0, 0, T_LONG), h_main)],
                 "rand26_b4": [(delta_window(lab, seg, node, F_rand * 4, 0, 0, T_LONG), hs_late)],
                 "late_b3": [(delta_window(lab, seg, node, F_late * 3, 0, 0, T_LONG), hs_late)], "late_b4": [(delta_window(lab, seg, node, F_late * 4, 0, 0, T_LONG), hs_late)],
                 "lateadj_b4": [(delta_window(lab, seg, node, F_lateadj * 4, 0, 0, T_LONG), hs_late)],
                 "lateanti_b4": [(delta_window(lab, seg, node, F_anti * 4, 0, 0, T_LONG), hs_late)], "latenon_b4": [(delta_window(lab, seg, node, F_non * 4, 0, 0, T_LONG), hs_late)]}
        masks, Qidx, base = category_masks(lab, ids, node)
        for cname, hooks in conds.items():
            st = AttnStats(lab, masks, Qidx, base); probs, _ = forward_multi(lab, ids, cand, hooks, ()); st.remove(); _, enr = st.result()
            res["dense"].setdefault(f"set{set_id}|{cname}", []).append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
            res["attn"].setdefault(f"set{set_id}|{cname}", []).append({f"L{l}H{h}": float(enr[l, h, CATS.index("nbr")]) for l, h in late})
            if cname == "none": res["override"].setdefault(f"set{set_id}|none_injmap", []).append(lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist())
        for cname, Fp in (("late_b4", port_features(lab, U_late, nref, perm=perm) * 4), ("lateadj_b4", F_lateadj[perm] * 4), ("lateanti_b4", port_features(lab, U_anti, nref, perm=perm) * 4),
                          ("latenon_b4", port_features(lab, U_non, nref, perm=perm) * 4), ("rand26_b4", F_rand[perm] * 4), ("own_b2", F_own[perm] * 2)):
            hs = h_main if cname == "own_b2" else hs_late
            probs, _ = forward_multi(lab, ids, cand, [(delta_window(lab, seg, node, Fp, 0, 0, T_LONG), hs)], ())
            res["override"].setdefault(f"set{set_id}|{cname}", []).append(dict(true_map=lab.seg_masses(probs, seg, node, segs, col)[0].tolist(), injected_map=lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist()))
        rngu = np.random.default_rng(8000 + s + 100 * set_id)
        for X in rngu.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rngu)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rngu, avoid=X), [X]])
            segs_u = [(inst, walk_q)]; ids_u, seg_u, node_u = lab.build_context(segs_u); cand_u, col_u = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col_u[int(inst["ids"][q])]] for q in lab.nbrs[X])), P_nbrs_Y=float(sum(p[col_u[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            res["unseen"].append(dict(set=set_id, seed=s, X=X, Y=Y, cond="none", **masses(forward_multi(lab, ids_u, cand_u, [], ())[0])))
            for cname, F, hs, b in (("own_b2", F_own, h_main, 2.0), ("rand26_b4", F_rand, hs_late, 4.0), ("late_b3", F_late, hs_late, 3.0), ("late_b4", F_late, hs_late, 4.0),
                                    ("lateadj_b4", F_lateadj, hs_late, 4.0), ("lateanti_b4", F_anti, hs_late, 4.0), ("latenon_b4", F_non, hs_late, 4.0)):
                full = delta_window(lab, seg_u, node_u, F * b, 0, 0, T_UNSEEN + 1); Fy = F.copy(); Fy[X] = F[Y]
                res["unseen"].append(dict(set=set_id, seed=s, X=X, Y=Y, cond=f"{cname}|trueX", **masses(forward_multi(lab, ids_u, cand_u, [(full, hs)], ())[0])))
                res["unseen"].append(dict(set=set_id, seed=s, X=X, Y=Y, cond=f"{cname}|claimY", **masses(forward_multi(lab, ids_u, cand_u, [(delta_window(lab, seg_u, node_u, Fy * b, 0, 0, T_UNSEEN + 1), hs)], ())[0])))
        a = lambda c: np.mean(res["dense"][f"set{set_id}|{c}"][-1][100:300])
        u = lambda c, k: np.mean([r[k] for r in res["unseen"] if r["set"] == set_id and r["seed"] == s and r["cond"] == c])
        print(f"[port3] set{set_id} seed{s}: uptake[100-300] none {a('none'):.3f} own {a('own_b2'):.3f} rand26 {a('rand26_b4'):.3f} late3 {a('late_b3'):.3f} late4 {a('late_b4'):.3f} lateadj {a('lateadj_b4'):.3f} anti {a('lateanti_b4'):.3f} non {a('latenon_b4'):.3f}"
              f" | unseen X: none {u('none','P_nbrs_X'):.3f} late4 {u('late_b4|trueX','P_nbrs_X'):.3f} claimY->Y {u('late_b4|claimY','P_nbrs_Y'):.3f} anti {u('lateanti_b4|trueX','P_nbrs_X'):.3f} non {u('latenon_b4|trueX','P_nbrs_X'):.3f} rand26 {u('rand26_b4|trueX','P_nbrs_X'):.3f}")
        json.dump(res, open(f"{OUT}/port3.json", "w"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["all", "screen", "ablate", "port", "universal", "port2", "port3"])
    ap.add_argument("--seeds", type=int, default=None); ap.add_argument("--h_main", type=int, default=None)
    args = ap.parse_args(); t0 = time.time()
    lab = Lab(); h_main = args.h_main or json.load(open(f"{HERE}/results_gt/e1.json"))["h_energy"]
    print(f"[main] h_main={h_main}")
    if args.stage in ("all", "screen"): stage_screen(lab, h_main, seeds=args.seeds or 3)
    if args.stage in ("all", "ablate"): stage_ablate(lab, h_main, seeds=args.seeds or 3)
    if args.stage in ("all", "port"): stage_port(lab, h_main, seeds=args.seeds or 6)
    if args.stage in ("all", "universal"): stage_universal(lab, h_main, seeds=args.seeds or 3)
    if args.stage == "port2": stage_port2(lab, h_main, seeds=args.seeds or 6)
    if args.stage == "port3": stage_port3(lab, h_main)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")
