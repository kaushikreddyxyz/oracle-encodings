"""Ground-truth (spectral-embedding) features injected into the model's NATIVE position subspace.

Hypothesis: a model that has an in-context position-reading circuit (Park et al. 2501.00070)
will use exogenously written ground-truth coordinates as a crutch -> faster in-context learning
and causal control over next-token predictions.

E1  native-subspace geometry: fit spectral coords -> residual-stream node means (linear map M)
    on several disjoint word sets / assignments; cross-transfer R^2, principal angles, per layer.
E2  single grid, new word set: inject GT coords via M from a DIFFERENT word set at every node token.
    dense uptake curves (vs none / extracted / random-subspace), override (permuted GT, scored under
    both maps), unseen-node causal test, layer & strength sweeps.
E3  multi-grid demos: [grid1][sep][grid2][sep][grid3][sep][query grid], all injected. Per-segment
    uptake curves (does each additional grid speed uptake?), factorial controls, random-subspace arm,
    unseen-node on the query grid with/without demos.

Runs on CUDA. Saves to results_gt/. Plotting is done locally (plot_gt.py).
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = os.environ.get("ICL_MODEL", "google/gemma-2-9b")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("ICL_OUT", os.path.join(HERE, "results_gt"))
os.makedirs(OUT, exist_ok=True)
SIDE, N = 4, 16
T_LONG = 1200      # long walk (E1, E2 dense)
T_DEMO = 500       # demo grid walk (E3)
T_QUERY = 600      # query grid walk (E3)
T_UNSEEN = 600     # context walk before the held-out query node
WIN = 200          # window (tokens) for node means

POOL = ("apple bird car water sand math fire moon tree salt gold wind rain star book door milk iron "
        "song road fish snow king hand rock lamp ship cake horse glass river bread stone cloud paper "
        "chair table house green black white light night sun hill lake wolf bear lion rose wine corn "
        "leaf ring bell coat shoe hat bag box cup key pen map flag ball drum coin desk wall roof gate "
        "bridge tower field farm boat train plane truck bus bike wheel engine motor queen judge nurse "
        "doctor pilot chef bank bone blood brain heart mouth tooth nose skin hair foot knee arm leg "
        "neck face eye ear lip cat dog cow pig goat duck frog ant bee owl fox deer crab seed root "
        "branch grass moss oak pine palm fern silk wool cotton leather steel copper silver tin lead "
        "brass clay chalk coal oil gas ice steam smoke dust mud ink soap candle mirror clock watch "
        "phone radio piano guitar violin flute knife fork spoon plate bowl pot pan oven sofa bed "
        "pillow blanket towel window floor stairs porch garden yard park street city town village "
        "island desert forest jungle ocean beach coast valley canyon cave planet comet rocket robot "
        "magnet laser crystal diamond pearl ruby jade amber").split()


# ----------------------------------------------------------------------------- graph + features
def build_grid(n_side=SIDE):
    coords = [(r, c) for r in range(n_side) for c in range(n_side)]
    n = len(coords)
    A = np.zeros((n, n), np.float32)
    for i, (r, c) in enumerate(coords):
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if (r + dr, c + dc) in coords:
                A[i, coords.index((r + dr, c + dc))] = 1
    nbrs = [np.nonzero(A[i])[0].tolist() for i in range(n)]
    return A, nbrs, coords


def spectral_coords(coords, k=3):
    """Laplacian eigenvectors of the 4x4 grid = P4 x P4 products. v(i)=cos(pi(i+.5)/4).
    Z2 = (v(row), v(col)) [eigenvalue lambda1, degenerate pair]; Z3 adds v(row)v(col) [2*lambda1]."""
    v = np.cos(np.pi * (np.arange(SIDE) + 0.5) / SIDE)
    Z = np.array([[v[r], v[c], v[r] * v[c]] for r, c in coords], np.float64)
    Z -= Z.mean(0)
    Z /= np.linalg.norm(Z, axis=0, keepdims=True)  # unit columns (scale absorbed by M)
    return Z[:, :k]


def random_walk(nbrs, T, rng, avoid=None):
    allowed = [i for i in range(len(nbrs)) if i != avoid]
    cur = int(rng.choice(allowed))
    walk = [cur]
    for _ in range(T - 1):
        opts = [j for j in nbrs[cur] if j != avoid]
        cur = int(rng.choice(opts))
        walk.append(cur)
    return np.array(walk)


def fit_M(Hc, Z):
    """least squares Z (16xk) -> Hc (16xd). returns M (kxd), in-sample R^2."""
    M, *_ = np.linalg.lstsq(Z, Hc, rcond=None)
    r2 = 1 - np.sum((Hc - Z @ M) ** 2) / np.sum(Hc ** 2)
    return M, float(r2)


def r2_fixed(M, Hc, Z):
    return float(1 - np.sum((Hc - Z @ M) ** 2) / np.sum(Hc ** 2))


def top_pcs(Hc, k):
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    return Vt[:k]


def r2_in_pcs(M, Hc, Z, k=3):
    """R^2 after projecting both Hc and the prediction onto Hc's top-k PCs."""
    P = top_pcs(Hc, k)
    Hk = Hc @ P.T
    return float(1 - np.sum((Hk - (Z @ M) @ P.T) ** 2) / np.sum(Hk ** 2))


def principal_angles(A, B):
    """A (kxd), B (mxd): cosines of principal angles between row spaces."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0, 1).tolist()


# ----------------------------------------------------------------------------- model wrapper
class Lab:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
        self.d = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        words, ids = [], []
        for w in POOL:
            t = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(t) == 1 and t[0] not in ids:
                words.append(w); ids.append(t[0])
        rng = np.random.default_rng(2024)
        perm = rng.permutation(len(words))
        n_sets = len(words) // N
        self.sets = [[(words[i], ids[i]) for i in perm[s * N:(s + 1) * N]] for s in range(n_sets)]
        assert n_sets >= 5, f"need >=5 disjoint word sets, got {n_sets}"
        self.sep_id = self.tok("\n", add_special_tokens=False)["input_ids"][-1]
        self.bos_id = self.tok.bos_token_id
        self.A, self.nbrs, self.coords = build_grid()
        self.Z3 = spectral_coords(self.coords, 3)
        self.Z2 = self.Z3[:, :2]
        print(f"[lab] {MODEL}: d={self.d} layers={self.n_layers} single-token words={len(words)} "
              f"-> {n_sets} sets of {N}; sep_id={self.sep_id}")
        self.hooks = []

    # -- grid instance: word set + assignment (position j -> token ids[j])
    def instance(self, set_id, assign_seed):
        rng = np.random.default_rng(10_000 * set_id + assign_seed + 17)
        perm = rng.permutation(N)
        ws = self.sets[set_id]
        return dict(set_id=set_id, assign_seed=assign_seed,
                    ids=np.array([ws[perm[j]][1] for j in range(N)]),
                    words=[ws[perm[j]][0] for j in range(N)])

    # -- context assembly
    def build_context(self, segments):
        ids, seg, node = [self.bos_id], [-1], [-1]
        for s, (inst, walk) in enumerate(segments):
            if s > 0:
                ids.append(self.sep_id); seg.append(-1); node.append(-1)
            for j in walk:
                ids.append(int(inst["ids"][j])); seg.append(s); node.append(int(j))
        return np.array(ids), np.array(seg), np.array(node)

    def delta_from_features(self, seg, node, feats):
        """feats: list per segment of (16xd float32) or None -> [T,d] injection tensor (bf16)."""
        delta = np.zeros((len(seg), self.d), np.float32)
        for p in range(len(seg)):
            if seg[p] >= 0 and feats[seg[p]] is not None:
                delta[p] = feats[seg[p]][node[p]]
        return torch.tensor(delta, dtype=torch.bfloat16, device=DEV)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.inference_mode()
    def forward(self, ids, cand_ids, hs_layers=(), delta=None, inject_hs=None):
        """returns probs over cand_ids [T, n_cand] and {h: hidden [T,d]} for h in hs_layers.
        delta [T,d] is added to hidden-state index inject_hs (hook on decoder layer inject_hs-1)."""
        if delta is not None:
            assert inject_hs is not None and 1 <= inject_hs < self.n_layers
            def fn(mod, inp, out):
                if isinstance(out, tuple):
                    out[0].add_(delta[None])
                    return out
                out.add_(delta[None]); return out
            self.hooks.append(self.model.model.layers[inject_hs - 1].register_forward_hook(fn))
        try:
            out = self.model(torch.tensor(ids, device=DEV)[None], output_hidden_states=bool(hs_layers))
        finally:
            self.clear_hooks()
        probs = torch.softmax(out.logits[0].float(), -1)[:, torch.tensor(cand_ids, device=DEV)]
        hs = {h: out.hidden_states[h][0].float().cpu().numpy() for h in hs_layers}
        return probs.cpu().numpy(), hs

    # -- metrics
    def seg_masses(self, probs, seg, node, segments, cand_col, nbr_fn=None):
        """per-segment arrays of rule-following mass at each position with an in-segment successor.
        nbr_fn(seg_index, j) -> list of node positions counted as valid neighbors (default: true)."""
        res = []
        for s, (inst, walk) in enumerate(segments):
            pos = np.nonzero(seg == s)[0]
            m = []
            for p in pos[:-1]:
                j = node[p]
                nb = self.nbrs[j] if nbr_fn is None else nbr_fn(s, j)
                m.append(sum(probs[p, cand_col[int(inst["ids"][q])]] for q in nb))
            res.append(np.array(m))
        return res

    def node_means(self, hs, seg, node, s, win=WIN):
        """mean hidden per node over the last `win` positions of segment s. [16,d] (NaN if unseen)."""
        pos = np.nonzero(seg == s)[0][-win:]
        H = np.full((N, hs.shape[1]), np.nan, np.float32)
        for j in range(N):
            pj = pos[node[pos] == j]
            if len(pj):
                H[j] = hs[pj].mean(0)
        return H

    def energy(self, H):
        ok = ~np.isnan(H[:, 0]); Hc = H[ok] - H[ok].mean(0)
        E = sum(np.sum((H[i] - H[j]) ** 2) for i in range(N) for j in range(N)
                if self.A[i, j] and ok[i] and ok[j])
        return float(E / (2 * np.sum(Hc ** 2) + 1e-9))


def cand_map(insts):
    ids = sorted({int(t) for inst in insts for t in inst["ids"]})
    return ids, {t: i for i, t in enumerate(ids)}


def nanfill(H):
    return np.where(np.isnan(H), np.nanmean(H, 0, keepdims=True), H)


# ============================================================================= E1
def stage_e1(lab):
    runs = [(s, a, w) for s in (0, 1, 2) for a in (0, 1) for w in (0, 1)] + [(3, 0, 0), (4, 0, 0)]
    layers = list(range(0, lab.n_layers + 1))
    means, energies, meta = [], [], []
    for (s, a, w) in runs:
        inst = lab.instance(s, a)
        walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(100 * s + 10 * a + w))
        ids, seg, node = lab.build_context([(inst, walk)])
        cand, col = cand_map([inst])
        probs, hs = lab.forward(ids, cand, hs_layers=layers)
        mass = lab.seg_masses(probs, seg, node, [(inst, walk)], col)[0]
        H = np.stack([lab.node_means(hs[h], seg, node, 0) for h in layers])  # [L,16,d]
        means.append(H.astype(np.float32))
        energies.append([lab.energy(H[h]) for h in layers])
        meta.append(dict(set_id=s, assign_seed=a, walk_seed=w, final_mass=float(mass[-200:].mean()),
                         words=inst["words"]))
        print(f"[E1] set{s} assign{a} walk{w}: final-200 mass={mass[-200:].mean():.3f}")
    means = np.stack(means)                      # [R,L,16,d]
    energies = np.array(energies)                # [R,L]
    np.savez_compressed(f"{OUT}/e1_means.npz", means=means, energies=energies)
    # ---- analysis per layer
    R = len(runs)
    analysis = {}
    e_mean = energies.mean(0)
    for h in layers:
        Hc = [nanfill(means[r, h].astype(np.float64)) for r in range(R)]
        Hc = [x - x.mean(0) for x in Hc]
        Ms, r2_in, r2_in_pc = [], [], []
        for r in range(R):
            M, r2 = fit_M(Hc[r], lab.Z2)
            Ms.append(M); r2_in.append(r2); r2_in_pc.append(r2_in_pcs(M, Hc[r], lab.Z2))
        cross = np.zeros((R, R)); cross_pc = np.zeros((R, R)); ang = np.zeros((R, R)); pcang = np.zeros((R, R))
        PCs = [top_pcs(Hc[r], 2) for r in range(R)]
        for i in range(R):
            for j in range(R):
                cross[i, j] = r2_fixed(Ms[i], Hc[j], lab.Z2)
                cross_pc[i, j] = r2_in_pcs(Ms[i], Hc[j], lab.Z2)
                ang[i, j] = np.mean(principal_angles(Ms[i], Ms[j]))
                pcang[i, j] = np.mean(principal_angles(PCs[i], PCs[j]))
        analysis[h] = dict(r2_in=r2_in, r2_in_pc=r2_in_pc, cross=cross.tolist(),
                           cross_pc=cross_pc.tolist(), angles=ang.tolist(), pc_angles=pcang.tolist(), energy=float(e_mean[h]))
    # summary: off-diagonal cross R^2 by relation type
    def rel(i, j):
        a, b = runs[i], runs[j]
        if i == j: return "self"
        if a[0] == b[0] and a[1] == b[1]: return "same_set_same_assign"
        if a[0] == b[0]: return "same_set_diff_assign"
        return "diff_set"
    summary = {}
    for h in layers:
        c = np.array(analysis[h]["cross"]); cpc = np.array(analysis[h]["cross_pc"]); an = np.array(analysis[h]["angles"]); pan = np.array(analysis[h]["pc_angles"])
        d = {}
        for k in ("self", "same_set_same_assign", "same_set_diff_assign", "diff_set"):
            idx = [(i, j) for i in range(R) for j in range(R) if rel(i, j) == k]
            d[k] = dict(cross=float(np.mean([c[i, j] for i, j in idx])),
                        cross_pc=float(np.mean([cpc[i, j] for i, j in idx])),
                        angle_cos=float(np.mean([an[i, j] for i, j in idx])),
                        pc_angle_cos=float(np.mean([pan[i, j] for i, j in idx])))
        summary[h] = dict(**d, r2_in=float(np.mean(analysis[h]["r2_in"])),
                          r2_in_pc=float(np.mean(analysis[h]["r2_in_pc"])), energy=float(e_mean[h]))
    h_energy = int(np.argmin(e_mean[6:-1]) + 6)
    h_xfer = int(max(range(8, lab.n_layers), key=lambda h: summary[h]["diff_set"]["cross_pc"]))  # hs[n_layers] is post-final-norm
    json.dump(dict(runs=meta, summary=summary, h_energy=h_energy, h_xfer=h_xfer, n_layers=lab.n_layers),
              open(f"{OUT}/e1.json", "w"), indent=1)
    print(f"[E1] h_energy={h_energy}  h_xfer(best diff-set cross R^2 in PCs)={h_xfer}")
    for h in sorted(h for h in {h_energy, h_xfer, 8, 16, 24, 32, lab.n_layers} if h <= lab.n_layers):
        s = summary[h]
        print(f"[E1]   hs{h:2d}: E={s['energy']:.3f} r2_in={s['r2_in']:.3f} r2_in_pc={s['r2_in_pc']:.3f} "
              f"| diff_set cross={s['diff_set']['cross']:.3f} cross_pc={s['diff_set']['cross_pc']:.3f} "
              f"angcos={s['diff_set']['angle_cos']:.3f} pcangcos={s['diff_set']['pc_angle_cos']:.3f} | same_set_diff_assign cross_pc="
              f"{s['same_set_diff_assign']['cross_pc']:.3f}")
    return h_energy, h_xfer


# ============================================================================= calibration helpers
def calibrate_M(lab, h, cal_sets=(0,), k=2):
    """pooled least-squares fit of Z_k -> centered node means at hs index h, over E1 runs of cal_sets."""
    e1 = json.load(open(f"{OUT}/e1.json")); d = np.load(f"{OUT}/e1_means.npz")
    Zs, Hs = [], []
    Z = lab.Z3[:, :k]
    for r, m in enumerate(e1["runs"]):
        if m["set_id"] in cal_sets:
            Hc = nanfill(d["means"][r, h].astype(np.float64)); Hc -= Hc.mean(0)
            Zs.append(Z); Hs.append(Hc)
    M, r2 = fit_M(np.concatenate(Hs), np.concatenate(Zs))
    return M.astype(np.float32), r2


def gt_features(lab, M, k=2, perm=None):
    """[16,d]: node j gets GT coords of position perm[j] (identity if None) mapped by M."""
    Z = lab.Z3[:, :k]
    if perm is not None:
        Z = Z[perm]
    return (Z @ M.astype(np.float64)).astype(np.float32)


def random_features(lab, target_norm, k=2, seed=123):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((lab.d, k)))
    F = lab.Z3[:, :k] @ Q.T
    return (F * target_norm / np.linalg.norm(F, axis=1).mean()).astype(np.float32)


def extracted_features(lab, inst, h, seed, k=3):
    """green reference: top-k-PC projection of centered node means from a separate walk on inst."""
    walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + seed))
    ids, seg, node = lab.build_context([(inst, walk)])
    cand, col = cand_map([inst])
    _, hs = lab.forward(ids, cand, hs_layers=(h,))
    Hc = nanfill(lab.node_means(hs[h], seg, node, 0).astype(np.float64)); Hc -= Hc.mean(0)
    P = top_pcs(Hc, k)
    return ((Hc @ P.T) @ P).astype(np.float32)


def far_node(lab, i, rng):
    dist = np.abs(np.array(lab.coords) - np.array(lab.coords[i])).sum(1)
    cands = np.nonzero(dist >= 3)[0]
    return int(rng.choice(cands))


# ============================================================================= E2
def stage_e2(lab, h_main, layers, seeds=6, query_set=4, cal_sets=(0,), cal_sets_b=(1,)):
    M, r2 = calibrate_M(lab, h_main, cal_sets)
    Mb, r2b = calibrate_M(lab, h_main, cal_sets_b)
    M3, r2_3 = calibrate_M(lab, h_main, cal_sets, k=3)
    F_gt = gt_features(lab, M); norm_gt = float(np.linalg.norm(F_gt, axis=1).mean())
    print(f"[E2] M(h{h_main}) from sets {cal_sets}: R2={r2:.3f}; from {cal_sets_b}: R2={r2b:.3f}; "
          f"GT3 R2={r2_3:.3f}; mean |F_gt|={norm_gt:.2f}")
    Ms_layer = {L: calibrate_M(lab, L, cal_sets)[0] for L in layers}
    results = {"meta": dict(h_main=h_main, layers=layers, r2=r2, r2b=r2b, r2_3=r2_3, norm_gt=norm_gt),
               "dense": {}, "override": {}, "unseen": []}

    def run_dense(name, feat_fn, inject_hs, nbr_fn=None):
        curves, curves_alt = [], []
        for s in range(seeds):
            inst = lab.instance(query_set, s)
            walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
            segs = [(inst, walk)]
            ids, seg, node = lab.build_context(segs)
            cand, col = cand_map([inst])
            F = feat_fn(inst, s)
            delta = lab.delta_from_features(seg, node, [F]) if F is not None else None
            probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=inject_hs)
            curves.append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
            if nbr_fn is not None:
                curves_alt.append(lab.seg_masses(probs, seg, node, segs, col, nbr_fn)[0].tolist())
        m = np.array(curves).mean(0)
        print(f"[E2] {name}: mass@[50,100,200,400,800,1200]=" +
              " ".join(f"{m[min(t, len(m))-1]:.2f}" for t in (50, 100, 200, 400, 800, 1200)))
        return curves, curves_alt

    # --- override map (defined up front so the baseline can be scored under it too)
    perm = np.random.default_rng(99).permutation(N)
    inv = np.argsort(perm)
    def inj_nbrs(s, j):   # tokens whose injected position is adjacent to j's injected position
        return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    # --- baseline & references
    c_none, c_none_inj = run_dense("none", lambda inst, s: None, h_main, nbr_fn=inj_nbrs)
    results["dense"]["none"] = c_none
    results["override"]["none_injmap"] = c_none_inj
    for beta in (1.0, 2.0, 3.0):
        results["dense"][f"native_gt2_b{beta}"] = run_dense(
            f"native_gt2_b{beta}", lambda inst, s, b=beta: F_gt * b, h_main)[0]
    results["dense"]["native_gt3_b2.0"] = run_dense(
        "native_gt3_b2.0", lambda inst, s: gt_features(lab, M3, k=3) * 2, h_main)[0]
    results["dense"]["native_gt2_calB_b2.0"] = run_dense(
        "native_gt2_calB_b2.0", lambda inst, s: gt_features(lab, Mb) * 2, h_main)[0]
    for rs in (123, 124, 125):
        results["dense"][f"random_gt2_b2.0_plane{rs}"] = run_dense(
            f"random_gt2_b2.0_plane{rs}", lambda inst, s, rs=rs: random_features(lab, norm_gt, seed=rs) * 2, h_main)[0]
    results["dense"]["extracted_b2.0"] = run_dense(
        "extracted_b2.0", lambda inst, s: extracted_features(lab, inst, h_main, s) * 2, h_main)[0]
    # --- layer sweep (native GT2, beta 2), M refit at each layer
    for L in layers:
        if L == h_main:
            continue
        results["dense"][f"native_gt2_b2.0_hs{L}"] = run_dense(
            f"native_gt2_b2.0_hs{L}", lambda inst, s, L=L: gt_features(lab, Ms_layer[L]) * 2, L)[0]
    # --- override: permuted GT (token j carries coords of position perm[j]); score both maps
    for beta in (2.0, 3.0):
        c_true, c_inj = run_dense(f"override_b{beta}", lambda inst, s, b=beta: gt_features(lab, M, perm=perm) * b,
                                  h_main, nbr_fn=inj_nbrs)
        shared = int(sum(1 for j in range(N) for q in lab.nbrs[j] if q in inj_nbrs(0, j)) // 2)
        results["override"][f"b{beta}"] = dict(true_map=c_true, injected_map=c_inj, perm=perm.tolist(),
                                                 fixed_points=int((perm == np.arange(N)).sum()), shared_edges=shared)
        # chance-level for the injected map given the true walk, for reference
    json.dump(results, open(f"{OUT}/e2.json", "w"))
    # --- unseen-node causal test
    rows = []
    for s in range(seeds):
        inst = lab.instance(query_set, s)
        rng = np.random.default_rng(8000 + s)
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk = random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X)
            walk_q = np.concatenate([walk, [X]])
            segs = [(inst, walk_q)]
            ids, seg, node = lab.build_context(segs)
            cand, col = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])),
                            P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])),
                            P_X=float(p[col[int(inst["ids"][X])]]),
                            P_Y=float(p[col[int(inst["ids"][Y])]]))
            conds = {}
            probs, _ = lab.forward(ids, cand)
            conds["none"] = masses(probs)
            F = F_gt * 2
            delta = lab.delta_from_features(seg, node, [F])
            probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=h_main)
            conds["ctx+query_trueX"] = masses(probs)
            Fy = F.copy(); Fy[X] = F[Y]           # claim X sits where Y is
            delta = lab.delta_from_features(seg, node, [Fy])
            probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=h_main)
            conds["ctx+query_claimY"] = masses(probs)
            Fq = np.zeros_like(F); Fq[X] = F[X]   # inject only at the query token
            delta = lab.delta_from_features(seg, node, [Fq])
            probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=h_main)
            conds["query_only_trueX"] = masses(probs)
            rows.append(dict(seed=s, X=X, Y=Y, conds=conds))
        print(f"[E2-unseen] seed {s}: " + " | ".join(
            f"{k}: nbrsX={np.mean([r['conds'][k]['P_nbrs_X'] for r in rows if r['seed']==s]):.3f}"
            for k in ("none", "ctx+query_trueX", "ctx+query_claimY", "query_only_trueX")))
    results["unseen"] = rows
    json.dump(results, open(f"{OUT}/e2.json", "w"))



# ============================================================================= E2b
def embed_rows(lab, inst, h=None):
    """[16,d] per-position token vectors: input embeddings (h=None) or hs index h of the word in isolation."""
    if h is None:
        W = lab.model.model.embed_tokens.weight.detach().float()
        return W[torch.tensor(inst["ids"], device=DEV)].cpu().numpy()
    rows = []
    for t in inst["ids"]:
        _, hs = lab.forward(np.array([lab.bos_id, int(t)]), [int(t)], hs_layers=(h,))
        rows.append(hs[h][-1])
    return np.stack(rows)


def analytic_features(lab, inst, h_embed, k=2):
    """calibration-free: F = Z @ (Z^T E) -> node j gets sum_i Z_ij * (sum_pos Z_pos,i * E_pos)."""
    E = embed_rows(lab, inst, h_embed).astype(np.float64)       # [16,d] position order
    AB = lab.Z3[:, :k].T @ E                                     # [k,d]
    F = lab.Z3[:, :k] @ AB
    return F.astype(np.float32)


def shared_subspace(lab, h, exclude_set, k=4):
    """orthonormal basis (d x m) of the union of top-k PC subspaces of E1 runs on word sets != exclude_set."""
    e1 = json.load(open(f"{OUT}/e1.json")); d = np.load(f"{OUT}/e1_means.npz")
    rows = []
    for r, m in enumerate(e1["runs"]):
        if m["set_id"] != exclude_set:
            Hc = nanfill(d["means"][r, h].astype(np.float64)); Hc -= Hc.mean(0)
            rows.append(top_pcs(Hc, k))
    B = np.concatenate(rows)
    Q, _ = np.linalg.qr(B.T)
    top2 = np.linalg.svd(B, full_matrices=False)[2][:2]      # dominant shared directions
    return Q, top2


def stage_e2b(lab, h_main, seeds=6, query_set=4):
    """GT content in run-specific native plane (self-calibrated) and analytic embedding-derived planes."""
    results = {"meta": dict(h_main=h_main), "dense": {}, "unseen": [], "override": {}, "diag": []}
    d_e2 = json.load(open(f"{OUT}/e2.json"))
    results["dense"]["none"] = d_e2["dense"]["none"]; results["dense"]["extracted_b2.0"] = d_e2["dense"]["extracted_b2.0"]
    norm_ref = None
    feats_cache = {}

    def self_M(inst, s):
        """M fit on a separate calibration walk on the SAME assignment (same walk extracted() uses)."""
        key = ("self", s)
        if key not in feats_cache:
            walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(777_000 + s))
            ids, seg, node = lab.build_context([(inst, walk)])
            cand, col = cand_map([inst])
            _, hs = lab.forward(ids, cand, hs_layers=(h_main,))
            Hc = nanfill(lab.node_means(hs[h_main], seg, node, 0).astype(np.float64)); Hc -= Hc.mean(0)
            M2, r2_2 = fit_M(Hc, lab.Z2); M3, r2_3 = fit_M(Hc, lab.Z3)
            P = top_pcs(Hc, 2)
            feats_cache[key] = dict(M2=M2.astype(np.float32), M3=M3.astype(np.float32), r2_2=r2_2, r2_3=r2_3, P=P, Hc=Hc)
        return feats_cache[key]

    def analytic(inst, s, h_embed, k=2):
        key = ("ana", s, h_embed, k)
        if key not in feats_cache:
            E = embed_rows(lab, inst, h_embed).astype(np.float64)
            AB = lab.Z3[:, :k].T @ E                                   # analytic plane directions [k,d]
            F = (lab.Z3[:, :k] @ AB).astype(np.float32)
            sm = self_M(inst, s)
            # diagnostics: plane alignment with the run's own PCs; scale to the self-calibrated norm
            cos = float(np.mean(principal_angles(AB, sm["P"])))
            nrm = np.linalg.norm(lab.Z2 @ sm["M2"], axis=1).mean()
            F = F * (nrm / np.linalg.norm(F, axis=1).mean())
            feats_cache[key] = dict(F=F.astype(np.float32), cos=cos)
        return feats_cache[key]

    def run_dense(name, feat_fn, nbr_fn=None):
        curves, alt = [], []
        for s in range(seeds):
            inst = lab.instance(query_set, s)
            walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))   # same walks as E2
            segs = [(inst, walk)]
            ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            F = feat_fn(inst, s)
            probs, _ = lab.forward(ids, cand, delta=lab.delta_from_features(seg, node, [F]), inject_hs=h_main)
            curves.append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
            if nbr_fn is not None:
                alt.append(lab.seg_masses(probs, seg, node, segs, col, nbr_fn)[0].tolist())
        m = np.array(curves).mean(0)
        print(f"[E2b] {name}: mass@[50,100,200,400,800,1200]=" + " ".join(f"{m[min(t, len(m))-1]:.2f}" for t in (50, 100, 200, 400, 800, 1200)))
        return curves, alt

    for beta in (1.0, 2.0, 3.0):
        results["dense"][f"self_gt2_b{beta}"] = run_dense(f"self_gt2_b{beta}", lambda inst, s, b=beta: (lab.Z2 @ self_M(inst, s)["M2"]) * b)[0]
    results["dense"]["self_gt3_b2.0"] = run_dense("self_gt3_b2.0", lambda inst, s: (lab.Z3 @ self_M(inst, s)["M3"]) * 2)[0]
    for h_embed, tag in ((None, "inE"), (h_main, f"hs{h_main}E")):
        for beta in (1.0, 2.0, 3.0):
            results["dense"][f"analytic_{tag}_gt2_b{beta}"] = run_dense(
                f"analytic_{tag}_gt2_b{beta}", lambda inst, s, b=beta, h=h_embed: analytic(inst, s, h)["F"] * b)[0]
    # --- shared-subspace planes (union of other word sets' top-4 PCs at h_main)
    Qsh, top2 = shared_subspace(lab, h_main, exclude_set=query_set)
    def shared_feats(inst, s, kind, seed=0):
        sm = self_M(inst, s); nrm = np.linalg.norm(lab.Z2 @ sm["M2"], axis=1).mean()
        if kind == "top2":
            plane = top2                                                         # [2,d]
        else:
            R2, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((Qsh.shape[1], 2)))
            plane = (Qsh @ R2).T                                                 # random 2-plane inside shared subspace
        F = lab.Z2 @ plane
        return (F * nrm / np.linalg.norm(F, axis=1).mean()).astype(np.float32)
    for beta in (2.0, 3.0):
        results["dense"][f"shared_top2_gt2_b{beta}"] = run_dense(f"shared_top2_gt2_b{beta}", lambda inst, s, b=beta: shared_feats(inst, s, "top2") * b)[0]
        for rs in (0, 1):
            results["dense"][f"shared_rand{rs}_gt2_b{beta}"] = run_dense(f"shared_rand{rs}_gt2_b{beta}", lambda inst, s, b=beta, rs=rs: shared_feats(inst, s, "rand", rs) * b)[0]
    results["meta"]["shared_dim"] = int(Qsh.shape[1])
    results["diag"] = [dict(seed=s, self_r2_2=self_M(lab.instance(query_set, s), s)["r2_2"],
                            self_r2_3=self_M(lab.instance(query_set, s), s)["r2_3"],
                            analytic_inE_cos=analytic(lab.instance(query_set, s), s, None)["cos"],
                            analytic_hsE_cos=analytic(lab.instance(query_set, s), s, h_main)["cos"]) for s in range(seeds)]
    print("[E2b] diag: " + " ".join(f"{k}={np.mean([r[k] for r in results['diag']]):.3f}" for k in results["diag"][0] if k != "seed"))
    json.dump(results, open(f"{OUT}/e2b.json", "w"))
    # --- override + unseen with self-calibrated GT (beta 2)
    perm = np.array(d_e2["override"]["b2.0"]["perm"]); inv = np.argsort(perm)
    def inj_nbrs(s, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    c_true, c_inj = run_dense("self_override_b2.0", lambda inst, s: (lab.Z2[perm] @ self_M(inst, s)["M2"]) * 2, nbr_fn=inj_nbrs)
    results["override"]["self_b2.0"] = dict(true_map=c_true, injected_map=c_inj)
    c_true, c_inj = run_dense("self_override_b3.0", lambda inst, s: (lab.Z2[perm] @ self_M(inst, s)["M2"]) * 3, nbr_fn=inj_nbrs)
    results["override"]["self_b3.0"] = dict(true_map=c_true, injected_map=c_inj)
    rows = []
    for s in range(seeds):
        inst = lab.instance(query_set, s); F = (lab.Z2 @ self_M(inst, s)["M2"]) * 2
        rng = np.random.default_rng(8000 + s)
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
            segs = [(inst, walk_q)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
            def masses(probs):
                p = probs[-1]
                return dict(P_nbrs_X=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[X])),
                            P_nbrs_Y=float(sum(p[col[int(inst["ids"][q])]] for q in lab.nbrs[Y])))
            conds = {}
            conds["none"] = masses(lab.forward(ids, cand)[0])
            conds["self_trueX"] = masses(lab.forward(ids, cand, delta=lab.delta_from_features(seg, node, [F]), inject_hs=h_main)[0])
            Fy = F.copy(); Fy[X] = F[Y]
            conds["self_claimY"] = masses(lab.forward(ids, cand, delta=lab.delta_from_features(seg, node, [Fy]), inject_hs=h_main)[0])
            Fq = np.zeros_like(F); Fq[X] = F[X]
            conds["self_queryOnly"] = masses(lab.forward(ids, cand, delta=lab.delta_from_features(seg, node, [Fq]), inject_hs=h_main)[0])
            rows.append(dict(seed=s, X=X, Y=Y, conds=conds))
    results["unseen"] = rows
    for c in ("none", "self_trueX", "self_claimY", "self_queryOnly"):
        print(f"[E2b-unseen] {c}: nbrsX={np.mean([r['conds'][c]['P_nbrs_X'] for r in rows]):.3f} nbrsY={np.mean([r['conds'][c]['P_nbrs_Y'] for r in rows]):.3f}")
    json.dump(results, open(f"{OUT}/e2b.json", "w"))


# ============================================================================= E2base
def stage_e2base(lab, h_main, seeds=12, query_set=4):
    """baseline + extracted reference (+ baseline scored under the override map) for N seeds, same walks as E2/E2b."""
    d_e2 = json.load(open(f"{OUT}/e2.json"))
    perm = np.array(d_e2["override"]["b2.0"]["perm"]); inv = np.argsort(perm)
    def inj_nbrs(s, j): return [int(inv[q]) for q in lab.nbrs[perm[j]]]
    out = {"dense": {"none": [], "extracted_b2.0": []}, "override": {"none_injmap": []}}
    for s in range(seeds):
        inst = lab.instance(query_set, s)
        walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(5000 + s))
        segs = [(inst, walk)]; ids, seg, node = lab.build_context(segs); cand, col = cand_map([inst])
        probs, _ = lab.forward(ids, cand)
        out["dense"]["none"].append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
        out["override"]["none_injmap"].append(lab.seg_masses(probs, seg, node, segs, col, inj_nbrs)[0].tolist())
        F = extracted_features(lab, inst, h_main, s) * 2
        probs, _ = lab.forward(ids, cand, delta=lab.delta_from_features(seg, node, [F]), inject_hs=h_main)
        out["dense"]["extracted_b2.0"].append(lab.seg_masses(probs, seg, node, segs, col)[0].tolist())
        print(f"[E2base] seed {s}: none final={np.mean(out['dense']['none'][-1][-200:]):.3f} extracted final={np.mean(out['dense']['extracted_b2.0'][-1][-200:]):.3f}")
    json.dump(out, open(f"{OUT}/e2base.json", "w"))


# ============================================================================= E3
def stage_e3(lab, h_main, seeds=6, query_set=4, demo_sets=(1, 2, 3), cal_sets=(0,)):
    M, r2 = calibrate_M(lab, h_main, cal_sets)
    F_gt = gt_features(lab, M) * 2.0
    norm_gt = float(np.linalg.norm(F_gt, axis=1).mean())
    F_rand = random_features(lab, norm_gt / 2.0) * 2.0
    results = {"meta": dict(h_main=h_main, r2=r2, T_DEMO=T_DEMO, T_QUERY=T_QUERY), "curves": {}, "unseen": {}}

    def build(s, n_demos):
        segs = []
        for k in range(n_demos):
            inst = lab.instance(demo_sets[k], s)
            segs.append((inst, random_walk(lab.nbrs, T_DEMO, np.random.default_rng(6000 + 100 * s + k))))
        instq = lab.instance(query_set, s)
        segs.append((instq, random_walk(lab.nbrs, T_QUERY, np.random.default_rng(7000 + s))))
        return segs

    def run(name, n_demos, demo_inj, query_inj, F, demo_perm=False):
        per_seg = []
        for s in range(seeds):
            segs = build(s, n_demos)
            ids, seg, node = lab.build_context(segs)
            cand, col = cand_map([i for i, _ in segs])
            if demo_perm:  # each demo grid gets geometrically scrambled (but internally consistent) features
                feats = [F[np.random.default_rng(31 + 10 * s + k).permutation(N)] for k in range(n_demos)] + [F if query_inj else None]
            else:
                feats = [F if demo_inj else None] * n_demos + [F if query_inj else None]
            delta = lab.delta_from_features(seg, node, feats) if any(f is not None for f in feats) else None
            probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=h_main)
            per_seg.append([m.tolist() for m in lab.seg_masses(probs, seg, node, segs, col)])
        q = np.array([ps[-1] for ps in per_seg]).mean(0)
        print(f"[E3] {name}: query mass@[50,100,200,400,600]=" +
              " ".join(f"{q[min(t, len(q))-1]:.2f}" for t in (50, 100, 200, 400, 600)) +
              (f" | demo finals=" + " ".join(f"{np.mean([ps[k][-100:] for ps in per_seg]):.2f}" for k in range(n_demos)) if n_demos else ""))
        results["curves"][name] = per_seg

    for n in (0, 1, 2, 3):
        run(f"native_demos{n}", n, True, True, F_gt)
        run(f"none_demos{n}", n, False, False, F_gt)
    run("native_demos3_demoOnly", 3, True, False, F_gt)
    run("native_demos3_queryOnly", 3, False, True, F_gt)
    run("native_demos3_demoScrambled", 3, True, True, F_gt, demo_perm=True)
    for n in (0, 3):
        run(f"random_demos{n}", n, True, True, F_rand)
    json.dump(results, open(f"{OUT}/e3.json", "w"))

    # --- unseen-node on the query grid, with 0 vs 3 injected demos
    rows = []
    for s in range(seeds):
        instq = lab.instance(query_set, s)
        rng = np.random.default_rng(9000 + s)
        for X in rng.choice(N, 3, replace=False):
            X = int(X); Y = far_node(lab, X, rng)
            walk_q = np.concatenate([random_walk(lab.nbrs, T_UNSEEN, rng, avoid=X), [X]])
            for n_demos in (0, 3):
                segs = [(lab.instance(demo_sets[k], s),
                         random_walk(lab.nbrs, T_DEMO, np.random.default_rng(6000 + 100 * s + k)))
                        for k in range(n_demos)] + [(instq, walk_q)]
                ids, seg, node = lab.build_context(segs)
                cand, col = cand_map([i for i, _ in segs])
                def masses(probs):
                    p = probs[-1]
                    return dict(P_nbrs_X=float(sum(p[col[int(instq["ids"][q])]] for q in lab.nbrs[X])),
                                P_nbrs_Y=float(sum(p[col[int(instq["ids"][q])]] for q in lab.nbrs[Y])))
                for cname, Fq, Fd in (("none", None, None), ("native_trueX", F_gt, F_gt),
                                      ("native_claimY", None, F_gt), ("random_trueX", F_rand, F_rand)):
                    if cname == "native_claimY":
                        Fq = F_gt.copy(); Fq[X] = F_gt[Y]
                    feats = [Fd] * n_demos + [Fq]
                    delta = lab.delta_from_features(seg, node, feats) if Fq is not None else None
                    probs, _ = lab.forward(ids, cand, delta=delta, inject_hs=h_main)
                    rows.append(dict(seed=s, X=X, Y=Y, n_demos=n_demos, cond=cname, **masses(probs)))
        for n_demos in (0, 3):
            print(f"[E3-unseen] seed {s} demos={n_demos}: " + " | ".join(
                f"{c}: nbrsX={np.mean([r['P_nbrs_X'] for r in rows if r['seed']==s and r['n_demos']==n_demos and r['cond']==c]):.3f}"
                for c in ("none", "native_trueX", "native_claimY", "random_trueX")))
    results["unseen"] = rows
    json.dump(results, open(f"{OUT}/e3.json", "w"))


# ============================================================================= main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "e1", "e2", "e3", "e2b", "e2base"])
    ap.add_argument("--h_main", type=int, default=None, help="hidden-state index for M fit + injection")
    ap.add_argument("--layers", type=str, default=None, help="comma list of hs indices for E2 layer sweep")
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    lab = Lab()
    h_main = args.h_main
    if args.stage in ("all", "e1"):
        h_energy, h_xfer = stage_e1(lab)
        if h_main is None:
            h_main = h_energy
    if h_main is None:
        h_main = json.load(open(f"{OUT}/e1.json"))["h_energy"]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else \
        sorted({h_main, max(8, h_main - 6), max(8, h_main - 12), max(8, h_main - 20)})
    assert 1 <= h_main < lab.n_layers and all(1 <= L < lab.n_layers for L in layers)
    print(f"[main] h_main={h_main} layers={layers}")
    if args.stage in ("all", "e2"):
        stage_e2(lab, h_main, layers, seeds=args.seeds)
    if args.stage in ("all", "e3"):
        stage_e3(lab, h_main, seeds=args.seeds)
    if args.stage == "e2b":
        stage_e2b(lab, h_main, seeds=args.seeds)
    if args.stage == "e2base":
        stage_e2base(lab, h_main, seeds=args.seeds)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")
