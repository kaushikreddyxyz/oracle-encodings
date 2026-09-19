"""In-context world models on a grid graph (Park et al. 2501.00070) + oracle injections.

Task: random walk on a 4x4 grid whose nodes are semantically unrelated single-token
words. Replicates rule-following accuracy / representation formation, then tests:
  A) inject structure coordinates into SHORT contexts -> faster in-context learning?
  B) inject a phantom-neighbor direction at the query -> edit the world state?

Runs on CUDA (H100). Saves npz/json artifacts to results/ for local plotting.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = os.environ.get("ICL_MODEL", "google/gemma-2-9b")
WORDS = ["apple", "bird", "car", "water", "sand", "math", "fire", "moon",
         "tree", "salt", "gold", "wind", "rain", "star", "book", "door"]
BACKUP = ["milk", "iron", "song", "road", "fish", "snow", "king", "hand"]
SIDE = 4          # 4x4 grid, 16 nodes
T_LONG = 1200     # long-walk length (tokens)
CKPTS = [50, 100, 200, 400, 800, 1200]
NW = 100          # window for mean reps
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def build_grid(n_side):
    n = n_side * n_side
    A = np.zeros((n, n), dtype=np.float32)
    coords = [(r, c) for r in range(n_side) for c in range(n_side)]
    for i, (r, c) in enumerate(coords):
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_side and 0 <= cc < n_side:
                A[i, coords.index((rr, cc))] = 1.0
    nbrs = [np.nonzero(A[i])[0].tolist() for i in range(n)]
    return A, nbrs, coords


def random_walk(nbrs, T, rng):
    walk = [rng.integers(len(nbrs))]
    for _ in range(T - 1):
        walk.append(int(rng.choice(nbrs[walk[-1]])))
    return np.array(walk)


class Lab:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
        words = []
        for w in WORDS + BACKUP:
            ids = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                words.append((w, ids[0]))
            if len(words) == SIDE * SIDE:
                break
        assert len(words) == SIDE * SIDE, f"only {len(words)} single-token words"
        self.words = [w for w, _ in words]
        self.ids = np.array([i for _, i in words])
        self.A, self.nbrs, self.coords = build_grid(SIDE)
        self.hooks = []

    def input_ids(self, walk):
        ids = [self.tok.bos_token_id] + [int(self.ids[j]) for j in walk]
        return torch.tensor([ids], device=DEV)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def add_hook(self, layer_idx, delta_fn):
        """delta_fn(hidden [1,T,d]) -> modified hidden. layer_idx = decoder layer index."""
        def fn(mod, inp, out):
            if isinstance(out, tuple):
                return (delta_fn(out[0]),) + out[1:]
            return delta_fn(out)
        self.hooks.append(self.model.model.layers[layer_idx].register_forward_hook(fn))

    @torch.inference_mode()
    def forward(self, walk, want_hidden=False):
        out = self.model(self.input_ids(walk), output_hidden_states=want_hidden)
        probs = torch.softmax(out.logits[0].float(), dim=-1)  # [T+1, V]
        node_probs = probs[:, torch.tensor(self.ids, device=DEV)].cpu().numpy()
        hidden = None
        if want_hidden:
            hidden = [h[0].float().cpu().numpy() for h in out.hidden_states]  # 33 x [T+1, d]
        return node_probs, hidden

    def neighbor_mass(self, walk, node_probs):
        """rule-following prob at each step t (predicting step t+1)."""
        mass = np.zeros(len(walk) - 1)
        for t in range(len(walk) - 1):
            mass[t] = node_probs[t + 1, self.nbrs[walk[t]]].sum()  # logits at input pos t+1
        return mass

    def window_reps(self, walk, hidden, ckpt, nw=NW):
        """mean rep per node over steps [ckpt-nw, ckpt); hs positions = step+1."""
        lo = max(0, ckpt - nw)
        reps = np.full((len(hidden), len(self.ids), hidden[0].shape[1]), np.nan, np.float32)
        for j in range(len(self.ids)):
            pos = [s + 1 for s in range(lo, ckpt) if walk[s] == j]
            if pos:
                for l, h in enumerate(hidden):
                    reps[l, j] = h[pos].mean(0)
        return reps

    @staticmethod
    def norm_energy(reps_l, A):
        """Dirichlet energy / total variance, NaN-safe."""
        ok = ~np.isnan(reps_l[:, 0])
        H = reps_l[ok]
        E = sum(A[i, j] * np.sum((reps_l[i] - reps_l[j]) ** 2)
                for i in range(len(A)) for j in range(len(A))
                if A[i, j] and ok[i] and ok[j])
        var = np.sum((H - H.mean(0)) ** 2)
        return float(E / (2 * var + 1e-9))


def stage_replicate(lab, seeds=4):
    all_mass, all_reps, walks = [], [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        walk = random_walk(lab.nbrs, T_LONG, rng)
        node_probs, hidden = lab.forward(walk, want_hidden=True)
        mass = lab.neighbor_mass(walk, node_probs)
        reps = np.stack([lab.window_reps(walk, hidden, c) for c in CKPTS])  # [C,33,16,d]
        all_mass.append(mass); all_reps.append(reps); walks.append(walk)
        print(f"[rep] seed {s}: final-100 mean neighbor mass = {mass[-100:].mean():.3f}")
    energy = np.array([[[Lab.norm_energy(all_reps[s][c, l], lab.A)
                         for l in range(all_reps[0].shape[1])]
                        for c in range(len(CKPTS))] for s in range(seeds)])
    np.savez_compressed(
        f"{OUT}/replicate.npz", mass=np.stack(all_mass), walks=np.stack(walks),
        reps=np.stack(all_reps).astype(np.float16), energy=energy, ckpts=np.array(CKPTS))
    # pick structure layer: min mean energy at final ckpt over mid/late hs
    e_fin = energy[:, -1, :].mean(0)
    lo, hi = 6, len(e_fin) - 2
    h_star = int(np.argmin(e_fin[lo:hi]) + lo)
    print(f"[rep] energy-min hidden idx h*={h_star}/{len(e_fin) - 1} (E={e_fin[h_star]:.3f})")
    json.dump({"h_star": h_star, "words": lab.words, "coords": lab.coords,
               "energy_final": e_fin.tolist()}, open(f"{OUT}/replicate.json", "w"))
    return h_star


def structure_vecs(reps_final, h, k=3):
    """centered node means at hs idx h, projected on top-k PCs. [16,d]"""
    H = reps_final[h].astype(np.float64)
    H = np.where(np.isnan(H), np.nanmean(H, 0, keepdims=True), H)
    mu = H.mean(0)
    Hc = H - mu
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    proj = (Hc @ Vt[:k].T) @ Vt[:k]
    return proj.astype(np.float32), mu.astype(np.float32), Vt[:k].astype(np.float32)


def stage_inject_fast(lab, h_star, seeds=8):
    d = np.load(f"{OUT}/replicate.npz")
    S_ext, _, _ = structure_vecs(d["reps"][0, -1].astype(np.float32), h_star)  # from seed0 long run
    scale = np.linalg.norm(S_ext, axis=1).mean()
    rng0 = np.random.default_rng(123)
    Q, _ = np.linalg.qr(rng0.standard_normal((lab.model.config.hidden_size, 2)))
    gc = np.array(lab.coords, np.float32); gc -= gc.mean(0)
    S_arb = (gc @ Q.T.astype(np.float32))
    S_arb *= scale / np.linalg.norm(S_arb, axis=1).mean()
    perm = np.random.default_rng(7).permutation(len(S_ext))
    conds = {"none": None, "extracted": S_ext, "shuffled": S_ext[perm], "arbitrary": S_arb}
    lengths = [16, 32, 64, 128]
    res = {}
    for cname, S in conds.items():
        betas = [0.0] if S is None else [1.0, 2.0]
        layers = [h_star - 1] if S is None else [h_star - 1, 8]
        for L in layers:
            for beta in betas:
                key = f"{cname}|L{L}|b{beta}"
                accs = np.zeros((len(lengths), seeds))
                for li, T in enumerate(lengths):
                    for s in range(seeds):
                        walk = random_walk(lab.nbrs, T, np.random.default_rng(1000 + s))
                        if S is not None:
                            Sg = torch.tensor(S * beta, dtype=torch.bfloat16, device=DEV)
                            wt = torch.tensor(walk, device=DEV)
                            def add(hidden, Sg=Sg, wt=wt):
                                hidden[:, 1:, :] += Sg[wt]
                                return hidden
                            lab.add_hook(L, add)
                        node_probs, _ = lab.forward(walk)
                        lab.clear_hooks()
                        accs[li, s] = node_probs[-1, lab.nbrs[walk[-1]]].sum()
                res[key] = accs.mean(1).tolist()
                print(f"[fast] {key}: " + " ".join(f"T{t}={a:.3f}" for t, a in zip(lengths, res[key])))
                if S is None:
                    break
            if S is None:
                break
    json.dump({"lengths": lengths, "res": res}, open(f"{OUT}/inject_fast.json", "w"))


def stage_fast_curve(lab, h_star, seeds=4):
    """Dense accuracy-vs-context curves: same long walks with/without injection."""
    d = np.load(f"{OUT}/replicate.npz")
    S_ext, _, _ = structure_vecs(d["reps"][0, -1].astype(np.float32), h_star)
    scale = np.linalg.norm(S_ext, axis=1).mean()
    rng0 = np.random.default_rng(123)
    Q, _ = np.linalg.qr(rng0.standard_normal((lab.model.config.hidden_size, 2)))
    gc = np.array(lab.coords, np.float32); gc -= gc.mean(0)
    S_arb = gc @ Q.T.astype(np.float32)
    S_arb *= scale / np.linalg.norm(S_arb, axis=1).mean()
    perm = np.random.default_rng(7).permutation(len(S_ext))
    conds = {"none": None, "extracted_b2": S_ext * 2, "shuffled_b2": S_ext[perm] * 2,
             "arbitrary_b2": S_arb * 2}
    L = h_star - 1
    curves = {c: [] for c in conds}
    for s in range(seeds):
        walk = random_walk(lab.nbrs, T_LONG, np.random.default_rng(2000 + s))
        for cname, S in conds.items():
            if S is not None:
                Sg = torch.tensor(S, dtype=torch.bfloat16, device=DEV)
                wt = torch.tensor(walk, device=DEV)
                def add(hidden, Sg=Sg, wt=wt):
                    hidden[:, 1:, :] += Sg[wt]
                    return hidden
                lab.add_hook(L, add)
            node_probs, _ = lab.forward(walk)
            lab.clear_hooks()
            curves[cname].append(lab.neighbor_mass(walk, node_probs).tolist())
        print(f"[curve] seed {s} done")
    json.dump({"L": L, "curves": curves}, open(f"{OUT}/fast_curve.json", "w"))


def stage_inject_phantom(lab, h_star, seeds=4):
    rows = []
    for s in range(seeds):
        rng = np.random.default_rng(50 + s)
        walk = random_walk(lab.nbrs, T_LONG, rng)
        node_probs, hidden = lab.forward(walk, want_hidden=True)
        reps = lab.window_reps(walk, hidden, T_LONG)
        S, mu, V2 = structure_vecs(reps, h_star, k=2)
        i = int(walk[-1])
        dist = np.abs(np.array(lab.coords) - np.array(lab.coords[i])).sum(1)
        dist[lab.nbrs[i]] = 0
        k = int(np.argmax(dist))  # far non-neighbor
        base = dict(cond="null", P_k=float(node_probs[-1, k]),
                    P_true_nbrs=float(node_probs[-1, lab.nbrs[i]].sum()),
                    P_k_nbrs=float(node_probs[-1, lab.nbrs[k]].sum()), seed=s, i=i, k=k)
        rows.append(base)
        mu_t = torch.tensor(mu, dtype=torch.bfloat16, device=DEV)
        V2_t = torch.tensor(V2, dtype=torch.bfloat16, device=DEV)      # [2,d]
        ck = torch.tensor((S[k] @ V2.T), dtype=torch.bfloat16, device=DEV)  # k's PC coords
        Sk = torch.tensor(S[k], dtype=torch.bfloat16, device=DEV)
        def move(hidden):
            h = hidden[:, -1, :]
            c = (h - mu_t) @ V2_t.T
            hidden[:, -1, :] = h - c @ V2_t + ck @ V2_t
            return hidden
        interventions = [("move", move)]
        for beta in (1.0, 2.0):
            interventions.append((f"phantom_b{beta}",
                                  lambda hidden, b=beta: (hidden[:, -1, :].add_(b * Sk), hidden)[1]))
        for cname, fn in interventions:
            lab.add_hook(h_star - 1, fn)
            npr, _ = lab.forward(walk)
            lab.clear_hooks()
            rows.append(dict(cond=cname, P_k=float(npr[-1, k]),
                             P_true_nbrs=float(npr[-1, lab.nbrs[i]].sum()),
                             P_k_nbrs=float(npr[-1, lab.nbrs[k]].sum()), seed=s, i=i, k=k))
        print(f"[phantom] seed {s} done (i={lab.words[i]}, k={lab.words[k]})")
    json.dump(rows, open(f"{OUT}/inject_phantom.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "replicate", "fast", "phantom", "curve"])
    ap.add_argument("--h_star", type=int, default=None)
    args = ap.parse_args()
    lab = Lab()
    h_star = args.h_star
    if args.stage in ("all", "replicate"):
        h_star = stage_replicate(lab)
    if h_star is None:
        h_star = json.load(open(f"{OUT}/replicate.json"))["h_star"]
    if args.stage in ("all", "fast"):
        stage_inject_fast(lab, h_star)
    if args.stage in ("all", "curve"):
        stage_fast_curve(lab, h_star)
    if args.stage in ("all", "phantom"):
        stage_inject_phantom(lab, h_star)
    print("DONE")
