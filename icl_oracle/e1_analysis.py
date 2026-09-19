"""E1 follow-up: is the in-context position plane spanned by the context words' own embeddings?

For each E1 run r (word set, assignment, walk): H_r = centered node means at hs h [16,d];
P_r = top-2 PCs of H_r. Test (a) walk-to-walk / assignment-to-assignment / set-to-set stability of
P_r; (b) fraction of P_r's directions lying in span(E_r), E_r = input embeddings of the 16 words of
that run (a 16-dim subspace of R^d; random baseline ~ 16/d); (c) alignment between the analytic
directions a = E_r^T z2, b = E_r^T z3 (spectral-weighted embedding sums) and P_r.
Runs on the pod (needs the embedding matrix only)."""
import glob, json, os, sys
import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results_gt"
e1 = json.load(open(f"{OUT}/e1.json")); d = np.load(f"{OUT}/e1_means.npz")
means = d["means"]  # [R, L, 16, d]
runs = e1["runs"]; R = len(runs)

# ---- embeddings of each run's words
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-9b/snapshots/*"))[0]
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
shard = f"{snap}/{idx['model.embed_tokens.weight']}"
with safe_open(shard, "pt") as f:
    W = f.get_tensor("model.embed_tokens.weight").float().numpy()
sys.path.insert(0, HERE)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-2-9b")
E = []
for m in runs:
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in m["words"]]
    E.append(W[ids])            # [16, d] in position order
E = np.stack(E)
D = W.shape[1]

from gt_inject import spectral_coords, build_grid, top_pcs, principal_angles
_, _, coords = build_grid(); Z = spectral_coords(coords, 3)


def frac_in_span(vecs, basis):
    """fraction of squared norm of each row of vecs (k,d) inside row-span of basis (m,d)."""
    Q, _ = np.linalg.qr(basis.T)          # d x m orthonormal
    proj = vecs @ Q
    return (np.sum(proj ** 2, 1) / np.sum(vecs ** 2, 1)).tolist()


def rel(i, j):
    a, b = runs[i], runs[j]
    if i == j: return "self"
    if a["set_id"] == b["set_id"] and a["assign_seed"] == b["assign_seed"]: return "same_set_same_assign"
    if a["set_id"] == b["set_id"]: return "same_set_diff_assign"
    return "diff_set"


rng = np.random.default_rng(0)
report = {}
for h in [8, 16, 20, 26, 30, 34, 40]:
    Hc = [means[r, h].astype(np.float64) for r in range(R)]
    Hc = [np.where(np.isnan(x), np.nanmean(x, 0, keepdims=True), x) for x in Hc]
    Hc = [x - x.mean(0) for x in Hc]
    P = [top_pcs(x, 2) for x in Hc]
    # (a) stability of PC planes by relation
    stab = {k: [] for k in ("same_set_same_assign", "same_set_diff_assign", "diff_set")}
    for i in range(R):
        for j in range(R):
            if i != j:
                stab[rel(i, j)].append(np.mean(principal_angles(P[i], P[j])))
    stab = {k: float(np.mean(v)) for k, v in stab.items()}
    # (b) fraction of PC directions inside span of own embeddings vs other run's embeddings vs random 16-dim
    own = [np.mean(frac_in_span(P[r], E[r])) for r in range(R)]
    other = [np.mean(frac_in_span(P[r], E[(r + 5) % R])) for r in range(R)]   # a different word set
    rand = [np.mean(frac_in_span(P[r], rng.standard_normal((16, D)))) for r in range(R)]
    # (b') same for the full centered node-mean matrix: variance fraction of Hc inside span(E_r)
    hc_own = [float(np.sum((Hc[r] @ np.linalg.qr(E[r].T)[0]) ** 2) / np.sum(Hc[r] ** 2)) for r in range(R)]
    hc_rand = [float(np.sum((Hc[r] @ np.linalg.qr(rng.standard_normal((D, 16)))[0]) ** 2) / np.sum(Hc[r] ** 2)) for r in range(R)]
    # (c) analytic directions a=E^T z2, b=E^T z3 vs PC plane; and R^2 of Hc explained by Z @ [a;b]
    ana_cos, ana_r2 = [], []
    for r in range(R):
        AB = (Z[:, :2].T @ E[r])                       # [2, d]
        ana_cos.append(np.mean(principal_angles(AB, P[r])))
        M, *_ = np.linalg.lstsq(Z[:, :2] @ AB, Hc[r], rcond=None)  # allow scaling/rotation within the plane
        pred = (Z[:, :2] @ AB) @ M
        ana_r2.append(float(1 - np.sum((Hc[r] - pred) ** 2) / np.sum(Hc[r] ** 2)))
    report[h] = dict(plane_stability=stab, pc_in_own_embed_span=float(np.mean(own)),
                     pc_in_other_embed_span=float(np.mean(other)), pc_in_random16_span=float(np.mean(rand)),
                     hc_var_in_own_embed_span=float(np.mean(hc_own)), hc_var_in_random16_span=float(np.mean(hc_rand)),
                     analytic_ab_vs_pc_cos=float(np.mean(ana_cos)), analytic_r2=float(np.mean(ana_r2)))
    s = report[h]
    print(f"hs{h:2d} | plane stability cos: same-assign {s['plane_stability']['same_set_same_assign']:.2f} "
          f"diff-assign {s['plane_stability']['same_set_diff_assign']:.2f} diff-set {s['plane_stability']['diff_set']:.2f} "
          f"| PC in span(own E) {s['pc_in_own_embed_span']:.2f} (other E {s['pc_in_other_embed_span']:.2f}, rand16 {s['pc_in_random16_span']:.3f}) "
          f"| Hc var in span(own E) {s['hc_var_in_own_embed_span']:.2f} (rand {s['hc_var_in_random16_span']:.3f}) "
          f"| analytic E^T z vs PC cos {s['analytic_ab_vs_pc_cos']:.2f} R2 {s['analytic_r2']:.2f}")
json.dump(report, open(f"{OUT}/e1_analysis.json", "w"), indent=1)
