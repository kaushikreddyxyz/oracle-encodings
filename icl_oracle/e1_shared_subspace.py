"""Do context-specific position planes live inside a SHARED higher-dim subspace?
For hs in layers: for each ordered pair of runs (i,j), fraction of run j's top-2 PC plane (norm^2)
inside run i's top-k PC subspace, k in {2,4,8,16}; grouped by relation; random baseline = k/d."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/results_gt"
e1 = json.load(open(f"{OUT}/e1.json")); d = np.load(f"{OUT}/e1_means.npz"); means = d["means"]; runs = e1["runs"]; R = len(runs)
def rel(i, j):
    a, b = runs[i], runs[j]
    if i == j: return "self"
    if a["set_id"] == b["set_id"] and a["assign_seed"] == b["assign_seed"]: return "same_assign"
    if a["set_id"] == b["set_id"]: return "diff_assign"
    return "diff_set"
rep = {}
for h in (8, 16, 26, 34, 40):
    Hc = []
    for r in range(R):
        x = means[r, h].astype(np.float64); x = np.where(np.isnan(x), np.nanmean(x, 0, keepdims=True), x); Hc.append(x - x.mean(0))
    V = [np.linalg.svd(x, full_matrices=False)[2] for x in Hc]   # 16 x d rows
    out = {}
    for k in (2, 4, 8, 15):
        fr = {"same_assign": [], "diff_assign": [], "diff_set": []}
        for i in range(R):
            Q = V[i][:k].T                     # d x k orthonormal
            for j in range(R):
                if i == j: continue
                P2 = V[j][:2]                  # 2 x d
                fr[rel(i, j)].append(float(np.sum((P2 @ Q) ** 2) / 2))
        out[k] = {kk: float(np.mean(v)) for kk, v in fr.items()}
    # also: pooled subspace across ALL other runs (union of their top-4 PCs, orthonormalized) -> is a run's plane inside?
    pooled = []
    for j in range(R):
        B = np.concatenate([V[i][:4] for i in range(R) if i != j and runs[i]["set_id"] != runs[j]["set_id"]])  # other word sets only
        Q, _ = np.linalg.qr(B.T)
        pooled.append(float(np.sum((V[j][:2] @ Q) ** 2) / 2))
    rep[h] = dict(by_k=out, pooled_other_sets_top4=float(np.mean(pooled)), pooled_dim=int(B.shape[0]))
    print(f"hs{h:2d} | frac of plane_j inside top-k subspace of run i (rand=k/3584): " +
          " | ".join(f"k={k}: same-assign {out[k]['same_assign']:.2f} diff-assign {out[k]['diff_assign']:.2f} diff-set {out[k]['diff_set']:.2f}" for k in (2, 4, 8, 15)) +
          f" | inside pooled top-4 of all OTHER word sets (dim {B.shape[0]}, rand {B.shape[0]/3584:.3f}): {np.mean(pooled):.2f}")
json.dump(rep, open(f"{OUT}/e1_shared_subspace.json", "w"), indent=1)
