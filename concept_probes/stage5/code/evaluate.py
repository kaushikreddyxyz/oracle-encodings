"""Generated-split metrics + controls for one family's trained probes.

Per (class, layer, candidate) on the generated val / form_holdout splits:
  - token-level Spearman ρ (masked), AUROC/AUPRC (y binarized at 0.5)
  - example-level operating point: τ = 95th pct of neutral val example scores
    (max-pooled over tokens); implicit recall + homograph FPR at τ (§6.3)
  - lexical G-ratio: AUROC(form_test pos vs neutrals) / AUROC(explicit val pos
    vs neutrals); implicit-slice fallback for single-form classes (deviation #2)
  - controls: shuffled-label ridge refit, Hewitt–Liang token-type control refit,
    random-direction ρ distribution (§6.3/§6.4)
  - bootstrap (example-level) CI on ρ; cross-seed std
Candidates: adam (per seed, chosen λ), ridge, dom, lda, logistic.
After the layer loop: §5.3 ensemble precondition + ridge stacking on val.

Positivity conventions (labels are judge-truth, Stage-4 handoff §semantics):
an example is "concept-present" iff its max token target ≥ 0.34 (≥ judge level
2/6) — the 2B-plausible-activation rule pins faint echoes at ~0.167, which
count as absent for recall/FPR denominators.

  python evaluate.py --family months --cache ... --probes ... --stage4 ... \
      --natstats ... --layers ... --out metrics/months.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from common import FamilyData, load_natstats, stable_seed

POS_THRESH = 0.34
PRIMARY = "ridge"        # exact minimizer; Adam kept as seed diagnostic          # judge-truth concept-present cutoff (>= level 2/6)
NEUTRAL_Q = 0.95           # τ = this quantile of neutral example scores


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    if scores_pos.size == 0 or scores_neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([scores_pos, scores_neg]))
    n_p, n_n = scores_pos.size, scores_neg.size
    return float((r[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return float("nan")
    order = np.argsort(-scores)
    yt = y_true[order]
    tp = np.cumsum(yt)
    prec = tp / (np.arange(yt.size) + 1)
    return float((prec * yt).sum() / yt.sum())


class LayerScorer:
    """Predictions of every candidate for one (family, layer)."""

    def __init__(self, probes_npz: Path, acts: torch.Tensor, mu, sd, device):
        z = np.load(probes_npz, allow_pickle=False)
        self.z = z
        self.classes = [str(c) for c in z["classes"]]
        self.lambdas = z["lambdas"]
        self.acts, self.mu, self.sd, self.device = acts, mu, sd, device

    def _project(self, idx: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
        out = np.empty(idx.size, dtype=np.float32)
        w_t = torch.from_numpy(np.ascontiguousarray(W)).to(self.device).float()
        for s in range(0, idx.size, 500_000):
            sl = torch.from_numpy(idx[s:s + 500_000]).to(self.device)
            h = (self.acts[sl].float() - self.mu) / self.sd
            out[s:s + 500_000] = (h @ w_t).cpu().numpy()
        return out + b

    def preds(self, ci: int, idx: np.ndarray) -> dict[str, np.ndarray]:
        z = self.z
        res = {}
        for s in range(z["seeds"].size):
            li = int(z["chosen_lambda_idx"][s, ci])
            res[f"adam_s{s}"] = self._project(idx, z["W_adam"][s, li, ci], float(z["b_adam"][s, li, ci]))
        if "chosen_lambda_ridge" in z:
            li = int(z["chosen_lambda_ridge"][ci])
        else:
            li = int(np.bincount(z["chosen_lambda_idx"][:, ci]).argmax())
        res["ridge"] = self._project(idx, z["W_ridge"][li, ci], float(z["b_ridge"][li, ci]))
        res["ridge_lambda_idx"] = li
        res["dom"] = self._project(idx, z["W_dom"][ci], 0.0)
        res["lda"] = self._project(idx, z["W_lda"][ci], 0.0)
        res["logistic"] = self._project(idx, z["W_logistic"][ci], float(z["b_logistic"][ci]))
        res["adam"] = np.mean([res[f"adam_s{s}"] for s in range(z["seeds"].size)], axis=0)
        return res

    def rand_preds(self, idx: np.ndarray) -> np.ndarray:
        R = self.z["rand_dirs"]
        out = np.empty((R.shape[0], idx.size), dtype=np.float32)
        for k in range(R.shape[0]):
            out[k] = self._project(idx, R[k], 0.0)
        return out


def example_max(scores: np.ndarray, ex: np.ndarray, n_ex: int) -> np.ndarray:
    out = np.full(n_ex, -np.inf, dtype=np.float32)
    np.maximum.at(out, ex, scores)
    return out


def ridge_refit(acts, mu, sd, idx, y, lam, device):
    idx_t = torch.from_numpy(idx).to(device)
    y_t = torch.from_numpy(y).to(device).float()
    d = acts.shape[1]
    n = idx_t.numel()
    A = torch.zeros(d, d, device=device)
    hsum = torch.zeros(d, device=device)
    hy = torch.zeros(d, device=device)
    for s in range(0, n, 400_000):
        h = (acts[idx_t[s:s + 400_000]].float() - mu) / sd
        A += h.T @ h
        hsum += h.sum(0)
        hy += h.T @ y_t[s:s + 400_000]
    hbar, ybar = hsum / n, y_t.mean()
    A -= n * torch.outer(hbar, hbar)
    c = hy - n * hbar * ybar
    w = torch.linalg.solve(A + lam * n * torch.eye(d, device=device), c)
    return w.cpu().numpy(), float(ybar - w @ hbar)


def evaluate_family(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",")]
    classes = args.classes.split(",") if args.classes else None
    fam = FamilyData(Path(args.stage4) / args.family / "final", classes)
    B, shift = args.buffer, args.read_shift

    # per-class split arrays (cache-index space) once
    splits = {}
    for cls in fam.classes:
        splits[cls] = {sp: fam.class_split_arrays(cls, sp, B, shift)
                       for sp in ("train", "val", "form_holdout")}

    # token-type ids aligned to the cache (Hewitt–Liang control)
    tok_type = np.zeros(fam.total_tokens, dtype=np.int64)
    for eid in fam.example_ids:
        off, n = fam.offsets[eid]
        tok_type[off:off + n] = fam.tokens[eid]

    def hl_labels(cls, t_y_masked):
        """Per-token-type random labels drawn from the CLASS's empirical label
        marginal (mostly zeros), so the control task has the same tie structure
        as the real task — a uniform 0..6 control is denser and incomparably
        easier for rank metrics."""
        rng = np.random.default_rng(stable_seed("hewitt-liang", cls))
        marginal = np.sort(t_y_masked)
        return marginal[rng.integers(0, len(marginal), size=257_000)].astype(np.float32)

    results = {cls: {} for cls in fam.classes}
    val_pred_store = {cls: {} for cls in fam.classes}   # layer -> adam preds on val
    for L in layers:
        acts = torch.from_numpy(
            np.load(Path(args.cache) / f"acts_l{L}.npy", mmap_mode="r")[:]).to(device)
        mu_np, sd_np = load_natstats(Path(args.natstats), L)
        mu = torch.from_numpy(mu_np).to(device)
        sd = torch.from_numpy(sd_np).to(device)
        sc = LayerScorer(Path(args.probes) / f"probes_l{L}.npz", acts, mu, sd, device)

        for ci, cls in enumerate(fam.classes):
            v_idx, v_y, v_m, v_ex, v_eids, v_roles = splits[cls]["val"]
            keep = v_m > 0
            preds = sc.preds(ci, v_idx)
            r = {"ridge_lambda_idx": preds.pop("ridge_lambda_idx", None)}
            ybin = (v_y[keep] >= args.binarize_at)
            for name, p in preds.items():
                pk = p[keep]
                r.setdefault("rho", {})[name] = spearman(pk, v_y[keep])
                if 0 < ybin.sum() < ybin.size:
                    r.setdefault("auroc", {})[name] = auroc(pk[ybin], pk[~ybin])
                    r.setdefault("auprc", {})[name] = auprc(ybin.astype(float), pk)
            r["rho_seed_std"] = float(np.std([r["rho"][f"adam_s{s}"] for s in range(3)]))
            val_pred_store[cls][L] = (preds[PRIMARY], v_y, v_m, v_ex)

            # ---- example-level operating point (adam candidate)
            n_ex = len(v_eids)
            ex_score = example_max(preds[PRIMARY], v_ex, n_ex)
            ex_ymax = example_max(v_y, v_ex, n_ex)
            roles = np.array(v_roles)
            neutral = roles == "neutral"
            if neutral.sum() >= 20:
                tau = float(np.quantile(ex_score[neutral], NEUTRAL_Q))
                imp = (roles == "implicit_pos") & (ex_ymax >= POS_THRESH)
                hard = (roles == "hard_neg") & (ex_ymax < POS_THRESH)
                r["tau"] = tau
                r["implicit_recall"] = float((ex_score[imp] > tau).mean()) if imp.sum() >= 10 else None
                # Homograph FPR: judge-truth gives wrong-sense hard negatives a
                # DELIBERATE faint-echo label (~1/6, the 2B rule), so a probe
                # scoring them just above the neutral floor is CORRECT, not a
                # false positive. A false positive = a hard negative scoring in
                # the confirmed-positive range → threshold at the 25th pct of
                # judged-positive example scores. Neutral-floor FPR kept as a
                # diagnostic.
                pos_conf = np.isin(roles, ("target_pos", "implicit_pos")) & (ex_ymax >= POS_THRESH)
                if pos_conf.sum() >= 20 and hard.sum() >= 10:
                    tau_strong = float(np.quantile(ex_score[pos_conf], 0.25))
                    r["tau_strong"] = tau_strong
                    r["homograph_fpr"] = float((ex_score[hard] > tau_strong).mean())
                    r["homograph_fpr_neutral_tau"] = float((ex_score[hard] > tau).mean())
                else:
                    r["homograph_fpr"] = None
                r["n_implicit"], r["n_hard"] = int(imp.sum()), int(hard.sum())
                # G-ratio numerator/denominator (detection vs the same neutrals)
                expl = np.isin(roles, ("target_pos", "explicit")) & (ex_ymax >= POS_THRESH)
                den = auroc(ex_score[expl], ex_score[neutral])
                f_idx, f_y, f_m, f_ex, f_eids, f_roles = splits[cls]["form_holdout"]
                n_form_pos = 0
                if len(f_eids) >= 10:
                    f_ymax = example_max(f_y, f_ex, len(f_eids))
                    n_form_pos = int((f_ymax >= POS_THRESH).sum())
                if n_form_pos >= 10:
                    f_pred = sc.preds(ci, f_idx)[PRIMARY]
                    f_score = example_max(f_pred, f_ex, len(f_eids))
                    num = auroc(f_score[f_ymax >= POS_THRESH], ex_score[neutral])
                    r["g_source"] = "form_holdout"
                else:
                    # single-form classes / hazard-colliding surfaces (e.g. "Jan"
                    # the name): §6.2 deviation-#2 fallback to the implicit slice
                    num = auroc(ex_score[imp], ex_score[neutral])
                    r["g_source"] = "implicit_fallback"
                r["n_form_pos"] = n_form_pos
                r["g_ratio"] = (num / den) if (den and not np.isnan(den) and den > 0) else None
                r["g_num_auroc"], r["g_den_auroc"] = num, den

            # ---- controls
            t_idx, t_y, t_m, *_ = splits[cls]["train"]
            tkeep = t_m > 0
            if "chosen_lambda_ridge" in sc.z:
                li = int(sc.z["chosen_lambda_ridge"][ci])
            else:
                li = int(np.bincount(sc.z["chosen_lambda_idx"][:, ci]).argmax())
            lam = float(sc.z["lambdas"][li])
            rng = np.random.default_rng(stable_seed("shuffle", cls, L))
            y_shuf = t_y[tkeep].copy(); rng.shuffle(y_shuf)
            w_s, b_s = ridge_refit(acts, mu, sd, t_idx[tkeep], y_shuf, lam, device)
            p_s = sc._project(v_idx[keep], w_s, b_s)
            r["shuffled_rho"] = spearman(p_s, v_y[keep])
            hl_map = hl_labels(cls, t_y[tkeep])
            y_hl_tr = hl_map[tok_type[t_idx[tkeep]]]
            w_h, b_h = ridge_refit(acts, mu, sd, t_idx[tkeep], y_hl_tr, lam, device)
            p_h = sc._project(v_idx[keep], w_h, b_h)
            r["hl_control_rho"] = spearman(p_h, hl_map[tok_type[v_idx[keep]]])
            r["hl_selectivity"] = (r["rho"][PRIMARY] - r["hl_control_rho"]
                                   if not np.isnan(r["hl_control_rho"]) else None)

            # ---- random directions
            rp = sc.rand_preds(v_idx[keep])
            rand_rhos = np.array([spearman(rp[k], v_y[keep]) for k in range(rp.shape[0])])
            r["rand_rho_q95"] = float(np.nanquantile(np.abs(rand_rhos), 0.95))
            r["rand_rho_mean"] = float(np.nanmean(rand_rhos))
            r["rand_margin"] = r["rho"][PRIMARY] - r["rand_rho_q95"]

            # ---- bootstrap CI on adam ρ (example-level resampling)
            boots = []
            rngb = np.random.default_rng(stable_seed("boot", cls, L))
            pk, yk, exk = preds[PRIMARY][keep], v_y[keep], v_ex[keep]
            order = np.argsort(exk, kind="stable")
            bounds = np.searchsorted(exk[order], np.arange(n_ex + 1))
            groups = [order[bounds[i]:bounds[i + 1]] for i in range(n_ex)]
            for _ in range(args.n_boot):
                pick = rngb.integers(0, n_ex, n_ex)
                sel = np.concatenate([groups[i] for i in pick]) if n_ex else np.array([], int)
                boots.append(spearman(pk[sel], yk[sel]))
            r["rho_ci95"] = [float(np.nanquantile(boots, 0.025)),
                             float(np.nanquantile(boots, 0.975))]
            r["chosen_lambda"] = lam
            results[cls][str(L)] = r
        del acts, sc
        torch.cuda.empty_cache()
        print(f"[evaluate] layer {L} done", flush=True)

    # ---- §5.3 ensemble per class
    for cls in fam.classes:
        store = val_pred_store[cls]
        Ls = sorted(store)
        p0, y0, m0, ex0 = store[Ls[0]]
        keep = m0 > 0
        P = np.stack([store[L][0][keep] for L in Ls])       # [12, V]
        y = y0[keep]
        resid = P - y[None, :]
        cc = np.corrcoef(resid)
        iu = np.triu_indices(len(Ls), 1)
        mean_rc = float(np.nanmean(cc[iu]))
        ens = {"mean_resid_corr": mean_rc, "skipped": mean_rc > 0.9}
        if not ens["skipped"]:
            X = P.T
            G = X.T @ X + 1e-3 * X.shape[0] * np.eye(len(Ls))
            alpha = np.linalg.solve(G, X.T @ y)
            p_ens = X @ alpha
            rho_ens = spearman(p_ens, y)
            best = max(results[cls].values(), key=lambda r: np.nan_to_num(r["rho"][PRIMARY], nan=-9))
            ens.update(alpha=alpha.tolist(), rho_val=rho_ens,
                       best_layer_rho=best["rho"][PRIMARY],
                       adopt_candidate=bool(rho_ens >= best["rho"][PRIMARY] + 0.03))
        results[cls]["ensemble"] = ens

    return {"family": args.family, "read_shift": shift, "buffer": B,
            "pos_thresh": POS_THRESH, "classes": fam.classes, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--stage4", default="concept_probes/stage4/data")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--natstats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="1,3,6,8,10,12,14,16,18,20,23,25")
    ap.add_argument("--classes")
    ap.add_argument("--read-shift", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=10)
    ap.add_argument("--binarize-at", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    res = evaluate_family(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[evaluate] wrote {args.out}")


if __name__ == "__main__":
    main()
