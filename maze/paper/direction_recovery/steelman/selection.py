"""Selection functions of the steelman rerun. Pure python on the metric rows, so the same code runs on the pod (to pick
the directions used in the interventions) and locally (tables, figures, selection.json).

`fair`      uses labels on validation mazes only, never W_inj  -> what an interpretability user could actually do.
`steelman`  uses the true graft directions for every choice     -> an UPPER BOUND on what the method family can reach.

SAE direction families (one direction per free connection):
  fair       the single latent with the largest |validation label correlation|              (the original protocol)
  oracle     the single live latent with the largest true |cosine| to that graft row         (uses W_inj)
  multi{m}   decoder image of the open-minus-wall mean code restricted to the m latents with the largest
             |label correlation| (m = 2, 4, 16, all): a supervised multi-latent edit; its m is an oracle choice under `steelman`
"""
FAMILIES = ("fair", "oracle", "multi2", "multi4", "multi16", "multiall")
GAUGE = "id"            # selection metric: mean |cosine| to the graft rows in the identifiable (zero-mean) gauge


def select_sae(rows, criterion):
    """rows: per-SAE metric dicts of one model and site -> (job id, family, info)."""
    if criterion == "fair":
        cand = [r for r in rows if r["variant"] in ("ctr", "raw")]
        best = max(cand, key=lambda r: (r["fair_corr"], -r["n_lat"])); fam = "fair"
    else:
        best, fam = max(((r, f) for r in rows for f in FAMILIES), key=lambda rf: rf[0][f"{rf[1]}_{GAUGE}"])
    info = dict(criterion=criterion, job=best["job"], family=fam, n_candidates=len(rows), cos_id=best[f"{fam}_id"], cos_raw=best[f"{fam}_raw"],
                fair_corr=best["fair_corr"], fvu=best["fvu"], L0=best["L0"], dead_frac=best["dead_frac"], n_lat=best["n_lat"])
    return best["job"], fam, info


def select_probe(rows, site, criterion, families=("difference_means", "ridge", "logistic")):
    """rows: probe metric dicts of one model -> (method name at the largest fitting size, info)."""
    n_max = max(r["n_fit"] for r in rows)
    cand = [r for r in rows if r["site"] == site and r["n_fit"] == n_max and r["method"].split("_")[0] in [f.split("_")[0] for f in families] and "cos_id" in r]
    # ("pinv" = joint pseudo-inverse of a readout matrix; it is only selected when asked for explicitly, never as a per-feature probe)
    if criterion == "fair":          # best held-out readout accuracy; ties (1e-4) go to the stronger regulariser
        cand = [r for r in cand if r["method"].startswith(("ridge", "logistic"))]
        top = max(r["val_acc"] for r in cand); tied = [r for r in cand if r["val_acc"] >= top - 1e-4]
        best = max(tied, key=lambda r: float(r["method"].split("_")[1]))
    else:
        best = max(cand, key=lambda r: r[f"cos_{GAUGE}"])
    return best["method"], dict(criterion=criterion, method=best["method"], n_fit=n_max, cos_id=best["cos_id"], cos_raw=best["cos_raw"], val_acc=best["val_acc"], n_candidates=len(cand))
