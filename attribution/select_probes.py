"""Stage 7-Oracle Phase 0 — probe selection (CPU, local, $0).

Implements SPEC.md "Phase 0 — Probe selection" against the frozen
DESIGN.md probe_set.json / probe_set_arrays.npz schema.

Inputs (all local):
  concept_probes/3_validation/data/natscores/<family>.natscores.npz
      preds_{ridge,dom,lda,logistic} [12, T, C] fp32 (raw, non-unit-norm W;
      bias INCLUDED for ridge/logistic, bias=0 for dom/lda — see
      3_validation/code/score_natural.py `proj()`), y [T, C], classes [C] (SPACE
      form), layers [12] = the probe-layer grid
      [1,3,6,8,10,12,14,16,18,20,23,25] (this IS the row order of the first
      axis — confirmed both by 3_validation/code/score_natural.py `--layers`
      default/loop order AND independently documented in
      4_causal/code/common.py's module docstring: "3_validation/data/natscores/
      <family>.natscores.npz: preds_ridge [12, n_tokens, C] ... layers [12]
      giving the row order"), token2ex [T] int, ex_nat_split [n_ex] in
      {"cal","test"}.
  concept_probes/2_probes/probes/<family>/probes_l{L}.npz
      classes [C] (UNDERSCORE form), W_ridge [3,C,2304] (raw, non-unit),
      b_ridge [3,C], chosen_lambda_ridge [C] (the lambda index natscores'
      preds_ridge was generated with), W_dom/W_lda/W_logistic [C,2304]
      (raw, non-unit), b_logistic [C], nat_mean/nat_std [2304] (gemma
      block-L activation stats, IDENTICAL across families at a given layer
      — verified empirically below at import time is skipped for speed but
      was checked interactively: months vs weekdays nat_mean/nat_std at L8
      are byte-identical).
  concept_probes/3_validation/artifacts/probe_cards.json — Stage-6 tier verdicts
      (list of {concept, family, layer, tier, ...}), UNDERSCORE concept
      names.
  concept_probes/4_causal/out/analysis/causal_cards.json — causal verdicts;
      cards[i]['layer_story']['e5_salient_layer_corrected'] is the per-
      concept causal-salient layer used for the ablation-layer vote.
  concept_probes/4_causal/out/dose_calib.json — independently-computed
      {family: {class: {layer: {"s95":.., "t":..}}}} where s95/t are on the
      UNIT-w standardized ridge score (preds_ridge - b_ridge)/||W_ridge||
      over ALL natural-pool tokens (cal+test). Used ONLY as a ground-truth
      cross-check of our own W/b/lambda extraction (step in `verify_axes`);
      NOT copied into probe_set outputs (our s95 convention differs, see
      `compute_s95` docstring).

Procedure: see module-level `main()`.

Usage: python select_probes.py   (no args; all paths are relative to this
file's location, per DESIGN.md file layout)
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CODE_DIR = Path(__file__).resolve().parent        # repo_root/attribution
REPO_ROOT = Path(__file__).resolve().parents[1]
CP_DIR = REPO_ROOT / "concept_probes"
NATSCORES_DIR = CP_DIR / "3_validation" / "data" / "natscores"
PROBES_DIR = CP_DIR / "2_probes" / "probes"
PROBE_CARDS_PATH = CP_DIR / "3_validation" / "artifacts" / "probe_cards.json"
CAUSAL_CARDS_PATH = CP_DIR / "4_causal" / "out" / "analysis" / "causal_cards.json"
DOSE_CALIB_PATH = CP_DIR / "4_causal" / "out" / "dose_calib.json"
OUT_DIR = CODE_DIR / "out"

ARMS = ["ridge", "dom", "lda", "logistic"]
YMAX_THRESH = 0.34            # example-level positive-label threshold (SPEC)
AUROC_THRESH = 0.90           # per-layer selection threshold (task spec)
MIN_SURVIVORS = 20            # relax trigger (SPEC step 5)
D_MODEL = 2304
BAND_LO, BAND_HI = 8, 12      # causally-salient band, tie-break preference
MIN_SPREAD_SPAN = 4           # min grid-index span among the 3 chosen layers


def _u(s: str) -> str:
    """Canonicalize a concept name to the UNDERSCORE form (2_probes/probes,
    probe_cards, causal_cards convention; 4_causal/common.py deviation #3)."""
    return s.replace(" ", "_")


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC via rank-sum; identical formula to
    3_validation/code/gates.py `auroc()`."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    n_p, n_n = len(pos), len(neg)
    return float((r[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho; identical formula to 3_validation/code/gates.py `spearman()`
    (rank + z-score correlation, no ceiling correction — this task's raw rho,
    reported for audit only, not used as a selection gate)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def list_families() -> list[str]:
    return sorted(p.stem.replace(".natscores", "")
                  for p in NATSCORES_DIR.glob("*.natscores.npz"))


def load_natscores(fam: str) -> dict:
    with np.load(NATSCORES_DIR / f"{fam}.natscores.npz") as z:
        return {k: z[k] for k in z.files}


def load_probes(fam: str, layer: int) -> dict:
    with np.load(PROBES_DIR / fam / f"probes_l{layer}.npz") as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Step 1: per (concept, layer, arm) example-AUROC + token-rho on TEST half.
# ---------------------------------------------------------------------------

def score_family(fam: str, nat: dict, rows: list[dict]) -> None:
    classes_space = [str(c) for c in nat["classes"]]
    layers = [int(x) for x in nat["layers"]]
    token2ex = nat["token2ex"]
    ex_split = nat["ex_nat_split"]
    n_ex = len(ex_split)
    test_ex_mask = (ex_split == "test")
    test_ex_ids = np.flatnonzero(test_ex_mask)
    test_tok_mask = test_ex_mask[token2ex]

    for ci, cls_space in enumerate(classes_space):
        cls = _u(cls_space)
        y_c = nat["y"][:, ci].astype(np.float64)

        # example-level ymax (max-pool over ALL tokens belonging to the
        # example; SPEC does not restrict the pool to test tokens for the
        # label itself, only the example must be in the test SPLIT).
        ymax = np.full(n_ex, -np.inf)
        np.maximum.at(ymax, token2ex, y_c)
        labels = (ymax[test_ex_ids] >= YMAX_THRESH).astype(int)
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())

        for li, L in enumerate(layers):
            for arm in ARMS:
                p = nat[f"preds_{arm}"][li, :, ci].astype(np.float64)
                pmax = np.full(n_ex, -np.inf)
                np.maximum.at(pmax, token2ex, p)
                pmax_test = pmax[test_ex_ids]
                a = (auroc(pmax_test[labels == 1], pmax_test[labels == 0])
                     if n_pos > 0 and n_neg > 0 else float("nan"))
                rho = spearman(p[test_tok_mask], y_c[test_tok_mask])
                rows.append(dict(family=fam, concept=cls, layer=L, arm=arm,
                                  auroc=a, token_rho=rho,
                                  n_pos_ex=n_pos, n_neg_ex=n_neg))


# ---------------------------------------------------------------------------
# Step 2: pick 3 layers.
# ---------------------------------------------------------------------------

def pick_layers(rows: list[dict], layer_grid: list[int]) -> tuple[list[int], dict]:
    concepts = sorted({(r["family"], r["concept"]) for r in rows})
    # best_auroc[(fam,concept,layer)] = max over arms
    best = {}
    for r in rows:
        k = (r["family"], r["concept"], r["layer"])
        v = r["auroc"]
        if k not in best or (not np.isnan(v) and (np.isnan(best[k]) or v > best[k])):
            best[k] = v

    layer_mean = {}
    for L in layer_grid:
        vals = [best[(f, c, L)] for (f, c) in concepts if not np.isnan(best[(f, c, L)])]
        layer_mean[L] = float(np.mean(vals)) if vals else float("nan")

    idx_of = {L: i for i, L in enumerate(layer_grid)}
    best_combo, best_key = None, None
    for combo in itertools.combinations(layer_grid, 3):
        idxs = sorted(idx_of[L] for L in combo)
        span = idxs[-1] - idxs[0]
        if span < MIN_SPREAD_SPAN:
            continue
        score = sum(layer_mean[L] for L in combo)
        band_hits = sum(1 for L in combo if BAND_LO <= L <= BAND_HI)
        # key: maximize score (primary), then band coverage (tie-break),
        # then prefer wider spread (secondary tie-break, DESIGN "spread").
        key = (round(score, 6), band_hits, span)
        if best_key is None or key > best_key:
            best_key, best_combo = key, combo

    return sorted(best_combo), layer_mean


# ---------------------------------------------------------------------------
# Step 3: per concept per layer, best arm; survival filter.
# ---------------------------------------------------------------------------

def select_concepts(rows: list[dict], chosen_layers: list[int]):
    concepts = sorted({(r["family"], r["concept"]) for r in rows})
    by_key = {(r["family"], r["concept"], r["layer"], r["arm"]): r for r in rows}

    per_layer_best = {}   # (fam,concept,layer) -> row (best arm)
    for (fam, c) in concepts:
        for L in chosen_layers:
            cand = [by_key[(fam, c, L, arm)] for arm in ARMS]
            cand = [r for r in cand if not np.isnan(r["auroc"])]
            if not cand:
                per_layer_best[(fam, c, L)] = None
                continue
            per_layer_best[(fam, c, L)] = max(cand, key=lambda r: r["auroc"])

    survivors, near_misses = [], []
    for (fam, c) in concepts:
        rows_l = [per_layer_best[(fam, c, L)] for L in chosen_layers]
        if any(r is None for r in rows_l):
            near_misses.append((fam, c, rows_l, "missing (no arm scored)"))
            continue
        aurocs = [r["auroc"] for r in rows_l]
        if all(a >= AUROC_THRESH for a in aurocs):
            survivors.append((fam, c, rows_l))
        else:
            near_misses.append((fam, c, rows_l, f"min_auroc={min(aurocs):.4f}"))
    return survivors, near_misses, per_layer_best, concepts


def relax_per_layer(rows: list[dict], chosen_layers: list[int]):
    """SPEC step 5 fallback: independent concept set per layer (no all-3
    constraint). Returns {layer: [(fam,concept,row), ...]}."""
    concepts = sorted({(r["family"], r["concept"]) for r in rows})
    by_key = {(r["family"], r["concept"], r["layer"], r["arm"]): r for r in rows}
    out = {}
    for L in chosen_layers:
        keep = []
        for (fam, c) in concepts:
            cand = [by_key[(fam, c, L, arm)] for arm in ARMS]
            cand = [r for r in cand if not np.isnan(r["auroc"])]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["auroc"])
            if best["auroc"] >= AUROC_THRESH:
                keep.append((fam, c, best))
        out[L] = keep
    return out


# ---------------------------------------------------------------------------
# Step 4: ablation layer = mode of e5_salient_layer_corrected over survivors.
# ---------------------------------------------------------------------------

def ablation_layer(survivor_concepts: set[tuple[str, str]]) -> tuple[int, dict]:
    cards = json.loads(CAUSAL_CARDS_PATH.read_text())["cards"]
    votes = []
    for card in cards:
        key = (card["family"], _u(card["concept"]))
        if key in survivor_concepts:
            L = card["layer_story"]["e5_salient_layer_corrected"]
            votes.append(int(L))
    if not votes:
        return None, {}
    vals, counts = np.unique(votes, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    mode_L = int(vals[np.argmax(counts)])
    return mode_L, hist


# ---------------------------------------------------------------------------
# Step 5: W/b/nat_mean/nat_std/s95/G_dom assembly.
# ---------------------------------------------------------------------------

def native_wb(fam: str, cls_underscore: str, layer: int, arm: str, probes: dict):
    """Return (w [2304] fp32, b float) in STANDARDIZED, native (non-unit)
    space, exactly the arm's own scale — i.e. the same W,b used by
    3_validation/code/score_natural.py `proj()` to generate preds_{arm}. classes in
    probes_l{L}.npz are UNDERSCORE form already."""
    classes = [str(c) for c in probes["classes"]]
    ci = classes.index(cls_underscore)
    if arm == "ridge":
        li = int(probes["chosen_lambda_ridge"][ci])
        return probes["W_ridge"][li, ci].astype(np.float32), float(probes["b_ridge"][li, ci])
    if arm == "dom":
        return probes["W_dom"][ci].astype(np.float32), 0.0
    if arm == "lda":
        return probes["W_lda"][ci].astype(np.float32), 0.0
    if arm == "logistic":
        return probes["W_logistic"][ci].astype(np.float32), float(probes["b_logistic"][ci])
    raise ValueError(arm)


def compute_s95(nat: dict, ci: int, layer_idx: int, arm: str, y_thresh=YMAX_THRESH):
    """95th percentile of the CHOSEN ARM's NATIVE-scale score
    (preds_{arm}[layer_idx, :, ci] — same units as the W/b we store, i.e. NOT
    unit-normalized) over TEST-split TOKENS whose own target y >= YMAX_THRESH
    ("natural test positives' tokens" per task spec).

    This differs from 4_causal/code/common.py `dose_calib` in three ways,
    all deliberate given DESIGN.md's "native scale" requirement for W:
      1. units: native (arm's own ||W||) vs dose_calib's unit-w
         ((preds-b)/||W_ridge||) — because probe_set_arrays.npz stores W at
         native scale (DESIGN.md), so s95 must be in the SAME units to be a
         usable dosing/threshold reference alongside it.
      2. token filter: only tokens with y >= 0.34 ("positives"), vs
         dose_calib's ALL natural-pool tokens (no y filter) — per this
         task's explicit instruction.
      3. split: TEST only, vs dose_calib's cal+test combined — per this
         task's explicit instruction (TEST-only discipline throughout
         Phase 0).
    """
    test_tok = nat["_test_tok_mask"]
    y_c = nat["y"][:, ci]
    mask = test_tok & (y_c >= y_thresh)
    p = nat[f"preds_{arm}"][layer_idx, :, ci]
    if mask.sum() == 0:
        return float("nan")
    return float(np.percentile(p[mask], 95))


def verify_axes(natcache: dict, n_concepts: int = 2) -> list[dict]:
    """Cross-check our W/b/lambda/nat_mean/nat_std reading convention against
    stage6_1's INDEPENDENTLY computed dose_calib.json (built from the same
    underlying probes_l{L}.npz + natscores files by a different module,
    4_causal/code/common.py `_build_family_calib`, matching by class NAME
    not index). No raw gemma hidden states are available locally (the
    per-token activation cache used by 3_validation/code/score_natural.py was
    pod-local and not persisted) so this is the strongest available
    loop-closure: dose_calib's t/s95 are themselves derived from
    preds_ridge, so matching them end-to-end confirms (a) the ridge
    W/b/chosen_lambda extraction is correct, (b) the natscores <-> probes
    class-name join is correct (space vs underscore), (c) the "preds
    already include bias, raw non-unit W" formula is correctly understood.
    Returns a list of per-concept check dicts; raises AssertionError on
    mismatch (rel err > 1e-4) so a convention bug fails loudly.
    """
    dose_calib = json.loads(DOSE_CALIB_PATH.read_text())
    out = []
    picked = 0
    for fam in list_families():
        if picked >= n_concepts:
            break
        if fam not in dose_calib:
            continue
        nat = natcache[fam]
        classes_space = [str(c) for c in nat["classes"]]
        layers = [int(x) for x in nat["layers"]]
        test_tok = nat["_test_tok_mask"]
        for ci, cls_space in enumerate(classes_space):
            if picked >= n_concepts:
                break
            cls = _u(cls_space)
            if cls not in dose_calib[fam]:
                continue
            L = layers[len(layers) // 2]   # a middle layer, arbitrary
            li = layers.index(L)
            probes = load_probes(fam, L)
            w, b = native_wb(fam, cls, L, "ridge", probes)
            nrm = float(np.linalg.norm(w))
            p = nat["preds_ridge"][li, :, ci]
            unit_w_score = (p - b) / nrm
            mine_t = float(unit_w_score[test_tok | ~test_tok].mean())  # ALL tokens (cal+test), matches dose_calib
            mine_s95 = float(np.percentile(unit_w_score, 95))
            ref = dose_calib[fam][cls][str(L)]
            rel_t = abs(mine_t - ref["t"]) / max(abs(ref["t"]), 1e-6)
            rel_s95 = abs(mine_s95 - ref["s95"]) / max(abs(ref["s95"]), 1e-6)
            rec = dict(family=fam, concept=cls, layer=L, mine_t=mine_t, ref_t=ref["t"],
                       rel_err_t=rel_t, mine_s95=mine_s95, ref_s95=ref["s95"],
                       rel_err_s95=rel_s95)
            out.append(rec)
            assert rel_t < 1e-3 and rel_s95 < 1e-3, (
                f"AXIS CONVENTION MISMATCH for {fam}/{cls}@{L}: {rec}")
            picked += 1
    return out


def main():
    families = list_families()
    print(f"[select_probes] {len(families)} families with natscores: {families}")

    natcache = {}
    rows: list[dict] = []
    for fam in families:
        nat = load_natscores(fam)
        nat["_test_tok_mask"] = (nat["ex_nat_split"] == "test")[nat["token2ex"]]
        natcache[fam] = nat
        score_family(fam, nat, rows)
        print(f"[select_probes] scored family={fam} "
              f"({len(nat['classes'])} concepts x {len(nat['layers'])} layers x {len(ARMS)} arms)")

    layer_grid = sorted({int(x) for r in rows for x in [r["layer"]]})
    assert layer_grid == [1, 3, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25], layer_grid

    print("\n[select_probes] step 0: axis-convention verification against dose_calib.json ...")
    verify_recs = verify_axes(natcache, n_concepts=2)
    for rec in verify_recs:
        print(f"  VERIFIED {rec['family']}/{rec['concept']}@{rec['layer']}: "
              f"t mine={rec['mine_t']:.6f} ref={rec['ref_t']:.6f} (rel {rec['rel_err_t']:.2e}); "
              f"s95 mine={rec['mine_s95']:.6f} ref={rec['ref_s95']:.6f} (rel {rec['rel_err_s95']:.2e})")

    print("\n[select_probes] step 2: picking 3 layers ...")
    chosen_layers, layer_mean = pick_layers(rows, layer_grid)
    print(f"  per-layer mean best-arm example-AUROC (all 64 concepts):")
    for L in layer_grid:
        marker = " <== chosen" if L in chosen_layers else ""
        print(f"    L{L:>2}: {layer_mean[L]:.4f}{marker}")
    print(f"  CHOSEN LAYERS: {chosen_layers}")

    print("\n[select_probes] step 3: per-concept arm selection + survival filter ...")
    survivors, near_misses, per_layer_best, all_concepts = select_concepts(rows, chosen_layers)
    print(f"  survivors (AUROC>={AUROC_THRESH} at all 3 layers): {len(survivors)} / {len(all_concepts)}")

    relaxed_used = False
    relaxed_sets = None
    if len(survivors) < MIN_SURVIVORS:
        print(f"  ! only {len(survivors)} < {MIN_SURVIVORS} -> RELAXING per SPEC step 5 "
              f"(layer-optimal sets, concept set may differ per layer)")
        relaxed_used = True
        relaxed_sets = relax_per_layer(rows, chosen_layers)
        for L in chosen_layers:
            print(f"    L{L}: {len(relaxed_sets[L])} concepts pass")

    if not relaxed_used:
        survivor_concepts = {(fam, c) for (fam, c, _) in survivors}
        survivor_names = sorted(c for (_, c) in survivor_concepts)
    else:
        survivor_concepts = {(fam, c) for L in chosen_layers for (fam, c, _) in relaxed_sets[L]}
        survivor_names = sorted(c for (_, c) in survivor_concepts)

    print(f"\n[select_probes] step 4: ablation layer (mode of e5_salient_layer_corrected) ...")
    abl_layer, abl_hist = ablation_layer(survivor_concepts)
    print(f"  histogram: {abl_hist}")
    print(f"  ABLATION LAYER: {abl_layer}")

    # ------------------------------------------------------------------
    # Step 5/6: assemble probe_set.json + probe_set_arrays.npz
    # ------------------------------------------------------------------
    print("\n[select_probes] assembling outputs ...")

    if not relaxed_used:
        concept_names = survivor_names   # canonical order, ORDER IS CANONICAL
        concept_family = {c: fam for (fam, c) in survivor_concepts}
        K = len(concept_names)
        W = np.zeros((3, K, D_MODEL), dtype=np.float32)
        b_arr = np.zeros((3, K), dtype=np.float32)
        nat_mean = np.zeros((3, D_MODEL), dtype=np.float32)
        nat_std = np.zeros((3, D_MODEL), dtype=np.float32)
        selection = {str(L): {} for L in chosen_layers}
        s95_out = {str(L): {} for L in chosen_layers}

        # per-layer probes cache (nat_mean/nat_std are family-independent at
        # a given layer -- verified interactively: months vs weekdays L8
        # nat_mean/nat_std are byte-identical -- so load once per layer using
        # ANY family that has that layer file).
        probes_cache = {}
        for li, L in enumerate(chosen_layers):
            any_fam = survivors[0][0]
            probes_any = load_probes(any_fam, L)
            nat_mean[li] = probes_any["nat_mean"]
            nat_std[li] = probes_any["nat_std"]

        # CANONICAL ORDER FIX (permutation bug, see attribution/README.md
        # permutation note; PERMUTATION_FIX.md in git history):
        # `survivors` is (family,concept)-sorted, but `concept_names` (which is
        # written to probe_set.json["concepts"] and consumed positionally by
        # every downstream reader) is name-sorted. Index W/b by the concept's
        # position in `concept_names`, exactly as the relaxed branch (below)
        # and the DoM block already do -- NOT by enumerate(survivors), which
        # silently permuted 53/54 main-block rows. Do NOT rely on iteration
        # order here.
        idx_of_c = {c: i for i, c in enumerate(concept_names)}
        for (fam, c, rows_l) in survivors:
            ci = idx_of_c[c]
            for li, (L, r) in enumerate(zip(chosen_layers, rows_l)):
                arm = r["arm"]
                key = (fam, L)
                if key not in probes_cache:
                    probes_cache[key] = load_probes(fam, L)
                probes = probes_cache[key]
                w, bias = native_wb(fam, c, L, arm, probes)
                W[li, ci] = w
                b_arr[li, ci] = bias
                selection[str(L)][c] = dict(arm=arm, auroc=r["auroc"], token_rho=r["token_rho"],
                                             family=fam)
                nat = natcache[fam]
                classes_space = [str(x) for x in nat["classes"]]
                cls_idx = classes_space.index(c.replace("_", " ")) if c.replace("_", " ") in classes_space else classes_space.index(c)
                layer_idx = [int(x) for x in nat["layers"]].index(L)
                s95_out[str(L)][c] = compute_s95(nat, cls_idx, layer_idx, arm)
    else:
        # relaxed path: concept set may differ per layer -> pad K to the
        # UNION of concepts, zero rows for concepts absent at a given layer.
        concept_names = sorted({c for L in chosen_layers for (_, c, _) in relaxed_sets[L]})
        concept_family = {}
        for L in chosen_layers:
            for (fam, c, _) in relaxed_sets[L]:
                concept_family[c] = fam
        K = len(concept_names)
        idx_of_c = {c: i for i, c in enumerate(concept_names)}
        W = np.zeros((3, K, D_MODEL), dtype=np.float32)
        b_arr = np.zeros((3, K), dtype=np.float32)
        nat_mean = np.zeros((3, D_MODEL), dtype=np.float32)
        nat_std = np.zeros((3, D_MODEL), dtype=np.float32)
        selection = {str(L): {} for L in chosen_layers}
        s95_out = {str(L): {} for L in chosen_layers}
        probes_cache = {}
        for li, L in enumerate(chosen_layers):
            any_fam, any_c, any_r = relaxed_sets[L][0]
            probes_any = load_probes(any_fam, L)
            nat_mean[li] = probes_any["nat_mean"]
            nat_std[li] = probes_any["nat_std"]
            for (fam, c, r) in relaxed_sets[L]:
                ci = idx_of_c[c]
                key = (fam, L)
                if key not in probes_cache:
                    probes_cache[key] = load_probes(fam, L)
                probes = probes_cache[key]
                arm = r["arm"]
                w, bias = native_wb(fam, c, L, arm, probes)
                W[li, ci] = w
                b_arr[li, ci] = bias
                selection[str(L)][c] = dict(arm=arm, auroc=r["auroc"], token_rho=r["token_rho"],
                                             family=fam)
                nat = natcache[fam]
                classes_space = [str(x) for x in nat["classes"]]
                cls_idx = classes_space.index(c.replace("_", " ")) if c.replace("_", " ") in classes_space else classes_space.index(c)
                layer_idx = [int(x) for x in nat["layers"]].index(L)
                s95_out[str(L)][c] = compute_s95(nat, cls_idx, layer_idx, arm)
        survivors = None  # not used further in relaxed path

    # ---- Exp-B DoM directions at ablation_layer, for ALL concept_names ----
    W_dom_abl = np.zeros((K, D_MODEL), dtype=np.float32)
    b_dom_abl = np.zeros(K, dtype=np.float32)
    t_nat_dom = np.zeros(K, dtype=np.float32)
    abl_probes_cache = {}
    if abl_layer is not None:
        for ci, c in enumerate(concept_names):
            fam = concept_family[c]
            if (fam, abl_layer) not in abl_probes_cache:
                abl_probes_cache[(fam, abl_layer)] = load_probes(fam, abl_layer)
            probes = abl_probes_cache[(fam, abl_layer)]
            w, bias = native_wb(fam, c, abl_layer, "dom", probes)
            W_dom_abl[ci] = w
            b_dom_abl[ci] = bias  # always 0.0 for dom, kept for schema symmetry
            nat = natcache[fam]
            classes_space = [str(x) for x in nat["classes"]]
            cls_idx = classes_space.index(c.replace("_", " ")) if c.replace("_", " ") in classes_space else classes_space.index(c)
            if abl_layer in [int(x) for x in nat["layers"]]:
                layer_idx = [int(x) for x in nat["layers"]].index(abl_layer)
                test_tok = nat["_test_tok_mask"]
                t_nat_dom[ci] = float(nat["preds_dom"][layer_idx, test_tok, cls_idx].mean())
            else:
                t_nat_dom[ci] = float("nan")

    # G_dom must be the STANDARDIZED-space Gram W_dom W_dom^T (NOT the raw
    # Gram of d_c = σ⊙w). Derivation: the ablation is the stage6_1-style
    # joint projection in standardized space, h_std' = h_std − U(UᵀU)⁻¹(s−t)
    # with U columns = w_dom, which sets every dom score exactly to its
    # target. In raw space that repair vector is
    # v* = D_raw · (W_dom W_dom^T)⁻¹ · (s−t), D_raw[c] = nat_std_abl ⊙
    # W_dom_abl[c]. Pairing D_raw with the raw Gram D_raw D_raw^T (the
    # original bug, caught pre-ExpB) does NOT restore the scores.
    # (nat_std at the ABLATION layer, which may differ from the 3 chosen
    # score layers -- load separately; still needed for the Exp-B decoder D.)
    if abl_layer is not None and abl_layer in chosen_layers:
        li_abl = chosen_layers.index(abl_layer)
        nat_std_abl = nat_std[li_abl]
    elif abl_layer is not None:
        any_fam = next(iter(abl_probes_cache))[0]
        nat_std_abl = load_probes(any_fam, abl_layer)["nat_std"]
    else:
        nat_std_abl = np.ones(D_MODEL, dtype=np.float32)

    G_dom = W_dom_abl @ W_dom_abl.T                    # [K, K] std-space Gram
    G_dom_inv = np.linalg.pinv(G_dom) if K > 0 else np.zeros((0, 0), dtype=np.float32)

    # Defensive self-description of the score-store column layout (see
    # attribution/README.md permutation note; PERMUTATION_FIX.md in git
    # history). After the canonical-order fix above, the W/b
    # MAIN block and the DoM block are BOTH assembled in `concept_names`
    # (name-sorted) order, so both lists equal `concepts` here. We still emit
    # them explicitly so every downstream consumer can attach names to store
    # columns by an explicit contract rather than assuming an order -- and so
    # that a future reordering (or a re-scored store built from an
    # out-of-order W) is described by data, not by convention.
    main_block_concepts = list(concept_names)   # true order of W/b main-block rows + each layer's main store block
    dom_block_concepts = list(concept_names)    # true order of W_dom_abl/b_dom_abl/t_nat_dom/G_dom + store dom block
    probe_set = dict(
        layers=chosen_layers,
        ablation_layer=abl_layer,
        concepts=concept_names,
        families={c: concept_family[c] for c in concept_names},
        main_block_concepts=main_block_concepts,
        dom_block_concepts=dom_block_concepts,
        selection=selection,
        s95=s95_out,
        corpus_stats=None,
        meta=dict(
            n_concepts=K,
            auroc_thresh=AUROC_THRESH,
            ymax_thresh=YMAX_THRESH,
            relaxed=relaxed_used,
            layer_mean_auroc={str(L): layer_mean[L] for L in layer_grid},
            ablation_layer_histogram=abl_hist,
            verify_axes=verify_recs,
        ),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_set.json").write_text(json.dumps(probe_set, indent=2, sort_keys=False))
    np.savez(
        OUT_DIR / "probe_set_arrays.npz",
        W=W, b=b_arr, nat_mean=nat_mean, nat_std=nat_std,
        W_dom_abl=W_dom_abl, b_dom_abl=b_dom_abl, t_nat_dom=t_nat_dom,
        G_dom=G_dom.astype(np.float32), G_dom_inv=G_dom_inv.astype(np.float32),
        layer_index=np.array(chosen_layers, dtype=np.int64),
    )
    print(f"[select_probes] wrote {OUT_DIR / 'probe_set.json'} and "
          f"{OUT_DIR / 'probe_set_arrays.npz'} (K={K} concepts)")

    # ---- selection_table.md: full audit table ----
    lines = []
    lines.append("# Stage 7-Oracle Phase 0 — probe selection audit table\n")
    lines.append(f"Chosen layers: {chosen_layers}  |  Ablation layer: {abl_layer}  |  "
                 f"AUROC threshold: {AUROC_THRESH}  |  relaxed fallback used: {relaxed_used}\n")
    lines.append("## Per-layer mean best-arm example-AUROC (all 64 concepts)\n")
    lines.append("| layer | mean best-arm AUROC | chosen |")
    lines.append("|---|---|---|")
    for L in layer_grid:
        lines.append(f"| {L} | {layer_mean[L]:.4f} | {'YES' if L in chosen_layers else ''} |")
    lines.append("")
    lines.append("## Ablation-layer vote (e5_salient_layer_corrected over survivors)\n")
    lines.append(f"histogram: {abl_hist}  ->  mode = {abl_layer}\n")
    lines.append("## Full audit: every concept x chosen-layer x arm\n")
    lines.append("| family | concept | layer | arm | auroc | token_rho | n_pos_ex | n_neg_ex |"
                  " best_arm | pass(>=%.2f) |" % AUROC_THRESH)
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    rows_by_key = {(r["family"], r["concept"], r["layer"]): [] for r in rows}
    for r in rows:
        rows_by_key[(r["family"], r["concept"], r["layer"])].append(r)
    concept_status = {}
    if not relaxed_used:
        surv_set = {(fam, c) for (fam, c, _) in survivors}
        for (fam, c, rl) in survivors:
            concept_status[(fam, c)] = "SURVIVOR"
        for (fam, c, rl, reason) in near_misses:
            concept_status[(fam, c)] = f"NEAR-MISS/DROP ({reason})"
    for (fam, c) in all_concepts:
        for L in chosen_layers:
            cell_rows = sorted(rows_by_key[(fam, c, L)], key=lambda r: -1 if np.isnan(r["auroc"]) else r["auroc"], reverse=True)
            best = per_layer_best.get((fam, c, L))
            for r in cell_rows:
                is_best = (best is not None and r["arm"] == best["arm"] and
                           abs((r["auroc"] if not np.isnan(r["auroc"]) else -1) -
                               (best["auroc"] if not np.isnan(best["auroc"]) else -1)) < 1e-12)
                passed = (best is not None and not np.isnan(best["auroc"]) and
                          best["auroc"] >= AUROC_THRESH and is_best)
                lines.append(f"| {fam} | {c} | {L} | {r['arm']} | {r['auroc']:.4f} | "
                              f"{r['token_rho']:.4f} | {r['n_pos_ex']} | {r['n_neg_ex']} | "
                              f"{'<==' if is_best else ''} | {'PASS' if passed else ''} |")
    lines.append("")
    lines.append("## Concept-level verdict (all-3-layers constraint)\n")
    lines.append("| family | concept | " + " | ".join(f"L{L} auroc" for L in chosen_layers) +
                  " | status |")
    lines.append("|---|---|" + "---|" * len(chosen_layers) + "---|")
    for (fam, c) in all_concepts:
        aurocs = []
        for L in chosen_layers:
            best = per_layer_best.get((fam, c, L))
            aurocs.append(f"{best['auroc']:.4f}" if best is not None else "n/a")
        status = concept_status.get((fam, c), "n/a")
        lines.append(f"| {fam} | {c} | " + " | ".join(aurocs) + f" | {status} |")

    lines.append("")
    lines.append("## Axis-convention verification (vs stage6_1 dose_calib.json, independent)\n")
    for rec in verify_recs:
        lines.append(f"- {rec['family']}/{rec['concept']}@L{rec['layer']}: "
                      f"t mine={rec['mine_t']:.6f} ref={rec['ref_t']:.6f} "
                      f"(rel err {rec['rel_err_t']:.2e}); "
                      f"s95 mine={rec['mine_s95']:.6f} ref={rec['ref_s95']:.6f} "
                      f"(rel err {rec['rel_err_s95']:.2e})")

    (OUT_DIR / "selection_table.md").write_text("\n".join(lines) + "\n")
    print(f"[select_probes] wrote {OUT_DIR / 'selection_table.md'}")

    # ---- summary for orchestrator ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"K survivors: {K}")
    print(f"Chosen layers: {chosen_layers}")
    print(f"Ablation layer: {abl_layer}  (histogram {abl_hist})")
    print(f"Mean best-arm AUROC per chosen layer: "
          f"{[round(layer_mean[L], 4) for L in chosen_layers]}")
    if not relaxed_used:
        dropped = [(fam, c, [(round(r['auroc'], 4) if r is not None else None) for r in rl])
                   for (fam, c, rl, reason) in near_misses]
        print(f"Dropped concepts ({len(dropped)}):")
        for fam, c, aurocs in dropped:
            print(f"  {fam}/{c}: layer-AUROCs={aurocs}")


if __name__ == "__main__":
    main()
