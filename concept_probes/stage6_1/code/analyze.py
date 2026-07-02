"""Stage 6.1 ANALYSIS — aggregate E1/E2/E4/E5 outputs into causal cards.

Spec: knowledge/concept_probes/task.md §6.1.8 (deliverables) + §6.1.0 (frozen
verdict rules). Reads the experiment outputs written by e1_attrib.py,
e2_cloze.py, e2_ppl.py, e4_ablate.py, e5_propagation.py (see those docstrings
for the npz/summary schemas) plus Stage-6 probe_cards.json and the local E0
geometry npz, and writes ONE json: <out>/causal_cards.json.

INPUT DISCOVERY / FLEET MERGE. Fleet runs are family-split: several pods each
write the same file names under their own out/ tree, later rsynced/merged into
concept_probes/stage6_1/out/. This script therefore:
  * accepts multiple --roots (comma list, later roots take precedence);
  * for each experiment script S globs BOTH <root>/S*/summary*.jsonl and
    <root>/S*summary*.jsonl (the pilot pod wrote out/e2_cloze_summary.jsonl at
    the top level as well as out/e2_cloze/summary.jsonl), skipping any path
    containing "smoke" or "dryrun";
  * concatenates rows in (root order, file order, line order) and DEDUPES on
    (concept, family, str(layer), arm, metric, config.get("k")) keeping the
    LAST occurrence (re-runs append to summary.jsonl, so last = newest);
  * finds per-family npz the same way (<root>/S*/<name>.npz), last root wins;
  * degrades gracefully per concept: every card section is null-able and the
    card records which experiments were present (`data_present`) and which
    verdict sub-criteria could not be evaluated (`criteria[arm]["missing"]`).

METRIC DEFINITIONS (all summary-row metrics are documented in the producing
script's docstring; the card-level derived quantities are defined here):

sufficiency (E2):
  best_layer          Per arm: the layer whose ridge-arm cloze_slope passes the
                      sufficiency bar (ci_low > 0 AND anti_steerable_frac
                      <= 0.25) with the largest slope; if no layer passes, the
                      layer with the largest slope value. Intensity concepts
                      (ordinal rows only, no cloze rows): the layer with max
                      ordinal_spearman (ties -> max ordinal_slope).
  cloze_slope         e2_cloze per-template OLS slope of the target-vs-sibling
                      completion-logprob delta over dose factors in [-2, 2],
                      averaged over templates (bootstrap CI over templates).
                      Reported for ridge/dom/rand at the RIDGE best layer.
  anti_steerable_frac fraction of templates with negative per-template slope
                      (Tan et al. anti-steerable examples).
  suppression_slope   same OLS restricted to factors <= 0 (suppression
                      symmetry at negative dose).
  ordinal_spearman    Spearman(dose factor, mean expected ordinal rank of the
                      graded completion set) — intensity axes only.
  ppl                 e2_ppl rel_delta_slope (OLS of relevant-bucket mean
                      token-logprob delta over factors in [-2,2]) and
                      irrel_delta_f2 (irrelevant-bucket delta at factor 2 =
                      the do-no-harm guard), per arm, at the concept's
                      Stage-6 card layer when available (else the first layer
                      e2_ppl ran).

necessity (E4, everywhere-ablation at all probed layers):
  diag_lp_delta       Δ mean logprob of concept-diagnostic next-tokens on
                      natural positives, per arm; rand_mean = mean over the 5
                      separate rand0..rand4 rows; other_mean = mean over the 3
                      other-concept-direction rows (specificity control).
  cloze_acc_delta / cloze_target_lp_delta   Δ class-keyed cloze accuracy /
                      target completion logprob.
  kl_guard            kl_neutral_nats per arm + ratio ridge/rand_mean (the
                      threshold is calibrated on random directions per task).
  restore             e4 selectivity-restore recovery fractions (dom add-back
                      after ridge everywhere-ablation), per steer factor.

specificity (E2 off-target readout, steering factor 2 at the card layer):
  For steer arm A the primary meter is the OPPOSITE fitted vector (ridge steer
  -> dom meter, dom steer -> ridge meter; §6.1.1 meter != intervention rule);
  the same-meter numbers are reported alongside. target_effect = readout delta
  of the steered concept's own probe; max_offtarget_sibling / _nonfamily =
  max |delta| over same-family (excl. self) / other-family readout probes.
  ratio = |target_effect| / max_offtarget_nonfamily.

layer_story (E1 + E5):
  e1_candidate_layer  argmax_l |concept-bin attribution| (E1 screening).
  e5_salient_layer_raw      argmax_l |behavioral-anchor deficit| over ALL
                      probed layers — mechanically biased to late layers
                      (ablating next to the unembed always moves logits).
  e5_salient_layer_corrected  the EARLIEST probed layer l <= 23 with
                      |anchor_deficit(l)| >= 70% of max_{l' <= 23}
                      |anchor_deficit(l')| (layer 25 excluded from both the
                      max and the candidates; frozen correction rule from the
                      wave-5 task).
  propagation_half_life     at the corrected salient layer l: the smallest
                      Δ = l' - l (probed l' > l) with |dom-meter deficit at
                      readout l'| <= 0.5 * |deficit at readout l|; null if the
                      deficit never halves by L25 (posset 'all').
  write_layer         argmax_l of E5 denoise-patching recovery m_denoise
                      (distributional denoise; e5 summary write_layer_denoise).
  copy_matrix_summary mean over l of C[l, next probed layer] and of C[l, 25]
                      (dom meter, all positions; C = identity-path share of
                      the readout deficit, weighted-median estimator; the
                      l'=25 column includes the final RMSNorm — impure).
  attn_self_repair    |frozen-attn deficit| - |deficit| mean over later probed
                      layers, per frozen layer (E5 --frozen; positive = attn
                      recomputation was repairing the ablation).

family_causal_rank (E4 rank curves, multiclass families; attached to every
concept of the family):
  collapse(k) = cloze_acc_clean - cloze_acc(k, concept-subspace basis).
  k50 / k90 = smallest k with collapse(k) >= 50% / 90% of collapse(k_max);
  null with a reason when rank curves were not run or the family never
  collapses (collapse(k_max) < 0.05). random_collapse_at_k90 = collapse at
  k90 under the matched-rank random-subspace control (Belrose license).

VERDICT RULES (frozen, task.md §6.1.0; computed PER ARM — the headline
`verdict` is the RIDGE arm, `dom_verdict` reported alongside):
  causal            (a) sufficiency: monotone dose-response at >= 1 layer —
                        cloze_slope ci_low > 0 with anti_steerable_frac <=
                        0.25 at that layer, OR ordinal_spearman >= 0.9 —
                        for THIS arm's rows;
                    (b) necessity (arm-agnostic by rule text: "ridge or dom"):
                        diag_lp_delta(ridge) or diag_lp_delta(dom) is
                        < -5 * |rand_mean| (5x more negative than the random
                        control) and < 0;
                    (c) specificity intact: |target_effect| > 3 *
                        max_offtarget_nonfamily (opposite-meter numbers).
                    A sub-criterion with NO data does not veto (recorded in
                    criteria[arm]["missing"] and in the fleet report), but a
                    measured failure does.
  read-only         Stage-6 deploy/caveat reader that fails the causal bar.
  artifact-suspect  fails the causal bar AND (Stage-6 reject OR necessity
                    indistinguishable from random). "Indistinguishable" is
                    implemented as: NEITHER ridge nor dom has
                    (value < 0 AND ci_high < rand_mean AND |value| >
                    2*|rand_mean|). artifact-suspect takes precedence over
                    read-only.
  Non-gating rule preserved: these verdicts never demote Stage-6.0 tiers.

Usage:
  python analyze.py                              # fleet: out/ -> out/causal_cards.json
  python analyze.py --roots <pilot_out>,<repo_out> --out <repo_out>/analysis_pilot
  python analyze.py --concepts january,harmfulness,europe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
STAGE_DIR = CODE_DIR.parent                       # concept_probes/stage6_1
CP_DIR = STAGE_DIR.parent                         # concept_probes
DEFAULT_CARDS = CP_DIR / "stage6" / "artifacts" / "probe_cards.json"

LAYERS = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25]
SCRIPTS = ("e1", "e2_cloze", "e2_ppl", "e4", "e5")
SCRIPT = "analyze"

# frozen thresholds (task.md §6.1.0 + wave-5 correction rule)
ANTI_MAX = 0.25            # max anti-steerable fraction for sufficiency
ORD_SPEARMAN_MIN = 0.9     # ordinal sufficiency bar
NECESSITY_X = 5.0          # arm must be 5x more negative than rand control
SPECIFICITY_X = 3.0        # target > 3x max off-family off-target
SALIENT_FRAC = 0.70        # corrected-salient threshold (of max over l<=23)
SALIENT_LMAX = 23          # layers above this excluded from the correction
RANK_MIN_COLLAPSE = 0.05   # below this the family "never collapses"
DIST_X = 2.0               # necessity distinguishable-from-random multiple


def canon(s) -> str:
    return str(s).replace(" ", "_")


def heartbeat(log: Path, msg: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {SCRIPT} {msg}\n")


def _f(x):
    """json-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def jsonsafe(x):
    if isinstance(x, dict):
        return {k: jsonsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonsafe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return _f(x)
    if isinstance(x, np.ndarray):
        return jsonsafe(x.tolist())
    return x


# ----------------------------------------------------------------- discovery
class Store:
    """Discovers + merges summary.jsonl rows and per-family npz across roots.

    Precedence: roots are scanned in CLI order; within a root, sorted file
    order; rows appended in line order. Dedup keeps the LAST row per key.
    npz: the LAST root that has the file wins. Paths containing 'smoke' or
    'dryrun' are always skipped.
    """

    def __init__(self, roots: list[Path]):
        self.roots = [Path(r) for r in roots]
        self._summaries: dict[str, list[dict]] = {}
        self._npz_cache: dict[tuple, object] = {}

    # ---- summaries
    def _summary_files(self, script: str) -> list[Path]:
        files: list[Path] = []
        for root in self.roots:
            cand: list[Path] = []
            cand += sorted(root.glob(f"{script}*/summary*.jsonl"))
            cand += sorted(root.glob(f"{script}*summary*.jsonl"))
            for p in cand:
                s = str(p)
                if "smoke" in s or "dryrun" in s:
                    continue
                # e2_cloze* glob must not swallow e2_ppl and vice versa; but
                # e2_cloze*/ matching e2_cloze_smoke already excluded above.
                if p not in files:
                    files.append(p)
        return files

    def summaries(self, script: str) -> list[dict]:
        if script not in self._summaries:
            dedup: dict[tuple, dict] = {}
            for p in self._summary_files(script):
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[{SCRIPT}] WARNING bad json line in {p}")
                            continue
                        cfg = r.get("config") or {}
                        key = (canon(r.get("concept")), r.get("family"),
                               str(r.get("layer")), r.get("arm"),
                               r.get("metric"), cfg.get("k"),
                               cfg.get("alpha_factor"))
                        dedup[key] = r          # keep last
            self._summaries[script] = list(dedup.values())
        return self._summaries[script]

    def rows(self, script: str, family: str, concept: str) -> list[dict]:
        c = canon(concept)
        return [r for r in self.summaries(script)
                if r.get("family") == family and canon(r.get("concept")) == c]

    def family_rows(self, script: str, family: str) -> list[dict]:
        return [r for r in self.summaries(script) if r.get("family") == family]

    # ---- npz
    def npz(self, script: str, name: str):
        """name e.g. 'months' / 'offtarget_months' / 'months.ablate'."""
        key = (script, name)
        if key not in self._npz_cache:
            found = None
            for root in self.roots:            # later roots override
                for d in sorted(root.glob(f"{script}*")):
                    if not d.is_dir() or "smoke" in d.name or "dryrun" in d.name:
                        continue
                    p = d / f"{name}.npz"
                    if p.exists():
                        found = p
            self._npz_cache[key] = (np.load(found, allow_pickle=True)
                                    if found else None)
        return self._npz_cache[key]


def row_index(rows: list[dict]):
    """(metric, arm) -> {layer_str: row}."""
    idx: dict[tuple, dict] = {}
    for r in rows:
        idx.setdefault((r["metric"], r["arm"]), {})[str(r["layer"])] = r
    return idx


def rv(row):
    """(value, ci_low, ci_high, n) from a summary row, None-safe."""
    if row is None:
        return None
    return {"value": _f(row.get("value")), "ci_low": _f(row.get("ci_low")),
            "ci_high": _f(row.get("ci_high")), "n": row.get("n")}


# --------------------------------------------------------------- sufficiency
def build_sufficiency(store: Store, family: str, cls: str, s6_layer):
    rows = store.rows("e2_cloze", family, cls)
    ppl_rows = store.rows("e2_ppl", family, cls)
    if not rows and not ppl_rows:
        return None, {}
    idx = row_index(rows)

    def layers_of(metric, arm):
        return sorted(idx.get((metric, arm), {}), key=lambda s: int(s))

    has_cloze = bool(idx.get(("cloze_slope", "ridge")))
    has_ord = bool(idx.get(("ordinal_spearman", "ridge")))
    intensity = has_ord and not has_cloze

    def pick_best(arm):
        """(layer, rule, pass) under the frozen sufficiency bar."""
        if has_cloze:
            passing, all_l = [], []
            for ls in layers_of("cloze_slope", arm):
                sl = idx[("cloze_slope", arm)][ls]
                af = idx.get(("anti_steerable_frac", arm), {}).get(ls)
                v, lo = sl.get("value"), sl.get("ci_low")
                afv = af.get("value") if af else None
                all_l.append((ls, v))
                if (lo is not None and lo > 0 and afv is not None
                        and afv <= ANTI_MAX):
                    passing.append((ls, v))
            pool = passing or all_l
            pool = [(l, v) for l, v in pool if v is not None]
            if not pool:
                return None, "cloze", False
            best = max(pool, key=lambda t: t[1])[0]
            return int(best), "cloze_ci>0&anti<=%.2f" % ANTI_MAX, bool(passing)
        if has_ord:
            pool = []
            for ls in layers_of("ordinal_spearman", arm):
                sp = idx[("ordinal_spearman", arm)][ls].get("value")
                osl = idx.get(("ordinal_slope", arm), {}).get(ls, {})
                pool.append((ls, sp if sp is not None else -np.inf,
                             osl.get("value") or -np.inf))
            if not pool:
                return None, "ordinal", False
            best = max(pool, key=lambda t: (t[1], t[2]))
            return (int(best[0]), "ordinal_spearman>=%.1f" % ORD_SPEARMAN_MIN,
                    bool(best[1] >= ORD_SPEARMAN_MIN))
        return None, "none", False

    best = {}
    for arm in ("ridge", "dom"):
        L, rule, ok = pick_best(arm)
        best[arm] = {"layer": L, "rule": rule, "pass": ok}

    ref_layer = best["ridge"]["layer"]

    def at(metric, arm, layer):
        if layer is None:
            return None
        return rv(idx.get((metric, arm), {}).get(str(layer)))

    suff = {
        "best_layer": ref_layer,
        "best_layer_rule": best["ridge"]["rule"],
        "intensity_axis": intensity,
        "cloze_slope": {a: at("cloze_slope", a, ref_layer)
                        for a in ("ridge", "dom", "rand")} if has_cloze else None,
        "anti_steerable_frac": {a: at("anti_steerable_frac", a, ref_layer)
                                for a in ("ridge", "dom", "rand")}
        if has_cloze else None,
        "suppression_slope": {a: at("suppression_slope", a, ref_layer)
                              for a in ("ridge", "dom", "rand")}
        if has_cloze else None,
        "ordinal_spearman": {a: at("ordinal_spearman", a, ref_layer)
                             for a in ("ridge", "dom", "rand")}
        if has_ord else None,
        "ordinal_slope": {a: at("ordinal_slope", a, ref_layer)
                          for a in ("ridge", "dom", "rand")}
        if has_ord else None,
        "dom_best_layer": best["dom"]["layer"],
    }

    # e2_ppl block: prefer the Stage-6 card layer, else first layer present.
    if ppl_rows:
        pidx = row_index(ppl_rows)
        ppl_layers = sorted({int(r["layer"]) for r in ppl_rows})
        pl = int(s6_layer) if (s6_layer is not None
                               and int(s6_layer) in ppl_layers) else \
            (ppl_layers[0] if ppl_layers else None)
        suff["ppl"] = {
            "layer": pl,
            "rel_delta_slope": {a: rv(pidx.get(("rel_delta_slope", a),
                                               {}).get(str(pl)))
                                for a in ("ridge", "dom", "rand")},
            "rel_delta_f2": {a: rv(pidx.get(("rel_delta_f2", a),
                                            {}).get(str(pl)))
                             for a in ("ridge", "dom", "rand")},
            "irrel_delta_f2": {a: rv(pidx.get(("irrel_delta_f2", a),
                                              {}).get(str(pl)))
                               for a in ("ridge", "dom", "rand")},
        }
    else:
        suff["ppl"] = None

    passes = {a: (best[a]["pass"] and best[a]["layer"] is not None)
              for a in ("ridge", "dom")}
    if not rows:                       # ppl-only concept: no dose-response data
        passes = {"ridge": None, "dom": None}
    return suff, passes


# ----------------------------------------------------------------- necessity
E4_ARM_GROUPS = {"rand": [f"rand{k}" for k in range(5)],
                 "other": [f"other{k}" for k in range(3)]}


def build_necessity(store: Store, family: str, cls: str):
    rows = [r for r in store.rows("e4", family, cls)
            if str(r["layer"]).startswith("all_")]
    restore_rows = [r for r in store.rows("e4", family, cls)
                    if r["metric"] == "cloze_recovery_frac"]
    if not rows and not restore_rows:
        return None, None, None
    idx = row_index(rows)

    def arm_row(metric, arm):
        d = idx.get((metric, arm), {})
        return next(iter(d.values())) if d else None

    def group_mean(metric, group):
        vals = [arm_row(metric, a) for a in E4_ARM_GROUPS[group]]
        vals = [v["value"] for v in vals if v and v.get("value") is not None]
        return _f(np.mean(vals)) if vals else None

    def block(metric):
        out = {a: rv(arm_row(metric, a)) for a in ("ridge", "dom")}
        out["rand_mean"] = group_mean(metric, "rand")
        out["rand_values"] = [
            _f((arm_row(metric, a) or {}).get("value"))
            for a in E4_ARM_GROUPS["rand"] if arm_row(metric, a)]
        out["other_mean"] = group_mean(metric, "other")
        return out if any(out[a] for a in ("ridge", "dom")) else None

    diag = block("diag_lp_delta")
    nec = {
        "diag_lp_delta": diag,
        "bpt_delta_bits": block("bpt_delta_bits"),
        "cloze_acc_delta": block("cloze_acc_delta"),
        "cloze_target_lp_delta": block("cloze_target_lp_delta"),
        "kl_guard": None,
        "restore": None,
    }
    kl = block("kl_neutral_nats")
    if kl:
        r, rm = (kl.get("ridge") or {}).get("value"), kl.get("rand_mean")
        kl["ratio_ridge_vs_rand"] = _f(r / rm) if (r and rm) else None
        nec["kl_guard"] = kl
    if restore_rows:
        nec["restore"] = [
            {"factor": (r.get("config") or {}).get("alpha_factor"),
             "layer": r.get("layer"), "recovery_frac": _f(r.get("value"))}
            for r in sorted(restore_rows,
                            key=lambda r: (r.get("config") or {})
                            .get("alpha_factor") or 0)]

    # frozen necessity rule ("ridge or dom" — arm-agnostic)
    nec_pass = None
    indist = None
    if diag:
        rand_mean = diag.get("rand_mean")
        arm_ok, arm_dist = [], []
        for a in ("ridge", "dom"):
            v = (diag.get(a) or {}).get("value")
            hi = (diag.get(a) or {}).get("ci_high")
            if v is None or rand_mean is None:
                continue
            arm_ok.append(v < 0 and v < -NECESSITY_X * abs(rand_mean))
            arm_dist.append(v < 0 and (hi is not None and hi < rand_mean)
                            and abs(v) > DIST_X * abs(rand_mean))
        if arm_ok:
            nec_pass = any(arm_ok)
            indist = not any(arm_dist)
    nec["pass_rule"] = (f"ridge-or-dom diag_lp_delta < -{NECESSITY_X:g}x"
                        "|rand_mean|")
    nec["necessity_pass"] = nec_pass
    nec["indistinguishable_from_random"] = indist
    return nec, nec_pass, indist


# --------------------------------------------------------------- specificity
def build_specificity(store: Store, family: str, cls: str):
    z = store.npz("e2_cloze", f"offtarget_{family}")
    if z is None:
        return None, {"ridge": None, "dom": None}
    steered = [canon(s) for s in z["steered"]]
    if canon(cls) not in steered:
        return None, {"ridge": None, "dom": None}
    si = steered.index(canon(cls))
    rc = [canon(c) for c in z["readout_concepts"]]
    rf = [str(f) for f in z["readout_families"]]
    meters = [str(m) for m in z["meters"]]
    arms = [str(a) for a in z["steer_arms"]]
    delta = z["delta"]                              # [nS, arm, 64, meter]
    try:
        tidx = next(j for j in range(len(rc))
                    if rc[j] == canon(cls) and rf[j] == family)
    except StopIteration:
        return None, {"ridge": None, "dom": None}
    sib = [j for j in range(len(rc)) if rf[j] == family and j != tidx]
    non = [j for j in range(len(rc)) if rf[j] != family]

    out = {"factor": _f(z["factor"]), "steer_layer": int(z["steer_layer"][si]),
           "meter_rule": "primary meter = opposite fitted vector "
                         "(ridge steer -> dom meter and vice versa)",
           "arms": {}}
    passes = {}
    for a in ("ridge", "dom"):
        if a not in arms:
            passes[a] = None
            continue
        ai = arms.index(a)
        mi_op = meters.index("dom" if a == "ridge" else "ridge")
        mi_same = meters.index(a)
        d_op = delta[si, ai, :, mi_op]
        d_same = delta[si, ai, :, mi_same]

        def top(idxs, d):
            if not idxs:
                return None
            j = max(idxs, key=lambda j: abs(d[j]))
            return {"concept": rc[j], "family": rf[j], "delta": _f(d[j])}

        tgt_op, tgt_same = _f(d_op[tidx]), _f(d_same[tidx])
        max_non = top(non, d_op)
        ratio = (_f(abs(tgt_op) / abs(max_non["delta"]))
                 if (tgt_op is not None and max_non
                     and max_non["delta"]) else None)
        max_non_same = top(non, d_same)
        ratio_same = (_f(abs(tgt_same) / abs(max_non_same["delta"]))
                      if (tgt_same is not None and max_non_same
                          and max_non_same["delta"]) else None)
        out["arms"][a] = {
            "target_effect": tgt_op,
            "target_effect_same_meter": tgt_same,
            "max_offtarget_sibling": top(sib, d_op),
            "max_offtarget_nonfamily": max_non,
            "ratio": ratio,
            "ratio_same_meter": ratio_same,
        }
        passes[a] = (ratio is not None and ratio > SPECIFICITY_X)
    return out, passes


# ---------------------------------------------------------------- layer story
def build_layer_story(store: Store, family: str, cls: str, s6_layer):
    e1z = store.npz("e1", family)
    e5z = store.npz("e5", family)
    e5rows = store.rows("e5", family, cls)
    if e1z is None and e5z is None and not e5rows:
        return None
    story = {"stage6_chosen_layer": s6_layer}

    # E1 candidate
    story["e1_candidate_layer"] = None
    story["e1_disagrees_with_stage6"] = None
    if e1z is not None:
        cls_list = [canon(c) for c in e1z["classes"]]
        if canon(cls) in cls_list:
            ci = cls_list.index(canon(cls))
            story["e1_candidate_layer"] = int(e1z["candidate_layer_abs"][ci])
            story["e1_disagrees_with_stage6"] = bool(
                int(e1z["disagree"][ci]) == 1)
    if story["e1_candidate_layer"] is None:
        for r in store.rows("e1", family, cls):
            cfg = r.get("config") or {}
            if "candidate_layer_abs" in cfg:
                story["e1_candidate_layer"] = int(cfg["candidate_layer_abs"])
                story["e1_disagrees_with_stage6"] = bool(cfg.get("disagree"))
                break

    # E5 npz block (key prefix '<class>__ridge__')
    pre = f"{canon(cls)}__ridge__"
    raw = corr = half = None
    copy_summary = None
    write_layer = None
    if e5z is not None and f"{pre}layers" in e5z.files:
        layers = [int(x) for x in e5z[f"{pre}layers"]]
        ad = np.asarray(e5z[f"{pre}anchor_deficit"], dtype=float) \
            if f"{pre}anchor_deficit" in e5z.files else \
            np.nanmean(e5z[f"{pre}anchor_abl_ex"], axis=1) \
            - np.nanmean(e5z[f"{pre}anchor_clean_ex"])
        aad = np.abs(ad)
        if np.isfinite(aad).any():
            raw = layers[int(np.nanargmax(np.where(np.isfinite(aad), aad,
                                                   -np.inf)))]
            # corrected: earliest l<=23 with |ad| >= 70% of max over l<=23
            mask = [i for i, L in enumerate(layers)
                    if L <= SALIENT_LMAX and np.isfinite(aad[i])]
            if mask:
                mx = max(aad[i] for i in mask)
                if mx > 0:
                    for i in mask:
                        if aad[i] >= SALIENT_FRAC * mx:
                            corr = layers[i]
                            break
        # propagation half-life at the corrected layer (dom meter, all pos)
        if corr is not None:
            meters = [str(m) for m in e5z["meters"]]
            possets = [str(p) for p in e5z["possets"]]
            mi, pi = meters.index("dom"), possets.index("all")
            defc = np.abs(e5z[f"{pre}deficit"][:, :, mi, pi])   # [L,R]
            li = layers.index(corr)
            d0 = defc[li, li]
            if np.isfinite(d0) and d0 > 0:
                for r in range(li + 1, len(layers)):
                    if np.isfinite(defc[li, r]) and defc[li, r] <= 0.5 * d0:
                        half = layers[r] - corr
                        break
        # copy-matrix summary (dom meter, all positions)
        meters = [str(m) for m in e5z["meters"]]
        possets = [str(p) for p in e5z["possets"]]
        mi, pi = meters.index("dom"), possets.index("all")
        C = e5z[f"{pre}C"][:, :, mi, pi]                        # [L,R]
        adj = [C[i, i + 1] for i in range(len(layers) - 1)]
        r25 = layers.index(25) if 25 in layers else None
        to25 = ([C[i, r25] for i in range(len(layers)) if layers[i] < 25]
                if r25 is not None else [])
        with np.errstate(invalid="ignore"):
            copy_summary = {
                "mean_adjacent": _f(np.nanmean(adj)) if adj else None,
                "mean_to_L25": _f(np.nanmean(to25)) if to25 else None,
                "meter": "dom", "posset": "all",
                "l25_note": "l'=25 readout includes the final RMSNorm",
            }
        if f"{pre}patch_m_denoise" in e5z.files:
            md = np.asarray(e5z[f"{pre}patch_m_denoise"], dtype=float)
            if np.isfinite(md).any():
                write_layer = layers[int(np.nanargmax(
                    np.where(np.isfinite(md), md, -np.inf)))]

    # summary-row fallbacks / extras
    idx = row_index(e5rows)
    if raw is None:
        d = idx.get(("salient_layer", "ridge"), {})
        if d:
            raw = int(next(iter(d.values()))["value"])
    if write_layer is None:
        d = idx.get(("write_layer_denoise", "ridge"), {})
        if d:
            write_layer = int(next(iter(d.values()))["value"])
    attn = [{"layer": int(r["layer"]), "value": _f(r["value"])}
            for r in e5rows if r["metric"] == "attn_self_repair_dom"]

    story.update({
        "e5_salient_layer_raw": raw,
        "e5_salient_layer_corrected": corr,
        "corrected_rule": (f"earliest probed l<={SALIENT_LMAX} with "
                           f"|anchor_deficit| >= {SALIENT_FRAC:.0%} of max "
                           f"over l<={SALIENT_LMAX} (L25 excluded)"),
        "propagation_half_life_layers": half,
        "write_layer": write_layer,
        "copy_matrix_summary": copy_summary,
        "attn_self_repair": attn or None,
    })
    return story


# ---------------------------------------------------------------- family rank
def build_family_rank(store: Store, family: str, cache: dict):
    if family in cache:
        return cache[family]
    z = store.npz("e4", f"{family}.rank")
    out = None
    if z is not None:
        ks = [int(k) for k in z["ks"]]
        bases = [str(b) for b in z["bases"]]
        bc, br = bases.index("concept"), bases.index("random")
        acc_clean = float(z["cloze_acc_clean"])
        acc = np.asarray(z["cloze_acc"], dtype=float)          # [K, 2]
        collapse_c = acc_clean - acc[:, bc]
        collapse_r = acc_clean - acc[:, br]
        full = collapse_c[-1] if np.isfinite(collapse_c[-1]) else np.nan
        if not np.isfinite(full) or full < RANK_MIN_COLLAPSE:
            out = {"n_classes": len(ks) + 1, "k50": None, "k90": None,
                   "reason": "family cloze never collapses "
                             f"(full collapse {_f(full)})",
                   "full_collapse": _f(full)}
        else:
            def first_k(frac):
                for i, k in enumerate(ks):
                    if (np.isfinite(collapse_c[i])
                            and collapse_c[i] >= frac * full):
                        return k, i
                return None, None
            k50, _ = first_k(0.5)
            k90, i90 = first_k(0.9)
            out = {"n_classes": len(ks) + 1,
                   "k50": k50, "k90": k90, "full_collapse": _f(full),
                   "random_collapse_at_k90":
                       _f(collapse_r[i90]) if i90 is not None else None,
                   "collapse_curve_concept": [_f(v) for v in collapse_c],
                   "collapse_curve_random": [_f(v) for v in collapse_r],
                   "ks": ks, "cloze_acc_clean": _f(acc_clean)}
    else:
        # fallback: reconstruct from summary rows concept == '<fam>(family)'
        rows = [r for r in store.family_rows("e4", family)
                if r.get("concept") == f"{family}(family)"
                and r.get("metric") == "cloze_acc_delta"]
        if rows:
            by = {}
            for r in rows:
                k = (r.get("config") or {}).get("k")
                if k is not None:
                    by.setdefault(r["arm"], {})[int(k)] = -float(r["value"])
            cc = by.get("concept_subspace", {})
            cr = by.get("random_subspace", {})
            if cc:
                ks = sorted(cc)
                collapse_c = np.array([cc[k] for k in ks])
                full = collapse_c[-1]
                if full < RANK_MIN_COLLAPSE:
                    out = {"n_classes": ks[-1] + 1, "k50": None, "k90": None,
                           "reason": "family cloze never collapses",
                           "full_collapse": _f(full)}
                else:
                    k50 = next((k for i, k in enumerate(ks)
                                if collapse_c[i] >= 0.5 * full), None)
                    k90 = next((k for i, k in enumerate(ks)
                                if collapse_c[i] >= 0.9 * full), None)
                    out = {"n_classes": ks[-1] + 1, "k50": k50, "k90": k90,
                           "full_collapse": _f(full),
                           "random_collapse_at_k90":
                               _f(cr.get(k90)) if k90 else None,
                           "source": "summary_rows"}
    if out is None:
        out = {"k50": None, "k90": None,
               "reason": "no E4 rank data (rank curves not run or family "
                         "not multiclass)"}
    cache[family] = out
    return out


# -------------------------------------------------------------------- verdict
def decide(arm: str, s6_tier: str, suff_pass, nec_pass, indist, spec_pass):
    """Frozen §6.1.0 verdict for one arm. Returns (verdict, criteria dict)."""
    missing = [name for name, v in
               (("sufficiency", suff_pass), ("necessity", nec_pass),
                ("specificity", spec_pass)) if v is None]
    causal = (suff_pass is True and nec_pass is True
              and spec_pass is not False)
    if causal:
        verdict = "causal"
    elif s6_tier == "reject" or indist is True:
        verdict = "artifact-suspect"
    elif s6_tier in ("deploy", "caveat"):
        verdict = "read-only"
    else:
        verdict = "artifact-suspect"
    crit = {"sufficiency_pass": suff_pass, "necessity_pass": nec_pass,
            "specificity_pass": spec_pass,
            "necessity_indistinguishable_from_random": indist,
            "missing": missing}
    return verdict, crit


# ----------------------------------------------------------------------- main
def build_cards(store: Store, probe_cards: list[dict], concepts_filter,
                include_all: bool, log: Path):
    rank_cache: dict = {}
    cards, skipped = [], []
    for i, pc in enumerate(tqdm(probe_cards, desc="causal cards")):
        cls_raw, family = pc["concept"], pc["family"]
        cls = canon(cls_raw)
        if concepts_filter and cls not in concepts_filter:
            continue
        s6_layer = int(pc["layer"])
        s6_tier = pc.get("tier") or pc.get("verdict")

        suff, suff_pass = build_sufficiency(store, family, cls, s6_layer)
        nec, nec_pass, indist = build_necessity(store, family, cls)
        spec, spec_pass = build_specificity(store, family, cls)
        story = build_layer_story(store, family, cls, s6_layer)
        rank = build_family_rank(store, family, rank_cache)

        def npz_has_cls(script, name, key="classes"):
            z = store.npz(script, name)
            if z is None or key not in getattr(z, "files", []):
                return False
            return canon(cls) in [canon(c) for c in z[key]]

        data_present = {
            "e1": bool(store.rows("e1", family, cls)
                       or npz_has_cls("e1", family)),
            "e2_cloze": bool(store.rows("e2_cloze", family, cls)),
            "e2_cloze_npz": npz_has_cls("e2_cloze", family),
            "e2_ppl": bool(store.rows("e2_ppl", family, cls)),
            "e4": nec is not None,
            "e4_rank": "reason" not in (rank or {}),
            "e5": story is not None and (
                story.get("e5_salient_layer_raw") is not None),
            "offtarget": spec is not None,
        }
        # family-level rank data alone does not make a per-concept card
        if not any(v for k, v in data_present.items()
                   if k != "e4_rank") and not include_all:
            skipped.append(f"{family}.{cls}")
            continue

        verdict, crit_r = decide("ridge", s6_tier, suff_pass.get("ridge"),
                                 nec_pass, indist, spec_pass.get("ridge"))
        dom_verdict, crit_d = decide("dom", s6_tier, suff_pass.get("dom"),
                                     nec_pass, indist, spec_pass.get("dom"))
        cards.append({
            "concept": cls, "concept_display": cls_raw, "family": family,
            "stage6": {"verdict": s6_tier, "layer": s6_layer},
            "data_present": data_present,
            "sufficiency": suff,
            "necessity": nec,
            "specificity": spec,
            "layer_story": story,
            "family_causal_rank": rank,
            "verdict": verdict,
            "dom_verdict": dom_verdict,
            "criteria": {"ridge": crit_r, "dom": crit_d},
        })
        if (i + 1) % 8 == 0:
            heartbeat(log, f"{family}.{cls} {i + 1}/{len(probe_cards)}")
    return cards, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", default=str(STAGE_DIR / "out"),
                    help="comma list of output roots; later roots take "
                         "precedence in merges")
    ap.add_argument("--cards", default=str(DEFAULT_CARDS),
                    help="stage6 probe_cards.json")
    ap.add_argument("--out", default=str(STAGE_DIR / "out"),
                    help="directory for causal_cards.json")
    ap.add_argument("--concepts", default="",
                    help="comma filter (canonical class names)")
    ap.add_argument("--all-concepts", action="store_true",
                    help="emit cards even for concepts with no 6.1 data")
    args = ap.parse_args(argv)

    roots = [Path(r) for r in args.roots.split(",") if r]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"progress_{SCRIPT}.log"
    heartbeat(log, f"start roots={[str(r) for r in roots]}")

    store = Store(roots)
    for s in SCRIPTS:
        files = store._summary_files(s)
        n = len(store.summaries(s))
        print(f"[{SCRIPT}] {s}: {len(files)} summary file(s), "
              f"{n} rows after dedup "
              f"({', '.join(str(p) for p in files) or 'none found'})")

    probe_cards = json.load(open(args.cards))
    concepts_filter = {canon(c) for c in args.concepts.split(",") if c}
    cards, skipped = build_cards(store, probe_cards, concepts_filter,
                                 args.all_concepts, log)

    counts: dict[str, int] = {}
    for c in cards:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "roots": [str(r) for r in roots],
        "probe_cards": args.cards,
        "n_cards": len(cards),
        "verdict_counts_ridge": counts,
        "skipped_no_data": skipped,
        "verdict_rules": ("frozen task.md §6.1.0; headline verdict = RIDGE "
                          "arm; see analyze.py docstring for thresholds"),
    }
    out_path = out_dir / "causal_cards.json"
    with open(out_path, "w") as f:
        json.dump(jsonsafe({"_meta": meta, "cards": cards}), f, indent=1)
    print(f"[{SCRIPT}] wrote {out_path}  ({len(cards)} cards; "
          f"verdicts {counts}; {len(skipped)} concepts skipped for no data)")
    heartbeat(log, f"DONE {len(cards)} cards")
    for c in cards:
        print(f"  {c['family']}.{c['concept']}: {c['verdict']} "
              f"(dom: {c['dom_verdict']}; stage6 {c['stage6']['verdict']}; "
              f"missing {c['criteria']['ridge']['missing']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
