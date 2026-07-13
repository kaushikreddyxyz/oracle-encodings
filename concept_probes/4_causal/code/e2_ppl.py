"""Stage 6.1 E2b — ActAdd perplexity-ratio (task.md Section 6.1.4; base-model-
native, generation-free). Steers one (concept, layer, arm) at ALL positions
with Intervention(mode='steer', alpha=factor*s95) and measures mean per-token
log-likelihood on concept-relevant vs irrelevant natural text.

TEXT-BUCKET DATA PATH (documented per task):
  3_validation/data/natural/eval/<family>.jsonl — one row per natural-pool example
  with fields: example_id, text, token_ids (gemma-2 ids, NO BOS — extraction
  prepended BOS at forward time, 2_probes/code/extract.py), n_tokens, and
  targets = {class: [[token_idx, strength], ...]} with strength in [0,1]
  aggregated from the K=3 judge (same rows as stage4 judged_nat.jsonl —
  example_ids match natscores ex_example_id; this file was chosen over
  judged_nat.jsonl because it already carries per-class per-token judged
  strengths AND the exact token_ids the probes/natstats were built from).
  * relevant(concept)  = rows with ymax = max strength over targets[concept]
    >= 0.34 (task's judge-confirmed-positive bar), seeded subsample capped at
    --n-rel (default 150).
  * irrelevant (shared) = rows with ALL classes' target lists empty (judged 0
    for every class of their family), sampled ceil(n_irrel/13) per natscore
    family with a fixed seed (61), shared across concepts; each concept drops
    its OWN family's rows from the pool (so ~ (12/13)*n_irrel remain).

Metrics per (concept, layer, arm, factor): mean per-token logprob on the
relevant bucket, on the irrelevant bucket (token-weighted; per-example means
stored for bootstrap), deltas vs alpha=0, i.e. log of the perplexity ratio
(ppl_ratio = exp(-delta)). Coherence guard = irrelevant-bucket delta.

Default layers per concept: probe-card chosen layer (3_validation/artifacts/
probe_cards.json) + one band neighbor (next entry in the 12-layer list above
the chosen layer; below if chosen is 25) + layer 12, deduped. --layers
overrides with a fixed list for all concepts. Default factors
{-1,-0.5,0.5,1,1.5,2,3} (+0 baseline always included). Arms: ridge, dom,
rand (first 5 rand_dirs; per-dir stored).

Outputs:
  out/e2_ppl/<family>.npz keys (per steered concept <cls>, canonical name):
    concepts [C], dirs [D], factors [F] (includes 0),
    <cls>__layers [nL], <cls>__s95 [nL],
    <cls>__ex_ids [E], <cls>__ex_bucket [E] (1 rel / 0 irrel),
    <cls>__ex_ntok [E], <cls>__ex_fam [E],
    <cls>__lp_ex [nL, D, F, E] per-example mean token logprob,
    <cls>__rel_curve / <cls>__irrel_curve [nL, D, F] token-weighted bucket
      mean logprob (delta vs the factor-0 slice = log ppl-ratio).
  out/e2_ppl/summary.jsonl — DESIGN schema rows
    {"concept","family","layer","arm","metric","value","n","ci_low","ci_high",
     "config":{...}} with metrics rel_delta_slope (OLS of rel delta over
    factors in [-2,2]), rel_delta_f2 / irrel_delta_f2 (at factor
    min(2, max factor)), CIs = bootstrap over examples. rand rows aggregate
    the 5 dirs.

Layer-25 caveat: interventions at L25 edit the raw block-25 output (logits see
it) but the L25 probe meter reads post-final-RMSNorm; rows carry
config.layer25_note. glorptitude is skipped (no natscores -> no dose calib).

--smoke: tiny random Gemma2 (test_interventions fixture pattern) + synthetic
token-id buckets/arms/natstats; CPU fp32; end-to-end. --dry-run: loads REAL
arms, dose_calib, probe cards and text buckets, prints planned work +
estimated forwards, no model forwards.

  python e2_ppl.py --families months --device cuda
  python e2_ppl.py --smoke
  python e2_ppl.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                          # noqa: E402
from common import (FAMILIES, LAYERS, N_RAND_ARMS, dose_calib,  # noqa: E402
                    load_arms, load_natstats)
from interventions import Hooks, Intervention          # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = STAGE_DIR.parent / "3_validation" / "data" / "natural" / "eval"
CARDS_PATH = STAGE_DIR.parent / "3_validation" / "artifacts" / "probe_cards.json"
DEFAULT_FACTORS = [-1, -0.5, 0.5, 1, 1.5, 2, 3]
FIT_RANGE = (-2.0, 2.0)
YMAX_POS = 0.34
IRREL_SEED = 61
SCRIPT = "e2_ppl"


def canon(s: str) -> str:
    return str(s).replace(" ", "_")


def heartbeat(log: Path, msg: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {SCRIPT} {msg}\n")


def _jsonsafe(x):
    if isinstance(x, dict):
        return {k: _jsonsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonsafe(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def append_summary(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(_jsonsafe(r)) + "\n")


def boot_ci(vals, seed=0, n=1000):
    v = np.asarray(vals, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan")
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def tiny_model(d=64):
    """test_interventions.py fixture pattern: 2-layer random Gemma2, fp32."""
    from transformers import Gemma2Config, Gemma2ForCausalLM
    cfg = Gemma2Config(
        vocab_size=128, hidden_size=d, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64, attn_implementation="eager")
    torch.manual_seed(0)
    model = Gemma2ForCausalLM(cfg)
    model.config.output_hidden_states = True
    model.eval()
    return model


# -------------------------------------------------------------------- context
class Ctx:
    def __init__(self, args):
        self.args = args
        self.smoke = args.smoke
        self._eval_cache: dict[str, list] = {}
        if self.smoke:
            d = 64
            rng = np.random.default_rng(7)
            self._mu = rng.normal(size=d).astype(np.float32)
            self._sd = (0.5 + rng.random(d)).astype(np.float32)
            self.families = {"famA": ["alpha"], "famB": ["bravo"]}
            self._dirs = {}
            for c in ["alpha", "bravo"]:
                g = np.random.default_rng(abs(zlib.crc32(c.encode())))
                self._dirs[c] = {
                    "ridge": (self._unit(g.normal(size=d)), 0.1),
                    "dom": (self._unit(g.normal(size=d)), 0.0),
                    "rand": [self._unit(g.normal(size=d)) for _ in range(5)]}
            self.card_layer = {("famA", "alpha"): 0, ("famB", "bravo"): 0}
        else:
            self.families = {f: [canon(c) for c in cs]
                             for f, cs in FAMILIES.items()}
            cards = json.load(open(CARDS_PATH))
            self.card_layer = {(c["family"], canon(c["concept"])):
                               int(c["layer"]) for c in cards}

    @staticmethod
    def _unit(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    def natstats(self, layer):
        return (self._mu, self._sd) if self.smoke else load_natstats(layer)

    def arms(self, family, cls, layer):
        return self._dirs[cls] if self.smoke else load_arms(family, cls, layer)

    def s95(self, family, cls, layer):
        if self.smoke:
            return 1.0
        return float(dose_calib(family, cls, layer)["s95"])

    def model_tok(self):
        if self.smoke:
            return tiny_model(), None
        return common.load_model(device=self.args.device,
                                 dtype=self.args.dtype)

    # ------------------------------------------------------------- text pool
    def eval_rows(self, family):
        """[(example_id, token_ids, ymax_by_class dict, neutral bool)]"""
        if family in self._eval_cache:
            return self._eval_cache[family]
        if self.smoke:
            rng = np.random.default_rng(abs(zlib.crc32(family.encode())))
            rows = []
            for i in range(12):
                ids = rng.integers(2, 128, size=int(rng.integers(8, 16))).tolist()
                pos = i < 6
                ym = {c: (0.6 if pos else 0.0) for c in self.families[family]}
                rows.append((f"{family}_ex{i}", ids, ym, not pos))
            self._eval_cache[family] = rows
            return rows
        path = EVAL_DIR / f"{family}.jsonl"
        rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                ym = {canon(c): max((s for _, s in v), default=0.0)
                      for c, v in r["targets"].items()}
                rows.append((r["example_id"], r["token_ids"], ym,
                             all(y == 0.0 for y in ym.values())))
        self._eval_cache[family] = rows
        return rows

    def natscore_families(self):
        if self.smoke:
            return sorted(self.families)
        return sorted(f for f in self.families
                      if (common.NATSCORES_DIR / f"{f}.natscores.npz").exists())

    def irrel_pool(self, n_irrel):
        """Shared neutral pool: ceil(n/13) rows per natscore family, seed 61.
        -> [(family, example_id, token_ids)]"""
        fams = self.natscore_families()
        per = int(np.ceil(n_irrel / max(len(fams), 1)))
        pool = []
        for fam in fams:
            neut = [(eid, ids) for eid, ids, _, isneut in self.eval_rows(fam)
                    if isneut]
            rng = np.random.default_rng(IRREL_SEED + zlib.crc32(fam.encode()) % 1000)
            take = min(per, len(neut))
            for i in sorted(rng.choice(len(neut), take, replace=False)):
                pool.append((fam, neut[i][0], neut[i][1]))
        return pool

    def rel_bucket(self, family, cls, n_rel):
        """Judge-confirmed positives (ymax >= YMAX_POS), seeded cap at n_rel.
        -> [(example_id, token_ids, ymax)]"""
        rows = [(eid, ids, ym.get(cls, 0.0))
                for eid, ids, ym, _ in self.eval_rows(family)
                if ym.get(cls, 0.0) >= (YMAX_POS if not self.smoke else 0.5)]
        if len(rows) > n_rel:
            rng = np.random.default_rng(zlib.crc32(f"{family}.{cls}".encode()))
            keep = sorted(rng.choice(len(rows), n_rel, replace=False))
            rows = [rows[i] for i in keep]
        return rows


# ------------------------------------------------------------------ batching
def pack_ids(id_lists, batch_tokens, pad_id=0, bos_id=2):
    """BOS-prepend + greedy length-sorted packing (B*Lmax <= batch_tokens).
    Yields (row_indices, ids [B,L], attn [B,L])."""
    seqs = [[bos_id] + list(ids) for ids in id_lists]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    groups, cur, maxlen = [], [], 0
    for i in order:
        m = max(maxlen, len(seqs[i]))
        if cur and m * (len(cur) + 1) > batch_tokens:
            groups.append(cur); cur, maxlen = [], 0
        cur.append(i); maxlen = max(maxlen, len(seqs[i]))
    if cur:
        groups.append(cur)
    out = []
    for idxs in groups:
        L = max(len(seqs[i]) for i in idxs)
        ids = torch.full((len(idxs), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(idxs), L), dtype=torch.long)
        for r, i in enumerate(idxs):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i], dtype=torch.long)
            attn[r, :len(seqs[i])] = 1
        out.append((idxs, ids, attn))
    return out


def token_logprobs(logits, ids):
    """fp32 [B, L-1] logprob of ids[:,1:], chunked fp32 upcast."""
    B, L, V = logits.shape
    out = torch.empty(B, L - 1, dtype=torch.float32, device=logits.device)
    chunk = max(1, int(2 ** 28 / max((L - 1) * V, 1)))
    for b0 in range(0, B, chunk):
        lg = logits[b0:b0 + chunk, :-1].float()
        tgt = ids[b0:b0 + chunk, 1:].unsqueeze(-1)
        out[b0:b0 + chunk] = (lg.gather(-1, tgt).squeeze(-1)
                              - torch.logsumexp(lg, dim=-1))
    return out


@torch.no_grad()
def eval_batches(model, batches, n_rows, device):
    """(sum_lp [E], n_tok [E]) over each row's real (non-BOS) tokens under the
    current model state."""
    s = np.zeros(n_rows, dtype=np.float64)
    n = np.zeros(n_rows, dtype=np.int64)
    for idxs, ids, attn in batches:
        out = model(ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=False)
        lp = token_logprobs(out.logits, ids.to(device))
        m = attn[:, 1:].to(device).bool()
        lp = torch.where(m, lp, torch.zeros_like(lp)).cpu().numpy()
        nt = attn[:, 1:].sum(dim=1).numpy()
        for r, i in enumerate(idxs):
            s[i] = lp[r].sum(); n[i] = int(nt[r])
    return s, n


# --------------------------------------------------------------------- plan
def dir_list(arms_sel):
    names = []
    for a in arms_sel:
        if a == "rand":
            names += [f"rand{k}" for k in range(N_RAND_ARMS)]
        else:
            names.append(a)
    return names


def get_dir(arms, name):
    if name.startswith("rand"):
        k = int(name[4:])
        return arms["rand"][min(k, len(arms["rand"]) - 1)]
    return arms[name][0]


def concept_layers(ctx, family, cls, args):
    """chosen card layer + band neighbor + 12, deduped/sorted; or --layers."""
    if args.layers:
        return list(args.layers)
    Lc = ctx.card_layer.get((family, cls), 12)
    grid = LAYERS if not ctx.smoke else [0, 1]
    i = grid.index(Lc) if Lc in grid else min(
        range(len(grid)), key=lambda j: abs(grid[j] - Lc))
    nb = grid[i + 1] if i + 1 < len(grid) else grid[i - 1]
    mid = 12 if not ctx.smoke else 0
    return sorted(set([Lc, nb, mid]))


def build_concept_examples(ctx, family, cls, pool, args):
    """(ex_ids, ex_fams, id_lists, bucket [E] 1/0) — rel + shared irrel with
    the concept's own family excluded from the irrel pool."""
    rel = ctx.rel_bucket(family, cls, args.n_rel)
    irr = [(f, eid, ids) for f, eid, ids in pool if f != family][:args.n_irrel]
    if args.limit:
        rel, irr = rel[:args.limit], irr[:args.limit]
    ex_ids = [eid for eid, _, _ in rel] + [eid for _, eid, _ in irr]
    ex_fams = [family] * len(rel) + [f for f, _, _ in irr]
    id_lists = [ids for _, ids, _ in rel] + [ids for _, _, ids in irr]
    bucket = np.array([1] * len(rel) + [0] * len(irr), dtype=np.int8)
    return ex_ids, ex_fams, id_lists, bucket


def plan(ctx, fams, args):
    """{family: [(cls, layers, n_ex, n_batches)]}, total forwards."""
    pool = ctx.irrel_pool(args.n_irrel)
    n_dirs = len(dir_list(args.arms))
    n_nz = sum(1 for f in args.factors if f != 0)
    out, total = {}, 0
    for fam in fams:
        per = []
        for cls in ctx.families[fam]:
            if args.classes and cls not in args.classes:
                continue
            layers = concept_layers(ctx, fam, cls, args)
            ex_ids, _, id_lists, bucket = build_concept_examples(
                ctx, fam, cls, pool, args)
            nb = len(pack_ids(id_lists, args.batch_tokens))
            nf = nb * (1 + len(layers) * n_dirs * n_nz)
            per.append((cls, layers, len(ex_ids), int(bucket.sum()), nb, nf))
            total += nf
        if per:
            out[fam] = per
    return out, pool, total


# ---------------------------------------------------------------------- run
def run_concept(ctx, model, fam, cls, layers, pool, args, log, device,
                bos_id, pad_id):
    factors = args.factors
    nz = [f for f in factors if f != 0]
    f0i = factors.index(0)
    dirs = dir_list(args.arms)
    ex_ids, ex_fams, id_lists, bucket = build_concept_examples(
        ctx, fam, cls, pool, args)
    E = len(ex_ids)
    batches = pack_ids(id_lists, args.batch_tokens, pad_id, bos_id)

    s0, ntok = eval_batches(model, batches, E, device)
    lp_ex = np.full((len(layers), len(dirs), len(factors), E), np.nan,
                    dtype=np.float32)
    sum_lp = np.full((len(layers), len(dirs), len(factors), E), np.nan,
                     dtype=np.float64)
    lp_ex[:, :, f0i] = (s0 / np.maximum(ntok, 1)).astype(np.float32)
    sum_lp[:, :, f0i] = s0
    s95_v = np.full(len(layers), np.nan, dtype=np.float32)

    for li, layer in enumerate(layers):
        s95 = ctx.s95(fam, cls, layer)
        s95_v[li] = s95
        arms = ctx.arms(fam, cls, layer)
        mu, sd = ctx.natstats(layer)
        for di, dname in enumerate(dirs):
            w = get_dir(arms, dname)
            for f in nz:
                fi = factors.index(f)
                iv = Intervention(layer, w, "steer", alpha=f * s95)
                with Hooks(model, [iv], {layer: (mu, sd)}):
                    s, _ = eval_batches(model, batches, E, device)
                lp_ex[li, di, fi] = (s / np.maximum(ntok, 1)).astype(np.float32)
                sum_lp[li, di, fi] = s
        heartbeat(log, f"{fam}.{cls} L{layer} {li + 1}/{len(layers)}")

    rel, irr = bucket == 1, bucket == 0
    ntokf = ntok.astype(np.float64)
    rel_curve = (sum_lp[..., rel].sum(-1) / max(ntokf[rel].sum(), 1)) \
        .astype(np.float32)
    irrel_curve = (sum_lp[..., irr].sum(-1) / max(ntokf[irr].sum(), 1)) \
        .astype(np.float32)
    return dict(layers=np.array(layers), s95=s95_v,
                ex_ids=np.array(ex_ids), ex_fam=np.array(ex_fams),
                ex_bucket=bucket, ex_ntok=ntok, lp_ex=lp_ex,
                rel_curve=rel_curve, irrel_curve=irrel_curve)


def summarize(fam, cls, res, args, dirs):
    factors = np.asarray(args.factors, float)
    f0i = int(np.where(factors == 0)[0][0])
    f2 = factors[np.abs(factors) <= 2].max()
    f2i = int(np.where(factors == f2)[0][0])
    fit = (factors >= FIT_RANGE[0]) & (factors <= FIT_RANGE[1])
    rel = res["ex_bucket"] == 1
    rows = []
    arm_groups = {a: ([a] if a != "rand" else
                      [d for d in dirs if d.startswith("rand")])
                  for a in args.arms}
    for li, layer in enumerate(res["layers"]):
        layer = int(layer)
        cfg = {"factors": args.factors, "fit_range": list(FIT_RANGE),
               "s95": float(res["s95"][li]), "n_rel": int(rel.sum()),
               "n_irrel": int((~rel).sum()), "f2": float(f2),
               "ymax_pos": YMAX_POS,
               "bucket_source": "3_validation/data/natural/eval/<family>.jsonl"}
        if layer == 25:
            cfg["layer25_note"] = ("L25 probe meter reads post-final-RMSNorm; "
                                   "exact-alpha identity holds -1..24 only")
        for arm, members in arm_groups.items():
            dsel = [dirs.index(m) for m in members]
            base = dict(concept=cls, family=fam, layer=layer, arm=arm,
                        config=cfg)
            # per-example deltas at each factor, averaged over the arm's dirs
            d_ex = (res["lp_ex"][li, dsel] -
                    res["lp_ex"][li, dsel][:, f0i:f0i + 1]).mean(axis=0)
            # slope of the rel-bucket delta curve, bootstrap over examples
            sl_ex = np.array([ols_slope(factors[fit], d_ex[fit, e])
                              for e in np.where(rel)[0]])
            m, lo, hi = boot_ci(sl_ex, seed=zlib.crc32(f"{cls}{layer}".encode()) % 2**31)
            rows.append({**base, "metric": "rel_delta_slope", "value": m,
                         "n": int(rel.sum()), "ci_low": lo, "ci_high": hi})
            m, lo, hi = boot_ci(d_ex[f2i, rel],
                                seed=zlib.crc32(f"{cls}{layer}f2".encode()) % 2**31)
            rows.append({**base, "metric": "rel_delta_f2", "value": m,
                         "n": int(rel.sum()), "ci_low": lo, "ci_high": hi})
            m, lo, hi = boot_ci(d_ex[f2i, ~rel],
                                seed=zlib.crc32(f"{cls}{layer}ir".encode()) % 2**31)
            rows.append({**base, "metric": "irrel_delta_f2", "value": m,
                         "n": int((~rel).sum()), "ci_low": lo, "ci_high": hi})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", default="")
    ap.add_argument("--classes", default="")
    ap.add_argument("--layers", default="",
                    help="fixed comma list; default per-concept "
                         "(card layer + band neighbor + 12)")
    ap.add_argument("--factors", default=",".join(map(str, DEFAULT_FACTORS)))
    ap.add_argument("--arms", default="ridge,dom,rand")
    ap.add_argument("--out", default=str(STAGE_DIR / "out" / SCRIPT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap examples per bucket (0 = n-rel/n-irrel)")
    ap.add_argument("--n-rel", type=int, default=150)
    ap.add_argument("--n-irrel", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.layers = [int(x) for x in args.layers.split(",") if x != ""]
    args.factors = [float(x) for x in args.factors.split(",") if x != ""]
    args.factors = [int(f) if f == int(f) else f for f in args.factors]
    if 0 not in args.factors:
        args.factors = sorted(args.factors + [0], key=float)
    args.arms = [a for a in args.arms.split(",") if a]
    args.classes = [canon(c) for c in args.classes.split(",") if c]
    if args.smoke:
        args.device, args.dtype = "cpu", "float32"
        args.n_rel, args.n_irrel = 6, 6
        if not args.layers:
            args.layers = [0]
        if [float(f) for f in args.factors] == \
                sorted([float(f) for f in DEFAULT_FACTORS] + [0.0]):
            args.factors = [-1, 0, 1, 2]
        args.out = str(STAGE_DIR / "out" / f"{SCRIPT}_smoke")

    ctx = Ctx(args)
    if args.families:
        fams = [f for f in args.families.split(",") if f]
    else:
        fams = ctx.natscore_families()
    skipped = [f for f in fams if f == "glorptitude"
               or (not ctx.smoke and f not in ctx.natscore_families())]
    for f in skipped:
        print(f"[{SCRIPT}] {f}: no natscores -> no dose calibration/buckets; "
              "skipped")
    fams = [f for f in fams if f not in skipped]

    planned, pool, total_fwd = plan(ctx, fams, args)
    n_concepts = sum(len(v) for v in planned.values())
    print(f"[{SCRIPT}] planned {n_concepts} concepts across "
          f"{len(planned)} families; estimated TOTAL forwards: {total_fwd} "
          f"(arms={args.arms}, factors={args.factors}, "
          f"layers={'per-concept auto' if not args.layers else args.layers}, "
          f"irrel pool={len(pool)} shared rows)")
    if args.dry_run:
        for fam, per in planned.items():
            for cls, layers, n_ex, n_rel, nb, nf in per:
                for L in layers:            # validate arms + calib plumbing
                    a = ctx.arms(fam, cls, L)
                    assert a["ridge"][0].ndim == 1
                    ctx.s95(fam, cls, L)
                print(f"  {fam}.{cls}: layers={layers} n_ex={n_ex} "
                      f"(rel={n_rel}, irrel={n_ex - n_rel}) "
                      f"batches={nb} forwards={nf}")
        print(f"[{SCRIPT}] DRY RUN ok — data plumbing validated, "
              "no model forwards")
        return 0
    if not planned:
        print(f"[{SCRIPT}] nothing to do")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"progress_{SCRIPT}.log"
    model, tok = ctx.model_tok()
    if not ctx.smoke:
        model.to(args.device)
        bos_id, pad_id = tok.bos_token_id, tok.pad_token_id or 0
    else:
        bos_id, pad_id = 1, 0
    dirs = dir_list(args.arms)

    done = 0
    for fam, per in planned.items():
        payload = {"concepts": np.array([c for c, *_ in per]),
                   "dirs": np.array(dirs),
                   "factors": np.array(args.factors, dtype=np.float32)}
        srows = []
        for cls, layers, *_ in tqdm(per, desc=f"{SCRIPT} {fam}"):
            res = run_concept(ctx, model, fam, cls, layers, pool, args, log,
                              args.device, bos_id, pad_id)
            for k, v in res.items():
                payload[f"{cls}__{k}"] = v
            srows += summarize(fam, cls, res, args, dirs)
            done += 1
            heartbeat(log, f"{fam}.{cls} concept {done}/{n_concepts}")
        np.savez_compressed(out_dir / f"{fam}.npz", **payload)
        append_summary(out_dir / "summary.jsonl", srows)
        print(f"[{SCRIPT}] wrote {out_dir / f'{fam}.npz'} "
              f"(+{len(srows)} summary rows)")
    heartbeat(log, "DONE")
    print(f"[{SCRIPT}] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
