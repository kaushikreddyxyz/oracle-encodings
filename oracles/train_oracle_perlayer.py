#!/usr/bin/env python3
"""Per-layer oracle trainer (Stage 7 corrected design: ONE layer per model).

Trains Qwen3-Embedding-0.6B (full fine-tune) + a shallow MLP head
(hidden 1024 -> 4096 -> GELU -> 54) to predict the 54 concept-probe
detection scores of EXACTLY ONE gemma-2-2b layer (6, 8, or 14) per token.

Objective (identical to stage-7 expA, the run that reached heldout median
R^2 = 0.6371 over 162 joint targets): MSE on corpus-standardized dequantized
scores:  target = (int8 * scale[l] + zero[l] - mean[l]) / std[l], per-column.
The one-layer rule is enforced physically: the stacked store scores_<sid>.npy
int8 [n, 3, 54] is sliced to [n, 54] at read time; nothing downstream ever
sees another layer's columns.

Data: the new stacked repos (corpus-scores / corpus-scores-overflow layout)
staged locally by prefetch_shards.py; raw text recovered from local ClimbMix
parquets (karpathy/climbmix-400b-shuffle) exactly as stage 7 did — the
gemma-token-id reproduction hard-assert is inherited from train_encoder.py.

Optimizer: Muon (Newton-Schulz orthogonalized momentum) for all 2D weight
matrices except embeddings; AdamW for embeddings, gains, biases. Cosine
schedule with warmup on both. (Stage 7 used plain AdamW; Muon is new here
per task spec.)

Stopping: whichever comes first —
  * token plateau: median heldout R^2 improves < --plateau-delta over the
    trailing --plateau-tokens of training tokens;
  * --max-tokens (default 1.3e9);
  * --max-hours wall clock.
Always checkpoints best.pt (val-best) + last.pt; optionally pushes stripped
checkpoints + metrics.jsonl to a HF model repo subfolder layerXX/.

Reuses train_encoder.py verbatim for: doc iteration/text recovery,
tokenization + alignment contract (process_doc), collate/gather, streaming
R^2, heartbeat. wandb logging is LIVE (stage 7 retro-logged).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_encoder as te  # noqa: E402

LAYER_TO_IDX = {6: 0, 8: 1, 14: 2}
K = 54


# ============================================================= stacked store
class StackedShardStore:
    """tokens_<sid>.npy int32 [n]; scores_<sid>.npy int8 [n, 3, 54];
    docs_<sid>.jsonl. Rows are sliced to ONE layer at read time."""

    def __init__(self, scores_dir, sid, layer_idx):
        self.sid = sid
        self.layer_idx = layer_idx
        self.tokens = np.load(os.path.join(scores_dir, f"tokens_{sid:05d}.npy"), mmap_mode="r")
        det = np.load(os.path.join(scores_dir, f"scores_{sid:05d}.npy"), mmap_mode="r")
        if det.ndim != 3 or det.shape[1:] != (3, K):
            raise ValueError(f"shard {sid}: expected [n,3,{K}] stacked store, got {det.shape}")
        self.det = det
        self.docs = []
        with open(os.path.join(scores_dir, f"docs_{sid:05d}.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.docs.append(json.loads(line))

    def doc_rows(self, start, n):
        """Physical one-layer slice: int8 [n, 54] for this store's layer."""
        return np.ascontiguousarray(self.det[start:start + n, self.layer_idx, :])


def wait_for_shard(scores_dir, sid, log_fn, poll_s=30):
    """Block until prefetch_shards.py marks the shard staged (.done sentinel)."""
    done = os.path.join(scores_dir, f"scores_{sid:05d}.npy.done")
    waited = 0
    while not os.path.exists(done):
        if waited == 0:
            log_fn(f"waiting for shard {sid} staging ({done})")
        time.sleep(poll_s)
        waited += poll_s
        if waited % 600 == 0:
            log_fn(f"still waiting for shard {sid} after {waited}s — prefetch behind or dead")
    return waited


def iter_docs_stacked(shards, scores_dir, climbmix_dir, layer_idx, loop, log_fn=print):
    """te.iter_docs adapted to the stacked [n,3,54] store + prefetch handshake.
    Yields the same dict contract process_doc expects, with scores_raw_i8
    already sliced to [n_doc, 54]."""
    while True:
        for sid in shards:
            wait_for_shard(scores_dir, sid, log_fn)
            store = StackedShardStore(scores_dir, sid, layer_idx)
            doc_idxs = [d["doc"] for d in store.docs]
            texts = te.load_shard_texts(climbmix_dir, sid, doc_idxs)
            for d in store.docs:
                text = texts.get(d["doc"])
                if text is None:
                    continue
                start, n = d["start"], d["n"]
                yield {
                    "sid": sid,
                    "doc_idx": d["doc"],
                    "text": text,
                    "gemma_ids": np.array(store.tokens[start:start + n]),
                    "scores_raw_i8": store.doc_rows(start, n),
                }
        if not loop:
            return


# ==================================================================== model
class OracleMLPHead(nn.Module):
    """hidden 1024 -> 4096 -> GELU -> 54. Each output dim = one probe."""

    def __init__(self, hidden_size, up_size, k):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, up_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(up_size, k)

    def forward(self, h):
        return self.fc2(self.act(self.fc1(h)))


# ===================================================================== Muon
@torch.no_grad()
def _zeropower_ns5(G, steps=5):
    """Newton-Schulz iteration -> approximate orthogonalization of G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon (momentum + Newton-Schulz orthogonalized update) for 2D weights."""

    def __init__(self, params, lr=5e-3, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.ndim != 2:
                    raise ValueError(f"Muon got a {g.ndim}-D param; only 2-D allowed")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                d = g.add(buf, alpha=mom) if group["nesterov"] else buf
                o = _zeropower_ns5(d, steps=group["ns_steps"])
                # rms-matched scale (modded-nanogpt convention)
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(o, alpha=-lr * scale)


def build_optimizers(model, head, muon_lr, adamw_lr):
    """Muon: every 2-D weight matrix except embeddings. AdamW: the rest."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon_params if (p.ndim == 2 and "embed" not in name.lower()) else adamw_params).append(p)
    muon_params += [head.fc1.weight, head.fc2.weight]
    adamw_params += [head.fc1.bias, head.fc2.bias]
    muon = Muon(muon_params, lr=muon_lr)
    adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=0.0)
    return muon, adamw, muon_params, adamw_params


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    # 0.1 floor: `total` is an ESTIMATE (max_tokens / est_tokens_per_step); if
    # the estimate is low the cosine completes early — never train at lr=0.
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))


def lr_lambda_cont(step, start_step, anchor_mult, end_step, rewarmup, floor=0.1):
    """Warm-restart schedule: linear re-warmup to `anchor_mult` (the original
    schedule's multiplier at the resume step — no LR jump vs the killed run),
    then cosine from anchor down to `floor` by `end_step` (projected epoch
    end). The re-warmup exists because stripped checkpoints carry no optimizer
    state: Muon momentum / AdamW moments rebuild during the ramp."""
    s = step - start_step
    if rewarmup > 0 and s < rewarmup:
        return anchor_mult * (s + 1) / rewarmup
    t = (s - rewarmup) / max(1, end_step - start_step - rewarmup)
    return floor + (anchor_mult - floor) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


# ================================================================== metrics
class SpearmanBuffer:
    """Reservoir of (target, pred) rows for per-probe Spearman at eval end."""

    def __init__(self, cap=200_000, seed=0):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.t, self.p, self.seen = [], [], 0

    def update(self, target, pred):
        n = target.shape[0]
        keep = min(n, max(0, self.cap - sum(x.shape[0] for x in self.t)))
        if keep > 0:
            idx = self.rng.choice(n, size=keep, replace=False) if keep < n else np.arange(n)
            self.t.append(target[idx])
            self.p.append(pred[idx])
        self.seen += n

    def spearman(self):
        if not self.t:
            return None
        t = np.concatenate(self.t, axis=0)
        p = np.concatenate(self.p, axis=0)

        def rank(a):
            # int8-quantized targets are heavily tied: use average ranks
            # (proper Spearman) when scipy is available; ordinal fallback.
            try:
                from scipy.stats import rankdata
                return np.stack([rankdata(a[:, j]) for j in range(a.shape[1])],
                                axis=1).astype(np.float64)
            except ImportError:
                order = np.argsort(a, axis=0)
                r = np.empty_like(order, dtype=np.float64)
                np.put_along_axis(r, order,
                                  np.arange(a.shape[0], dtype=np.float64)[:, None], axis=0)
                return r
        rt, rp = rank(t), rank(p)
        rt -= rt.mean(axis=0)
        rp -= rp.mean(axis=0)
        num = (rt * rp).sum(axis=0)
        den = np.sqrt((rt ** 2).sum(axis=0) * (rp ** 2).sum(axis=0))
        return num / np.maximum(den, 1e-12)


# ==================================================================== eval
@torch.no_grad()
def run_eval(val_shards, scores_dir, climbmix_dir, layer_idx, gemma_tok, qwen_tok,
             model, head, mean_l, std_l, zero_l, scale_l, concepts, families,
             bsz_docs, max_gemma_tokens, max_qwen_tokens, min_gemma_tokens,
             eval_tokens, device, log_fn):
    acc = te.R2Accumulator(K)
    sp = SpearmanBuffer()
    buf, n_tokens = [], 0
    stream = iter_docs_stacked(val_shards, scores_dir, climbmix_dir, layer_idx,
                                loop=False, log_fn=log_fn)

    def flush():
        nonlocal buf
        if not buf:
            return 0
        input_ids, attn_mask = te.collate_batch(buf, qwen_tok, device)
        hidden = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        feats, scores_i8 = te.gather_targets(buf, None, hidden)
        pred = head(feats).float().cpu().numpy()
        raw = te.dequantize(scores_i8, zero_l, scale_l)
        target = te.standardize(raw, mean_l, std_l)
        acc.update(target, pred)
        sp.update(target, pred)
        buf = []
        return raw.shape[0]

    for doc in stream:
        if n_tokens >= eval_tokens:
            break
        pd = te.process_doc(doc, gemma_tok, qwen_tok, max_gemma_tokens,
                             max_qwen_tokens, min_gemma_tokens, assert_tokens=False)
        if pd is None:
            continue
        buf.append(pd)
        if len(buf) >= bsz_docs:
            n_tokens += flush()
    n_tokens += flush()

    r2 = acc.r2()
    rho = sp.spearman()
    fam_groups = {}
    for c, name in enumerate(concepts):
        fam_groups.setdefault(families[c], []).append(r2[c])
    res = {
        "n_eval_tokens": n_tokens,
        "median_r2": float(np.median(r2)),
        "mean_r2": float(np.mean(r2)),
        "min_r2": float(np.min(r2)),
        "per_probe_r2": {n: float(v) for n, v in zip(concepts, r2)},
        "per_family_median_r2": {f: float(np.median(v)) for f, v in fam_groups.items()},
    }
    if rho is not None:
        res["median_spearman"] = float(np.median(rho))
        res["per_probe_spearman"] = {n: float(v) for n, v in zip(concepts, rho)}
    return res


# ============================================================== checkpoints
def save_ckpt(path, step, train_tokens, model, head, muon, adamw, args, meta, strip=False):
    state = {
        "step": step,
        "train_tokens": train_tokens,
        "encoder_state": model.state_dict(),
        "head_state": head.state_dict(),
        "args": vars(args),
        **meta,
    }
    if not strip:
        state["muon_state"] = muon.state_dict()
        state["adamw_state"] = adamw.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def hf_push(repo, layer, out_dir, log_fn, subdir=None):
    """Push stripped best.pt + metrics.jsonl to HF model repo <subdir>/
    (default layerXX/). Continuation runs MUST pass a distinct subdir so the
    original checkpoints are never overwritten."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True)
        sub = subdir or f"layer{layer:02d}"
        for fn in ("best_stripped.pt", "metrics.jsonl"):
            p = os.path.join(out_dir, fn)
            if os.path.exists(p):
                api.upload_file(path_or_fileobj=p, path_in_repo=f"{sub}/{fn}",
                                repo_id=repo, repo_type="model")
        log_fn(f"pushed {sub}/ to {repo}")
    except Exception as e:  # noqa: BLE001 — push is best-effort, never kills training
        log_fn(f"HF push FAILED (non-fatal): {e!r}")


# ===================================================================== main
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, required=True, choices=[6, 8, 14])
    p.add_argument("--scores", required=True, help="dir with stacked scores/tokens/docs + quant.json/corpus_stats.json/columns.json")
    p.add_argument("--climbmix-dir", required=True)
    p.add_argument("--train-shards", required=True)
    p.add_argument("--val-shards", required=True)
    p.add_argument("--model", default=te.QWEN_PRIMARY)
    p.add_argument("--gemma-model", default=te.GEMMA_TOKENIZER_DEFAULT)
    p.add_argument("--head-up", type=int, default=4096)
    p.add_argument("--muon-lr", type=float, default=5e-3)
    p.add_argument("--adamw-lr", type=float, default=1e-4)
    p.add_argument("--bsz-docs", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-tokens", type=float, default=1.3e9)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--plateau-delta", type=float, default=0.005)
    p.add_argument("--plateau-tokens", type=float, default=150e6)
    p.add_argument("--min-tokens-before-stop", type=float, default=200e6)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--max-gemma-tokens", type=int, default=2048)
    p.add_argument("--max-qwen-tokens", type=int, default=3072)
    p.add_argument("--min-gemma-tokens", type=int, default=64)
    p.add_argument("--assert-first-n-docs", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--est-tokens-per-step", type=float, default=27_500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb-project", default="stage7-oracle")
    p.add_argument("--wandb-entity", default="kaushikreddyxyz-")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--hf-repo", default="kaushikreddyxyz/oracle-encoders")
    p.add_argument("--hf-subdir", default=None,
                   help="repo subfolder for pushes (default layerXX/); "
                        "continuation runs must set e.g. layer06/cont1")
    p.add_argument("--cont-anchor-mult", type=float, default=None,
                   help="warm-restart: LR multiplier at the resume step "
                        "(original schedule's value there); enables the "
                        "continuation schedule when set with --cont-end-step")
    p.add_argument("--cont-end-step", type=int, default=None,
                   help="warm-restart: absolute step at which the continuation "
                        "cosine reaches the 0.1 floor (projected epoch end)")
    p.add_argument("--cont-rewarmup-steps", type=int, default=150,
                   help="warm-restart: linear LR re-warmup steps after resume "
                        "(optimizer state is absent in stripped checkpoints)")
    p.add_argument("--push-every-min", type=float, default=90.0)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--heartbeat-path", default=None)
    p.add_argument("--out", required=True)
    return p


def run_training(args, encoder_and_tok=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    layer_idx = LAYER_TO_IDX[args.layer]
    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    hb_path = args.heartbeat_path or f"/workspace/hb_L{args.layer}.txt"

    def log(msg):
        print(f"[oracleL{args.layer}] {msg}", flush=True)

    # ---- store metadata: per-layer slices of [3][54] arrays ----
    with open(os.path.join(args.scores, "quant.json")) as f:
        q = json.load(f)
    zero_l = np.array(q["zero"], dtype=np.float32)[layer_idx]
    scale_l = np.array(q["scale"], dtype=np.float32)[layer_idx]
    with open(os.path.join(args.scores, "corpus_stats.json")) as f:
        cs = json.load(f)
    mean_l = np.array(cs["mean"], dtype=np.float32)[layer_idx]
    std_l = np.maximum(np.array(cs["std"], dtype=np.float32)[layer_idx], 1e-6)
    with open(os.path.join(args.scores, "columns.json")) as f:
        cols = json.load(f)
    concepts, families = cols["concepts"], cols["families"]
    for name, arr in (("zero", zero_l), ("scale", scale_l), ("mean", mean_l), ("std", std_l)):
        if arr.shape != (K,):
            raise ValueError(f"quant/stats {name} slice has shape {arr.shape}, expected ({K},)")
    if len(concepts) != K or len(families) != K:
        raise ValueError("columns.json concepts/families must have 54 entries")

    train_shards = [int(x) for x in args.train_shards.split(",") if x]
    val_shards = [int(x) for x in args.val_shards.split(",") if x]
    if set(train_shards) & set(val_shards):
        raise ValueError(f"train/val overlap: {sorted(set(train_shards) & set(val_shards))}")

    # ---- model ----
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    if encoder_and_tok is not None:
        model, qwen_tok, model_name = encoder_and_tok
        model.to(args.device)
    else:
        model, qwen_tok, model_name = te.load_encoder(args.model, dtype, args.device)
    if qwen_tok.pad_token_id is None:  # te.load_encoder sets this; injected toks may not
        qwen_tok.pad_token = qwen_tok.eos_token
    gemma_tok = te.load_gemma_tokenizer(args.gemma_model)
    hidden_size = model.config.hidden_size
    head = OracleMLPHead(hidden_size, args.head_up, K).to(args.device)
    if dtype == torch.bfloat16:
        head = head.to(dtype)
    model.train()

    muon, adamw, muon_params, adamw_params = build_optimizers(
        model, head, args.muon_lr, args.adamw_lr)
    trainable = muon_params + adamw_params
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    log(f"model={model_name} hidden={hidden_size}; Muon params={n_muon/1e6:.1f}M "
        f"AdamW params={n_adamw/1e6:.1f}M; layer={args.layer} (idx {layer_idx})")

    est_total_steps = int(args.max_tokens / args.est_tokens_per_step)
    meta = {"layer": args.layer, "layer_idx": layer_idx, "mode": "perlayer-expA",
            "model_name": model_name, "hidden_size": hidden_size, "K": K,
            "concepts": concepts, "families": families, "head_up": args.head_up}

    start_step, train_tokens = 0, 0.0
    if args.resume and os.path.exists(args.resume):
        st = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(st["encoder_state"])
        head.load_state_dict(st["head_state"])
        if "muon_state" in st:
            muon.load_state_dict(st["muon_state"])
            adamw.load_state_dict(st["adamw_state"])
        start_step = st["step"]
        train_tokens = st.get("train_tokens", start_step * args.est_tokens_per_step)
        log(f"resumed from {args.resume} at step {start_step} ({train_tokens/1e6:.0f}M tokens)")

    # ---- wandb (live) ----
    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            name=args.wandb_name or f"oracle-L{args.layer}-perlayer",
                            config={**vars(args), "n_muon_params": n_muon,
                                    "n_adamw_params": n_adamw, "model_name": model_name},
                            resume="allow", id=None)
        except Exception as e:  # noqa: BLE001
            log(f"wandb init failed (continuing without): {e!r}")
            wb = None

    train_stream = iter_docs_stacked(train_shards, args.scores, args.climbmix_dir,
                                      layer_idx, loop=True, log_fn=log)

    best_metric, best_path = -1e9, os.path.join(args.out, "best.pt")
    last_path = os.path.join(args.out, "last.pt")
    eval_history = []  # (train_tokens, median_r2)
    n_docs_asserted, step = 0, start_step
    t0, last_hb, tokens_since_hb, last_loss = time.time(), 0.0, 0, float("nan")
    last_push = time.time()
    stop_reason = None

    while True:
        if train_tokens >= args.max_tokens:
            stop_reason = f"max tokens {args.max_tokens:.2e} reached"
            break
        if (time.time() - t0) / 3600.0 >= args.max_hours:
            stop_reason = f"wall clock {args.max_hours}h reached"
            break

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        accum_loss, n_accum_tokens = 0.0, 0
        for _ in range(args.grad_accum):
            buf = []
            while len(buf) < args.bsz_docs:
                doc = next(train_stream)
                assert_this = n_docs_asserted < args.assert_first_n_docs
                pd = te.process_doc(doc, gemma_tok, qwen_tok, args.max_gemma_tokens,
                                     args.max_qwen_tokens, args.min_gemma_tokens,
                                     assert_tokens=assert_this)
                if assert_this:
                    n_docs_asserted += 1
                if pd is None:
                    continue
                buf.append(pd)
            input_ids, attn_mask = te.collate_batch(buf, qwen_tok, args.device)
            ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if args.device.startswith("cuda") else te._nullcontext())
            with ctx:
                hidden = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            feats, scores_i8 = te.gather_targets(buf, None, hidden)
            pred = head(feats)  # bf16 in, bf16 out — do NOT cast (stage-7 dtype trap)
            raw = te.dequantize(scores_i8, zero_l, scale_l)
            target = te.standardize(raw, mean_l, std_l)
            target_t = torch.as_tensor(target, dtype=pred.dtype, device=args.device)
            loss = ((pred - target_t) ** 2).mean()
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            n_accum_tokens += scores_i8.shape[0]

        last_loss = accum_loss / args.grad_accum
        if not math.isfinite(last_loss):
            # break BEFORE stepping: weights stay at the last finite update,
            # so best.pt AND last.pt remain uncorrupted.
            stop_reason = f"NON-FINITE LOSS at step {step + 1} — stopped before applying update"
            break
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if args.cont_anchor_mult is not None and args.cont_end_step is not None:
            lr_mult = lr_lambda_cont(step, start_step, args.cont_anchor_mult,
                                     args.cont_end_step, args.cont_rewarmup_steps)
        else:
            lr_mult = lr_lambda(step, args.warmup_steps, est_total_steps)
        for g in muon.param_groups:
            g["lr"] = args.muon_lr * lr_mult
        for g in adamw.param_groups:
            g["lr"] = args.adamw_lr * lr_mult
        muon.step()
        adamw.step()
        step += 1
        train_tokens += n_accum_tokens
        tokens_since_hb += n_accum_tokens

        now = time.time()
        if now - last_hb >= 60.0:
            tok_s = tokens_since_hb / max(now - last_hb, 1e-6)
            te.heartbeat(hb_path, step=step, loss=last_loss, layer=args.layer,
                         train_tokens=train_tokens,
                         median_r2=(eval_history[-1][1] if eval_history else None),
                         tok_per_s=tok_s)
            if wb:
                wb.log({"train/loss": last_loss, "train/tokens": train_tokens,
                        "train/tok_per_s": tok_s, "train/lr_mult": lr_mult}, step=step)
            last_hb, tokens_since_hb = now, 0

        if step % args.eval_every == 0:
            model.eval()
            res = run_eval(val_shards, args.scores, args.climbmix_dir, layer_idx,
                           gemma_tok, qwen_tok, model, head, mean_l, std_l, zero_l,
                           scale_l, concepts, families, args.bsz_docs,
                           args.max_gemma_tokens, args.max_qwen_tokens,
                           args.min_gemma_tokens, args.eval_tokens, args.device, log)
            model.train()
            metric = res["median_r2"]
            eval_history.append((train_tokens, metric))
            entry = {"step": step, "train_tokens": train_tokens, "loss": last_loss, **res}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            log(f"step={step} tokens={train_tokens/1e6:.0f}M loss={last_loss:.4f} "
                f"median_r2={metric:.4f} spearman={res.get('median_spearman')}")
            if wb:
                wb.log({"eval/median_r2": metric, "eval/mean_r2": res["mean_r2"],
                        "eval/min_r2": res["min_r2"],
                        "eval/median_spearman": res.get("median_spearman"),
                        **{f"family/{k}": v for k, v in res["per_family_median_r2"].items()},
                        "train/tokens": train_tokens}, step=step)

            if metric > best_metric:
                best_metric = metric
                save_ckpt(best_path, step, train_tokens, model, head, muon, adamw, args, meta)
                save_ckpt(os.path.join(args.out, "best_stripped.pt"), step, train_tokens,
                          model, head, muon, adamw, args, meta, strip=True)
            save_ckpt(last_path, step, train_tokens, model, head, muon, adamw, args, meta)

            # token-based plateau
            if train_tokens >= args.min_tokens_before_stop:
                ref = None
                for tk, m in eval_history:
                    if tk <= train_tokens - args.plateau_tokens:
                        ref = m
                if ref is not None and (metric - ref) < args.plateau_delta:
                    stop_reason = (f"plateau: Δmedian_r2={metric - ref:.4f} < "
                                   f"{args.plateau_delta} over trailing "
                                   f"{args.plateau_tokens/1e6:.0f}M tokens")
                    break

            if (not args.no_push) and (time.time() - last_push) / 60.0 >= args.push_every_min:
                hf_push(args.hf_repo, args.layer, args.out, log, subdir=args.hf_subdir)
                last_push = time.time()

    log(f"STOP: {stop_reason} (step={step}, tokens={train_tokens/1e6:.0f}M, "
        f"best median_r2={best_metric:.4f})")
    save_ckpt(last_path, step, train_tokens, model, head, muon, adamw, args, meta)
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"final": True, "stop_reason": stop_reason, "step": step,
                             "train_tokens": train_tokens, "best_median_r2": best_metric}) + "\n")
    if not args.no_push:
        hf_push(args.hf_repo, args.layer, args.out, log, subdir=args.hf_subdir)
    te.heartbeat(hb_path, step=step, loss=last_loss, layer=args.layer,
                 train_tokens=train_tokens, median_r2=best_metric, tok_per_s=0,
                 done=True, stop_reason=stop_reason)
    if wb:
        wb.summary["best_median_r2"] = best_metric
        wb.summary["stop_reason"] = stop_reason
        wb.finish()
    return {"final_step": step, "train_tokens": train_tokens,
            "best_metric": best_metric, "stop_reason": stop_reason}


def main():
    args = build_argparser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
