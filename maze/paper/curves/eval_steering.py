#!/usr/bin/env python3
"""Activation steering of the BASE models on the standard distance-stratified eval set.

Same items as eval_distance.py: prompt = token-space maze A (+ its origin/target); target = maze B at a controlled
Hamming distance. No graft: the base model's activations are edited so that it should solve B instead of A.

A and B are spanning trees, so A -> B removes k connections and adds k. Every removed connection r is paired with an
added connection a (fixed random matching per item), and r's adjacency-list chunk `c1 <--> c2 ;` is rewritten into a's
at the four chunk tokens (sites A, <-->, B, ;), in resid_post of block `layer`, during the prompt pass only:
  none      no intervention (floor: the model follows its prompt)
  oracle    token-level ceiling: the two coordinate tokens of r's chunk are replaced by a's (no activation edit)
  dom       standard steering: x += alpha * (mu[site][a] - mu[site][r])      (difference-of-means steering vectors)
  patch     class-mean activation patch: x <- mu[site][a]
  sae       SAE feature steering: latents of r := 0, latents of a := value * typical, x += decoder(delta latents)
Classes are DIRECTED connections (first cell, second cell) = 4n(n-1) classes; statistics and the SAE come from 6,000
fresh mazes (seed 777) of the same grid size, per model.

  python eval_steering.py prep --run_dir <base run>
  python eval_steering.py run  --run_dir <base run> --conds none,oracle,dom:L0:a1,patch:L0,sae:L0:n4:v2 [--tune True]
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE))
import maze._pathsetup  # noqa
import json, re, time
from pathlib import Path
import fire, numpy as np, torch
import torch.nn.functional as F
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder
from maze.core.data import PretokDataset, pretokenize
from eval_ckpts import feat_idx, load_ckpt
import eval_distance as ED
import sae_lib

WORK = Path("/workspace/steer")
OFFSETS = (-3, -2, -1, 0)          # chunk tokens relative to the `;`: A, <-->, B, ;
SITES = ("A", "conn", "B", "semi")


# ----------------------------------------------------------------------------- model passes
def _block(blk, resid, H, dh):
    B, P, _ = resid.shape
    q, k, v = blk.attn.qkv(blk.ln1(resid)).split(blk.attn.d_inner, dim=-1)
    q, k, v = (t.view(B, P, H, dh).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=blk.attn.scale)
    resid = resid + blk.attn.out(y.transpose(1, 2).reshape(B, P, -1))
    return resid + blk.mlp(blk.ln2(resid)), k, v


@torch.no_grad()
def resid_post(model, toks, layer):
    """resid_post of block `layer` for right-padded rows (causal => padding never reaches real positions)."""
    H, dh = model.cfg.n_heads, model.cfg.d_head
    resid = model.embed(toks) + model.pos_embed(torch.arange(toks.shape[1], device=toks.device))[None]
    for i, blk in enumerate(model.blocks):
        resid, _, _ = _block(blk, resid, H, dh)
        if i == layer:
            return resid


@torch.no_grad()
def greedy_kv_hook(model, prompt, max_new, path_end_id, pad_id, hook_layer=None, hook=None):
    """eval_ckpts.greedy_kv without an injector; `hook(resid)->resid` edits resid_post of block hook_layer in the
    prompt pass (later blocks' K/V caches are built from the edited stream, so the edit persists through decoding)."""
    dev = prompt.device; B, P = prompt.shape; H, dh = model.cfg.n_heads, model.cfg.d_head
    pos = torch.arange(P + max_new, device=dev)
    resid = model.embed(prompt) + model.pos_embed(pos[:P])[None]
    caches = []
    for i, blk in enumerate(model.blocks):
        resid, k, v = _block(blk, resid, H, dh)
        K = torch.empty(B, H, P + max_new, dh, device=dev, dtype=k.dtype); K[:, :, :P] = k
        V = torch.empty(B, H, P + max_new, dh, device=dev, dtype=v.dtype); V[:, :, :P] = v
        caches.append((K, V))
        if hook is not None and i == hook_layer:
            resid = hook(resid)
    cur = model.unembed(model.ln_final(resid[:, -1])).argmax(-1)
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(max_new):
        out[:, step] = torch.where(done, torch.full_like(cur, pad_id), cur)
        done = done | (cur == path_end_id)
        if bool(done.all()) or step == max_new - 1:
            break
        t = P + step
        resid = model.embed(cur) + model.pos_embed(pos[t])[None]
        for i, blk in enumerate(model.blocks):
            q, k, v = blk.attn.qkv(blk.ln1(resid)).split(blk.attn.d_inner, dim=-1)
            K, V = caches[i]; K[:, :, t] = k.view(B, H, dh); V[:, :, t] = v.view(B, H, dh)
            y = F.scaled_dot_product_attention(q.view(B, H, 1, dh), K[:, :, : t + 1], V[:, :, : t + 1], scale=blk.attn.scale)
            resid = resid + blk.attn.out(y.reshape(B, -1))
            resid = resid + blk.mlp(blk.ln2(resid))
        cur = model.unembed(model.ln_final(resid)).argmax(-1)
    return out


# ----------------------------------------------------------------------------- chunks / classes
def feat_cells(f, n):
    if f < n * n:
        r, c = divmod(f, n); return (r, c), (r + 1, c)
    r, c = divmod(f - n * n, n); return (r, c), (r, c + 1)


def chunk_table(toks, semi_id, conn_id, id_to_coord, n):
    """toks [N, L] (numpy) -> semicolon positions [N, E] and directed class ids [N, E] (class = 2*feature + flipped)."""
    N, L = toks.shape
    m = np.zeros_like(toks, dtype=bool); m[:, 3:] = (toks[:, 3:] == semi_id) & (toks[:, 1:-2] == conn_id)
    rows, pos = np.nonzero(m); E = len(pos) // N
    assert E * N == len(pos) and (np.bincount(rows, minlength=N) == E).all()
    lut_r = np.full(max(id_to_coord) + 1, -1); lut_c = lut_r.copy()
    for i, (r, c) in id_to_coord.items(): lut_r[i] = r; lut_c[i] = c
    ta, tb = toks[rows, pos - 3], toks[rows, pos - 1]
    ra, ca, rb, cb = lut_r[ta], lut_c[ta], lut_r[tb], lut_c[tb]
    assert (ra >= 0).all() and (rb >= 0).all()
    horiz = ra == rb
    f = np.where(horiz, n * n + ra * n + np.minimum(ca, cb), np.minimum(ra, rb) * n + ca)
    flipped = np.where(horiz, ca > cb, ra > rb)
    return pos.reshape(N, E), (2 * f + flipped).reshape(N, E)


def load_vocab(run_dir):
    cfg = ConfigHolder.load(json.load(open(Path(run_dir) / "config_holder.json")))
    vocab = cfg.tokenizer.get_vocab()
    coord_re = re.compile(r"^\((\d+),(\d+)\)$")
    id_to_coord = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items() if (m := coord_re.match(t))}
    return cfg, vocab, id_to_coord


# ----------------------------------------------------------------------------- SAE latent tables (SAEs themselves: sae_lib.py)
def sae_tables(sae, scale, Xs, y, n_cls, M, device, n_fit=200000, seed=0, sel_steps=300):
    """Per site: top-M latents per directed class (logistic weight x class-mean activation, active latents only)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sel = torch.randperm(len(y), generator=g)[:n_fit].to(device)
    with torch.no_grad():
        acts = sae.encode(Xs[sel].float() / scale); yy = y[sel]
    n_tr = int(0.8 * len(yy)); lin = torch.nn.Linear(acts.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(sel_steps):
        loss = F.cross_entropy(lin(acts[:n_tr]), yy[:n_tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((lin(acts[n_tr:]).argmax(-1) == yy[n_tr:]).float().mean())
        onehot = F.one_hot(yy, n_cls).float(); cnt = onehot.sum(0).clamp(min=1)
        mean_act = (onehot.T @ acts) / cnt[:, None]                                  # [n_cls, n_lat]
        on = (acts > 0).float(); tp = onehot.T @ on; fp = on.sum(0)[None] - tp; fn = cnt[:, None] - tp
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        score = torch.where(mean_act > 0, lin.weight * mean_act, torch.full_like(mean_act, -1e9))
        top = score.topk(M, dim=1).indices; val = mean_act.gather(1, top)
        val = torch.where(score.gather(1, top) > -1e8, val, torch.zeros_like(val))   # classes with < M active latents
        # classifier-free alternative: latents whose class-mean activation most exceeds their mean over all classes
        present = (cnt > 1).float()[:, None]; glob = (mean_act * present).sum(0, keepdim=True) / present.sum()
        score_md = torch.where(mean_act > 0, mean_act - glob, torch.full_like(mean_act, -1e9))
        top_md = score_md.topk(M, dim=1).indices; val_md = mean_act.gather(1, top_md)
        val_md = torch.where(score_md.gather(1, top_md) > -1e8, val_md, torch.zeros_like(val_md))
    return dict(lat=top, val=val, lat_md=top_md, val_md=val_md, acc=acc, best_f1=f1.max(1).values)


# ----------------------------------------------------------------------------- prep
def prep(run_dir: str, layers="0", n_train: int = 12000, train_seed: int = 777,
         saes="relu-m1.5-l0.03", sae_steps: int = 10000, n_lat_max: int = 16, sel_steps: int = 300, tag: str = "", part: str = ""):
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run = Path(run_dir); n = int(re.search(r"dfs-g(\d+)-", run.name).group(1))
    out = WORK / run.parent.name / f"stats{tag}{part}.pt"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg, vocab, id_to_coord = load_vocab(run)
    layers = [int(x) for x in (layers if isinstance(layers, (list, tuple)) else str(layers).split(","))]
    sae_specs = [x for x in (saes if isinstance(saes, (list, tuple)) else str(saes).split(",")) if x]
    f_exp = 4 * n * (n - 1) + n * n                                    # directed connections + cell identities expected at this site
    pre = WORK / f"train-g{n}-s{train_seed}-n{n_train}.pt"
    if not pre.exists():
        ds = MazeDataset.from_config(MazeDatasetConfig(name="steertrain", grid_n=n, n_mazes=n_train, maze_ctor=LatticeMazeGenerators.gen_dfs, seed=train_seed),
                                     do_generate=True, load_local=False, save_local=False, gen_parallel=False, verbose=False)
        pretokenize(ds, cfg.maze_tokenizer, vocab, pre, bos_id=int(cfg.tokenizer.bos_token_id), workers=8)
    pt = PretokDataset(pre); seqs = [pt[i][0] for i in range(len(pt))]
    pad_id = vocab[SPECIAL_TOKENS.PADDING]; L = max(len(s) for s in seqs)
    toks = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs): toks[i, : len(s)] = s
    real = toks != pad_id
    semis, cls = chunk_table(toks.numpy(), vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE], vocab[SPECIAL_TOKENS.CONNECTOR], id_to_coord, n)
    n_cls = 4 * n * n                                                            # 2 * (2 n^2) incl. never-occurring boundary classes
    model, _ = load_ckpt(run / "model.final.pt", None, device)
    semis_t = torch.tensor(semis, device=device); y = torch.tensor(cls, device=device).flatten()
    stats = dict(grid_n=n, layers=layers, n_cls=n_cls, sites=SITES, mu={}, sae={}, log={})
    t0 = time.time()
    for layer in layers:
        R = torch.cat([resid_post(model, toks[lo: lo + 500].to(device), layer).half() for lo in range(0, len(toks), 500)])    # [N, L, d]
        rows = torch.arange(len(toks), device=device)[:, None].expand_as(semis_t)
        site_X = [R[rows, semis_t + o].reshape(-1, R.shape[-1]) for o in OFFSETS]                                           # 4 x [N*E, d]
        onehot = F.one_hot(y, n_cls).half(); cnt = onehot.float().sum(0)
        glob = [X.float().mean(0) for X in site_X]
        mu = torch.stack([torch.where(cnt[:, None] > 0, (onehot.float().T @ X.float()) / cnt.clamp(min=1)[:, None], g[None]) for X, g in zip(site_X, glob)])
        stats["mu"][layer] = mu.cpu()                                                                                       # [4, n_cls, d]
        if layer == layers[0]:
            Xall = R[real.to(device)]                                                                                      # rows ordered by maze
            n_ho = max(100, int(0.05 * len(toks))); tho = toks[-n_ho:].to(device); real_ho = tho != pad_id
            ps_id = vocab[SPECIAL_TOKENS.PATH_START]; after = (tho == ps_id).cumsum(1) > 0; loss_mask = after & real_ho & torch.roll(real_ho, -1, 1)
            loss_mask[:, -1] = False
            for spec in sae_specs:
                sd = sae_lib.train(Xall, spec, f_exp, device, steps=sae_steps)
                sae = sae_lib.build(sd, device); scale = sd["scale"]
                tabs = [sae_tables(sae, scale, X, y, n_cls, min(n_lat_max, sd["n_lat"]), device, sel_steps=sel_steps) for X in site_X]
                lrec = sae_lib.loss_recovered(model, _block, tho, loss_mask, real_ho, layer + 1, sae, scale)       # resid_post L = resid_pre L+1
                stats["sae"][spec] = dict(sd=sd, layer=layer, lat=torch.stack([t["lat"] for t in tabs]).cpu(), val=torch.stack([t["val"] for t in tabs]).cpu())
                stats["log"][spec] = dict(n_lat=sd["n_lat"], f_exp=f_exp, heldout=sd["heldout"], converged_ratio=sd["converged_ratio"], curve=sd["curve"], **lrec,
                                          logit_acc={s_: t["acc"] for s_, t in zip(SITES, tabs)},
                                          frac_cls_f1_gt_0p9={s_: float((t["best_f1"][cnt > 0] > 0.9).float().mean()) for s_, t in zip(SITES, tabs)})
                print(f"[prep g{n}] SAE {spec}: n_lat {sd['n_lat']} heldout {sd['heldout']} loss_recovered {lrec['loss_recovered']:.3f} "
                      f"| connection decodable at B {stats['log'][spec]['logit_acc']['B']:.2f}", flush=True)
                del sae
            del Xall
        del R, site_X
    stats["log"]["min_class_count"] = int(cnt[cnt > 0].min()); stats["log"]["seconds"] = time.time() - t0
    torch.save(stats, out)
    json.dump(stats["log"], open(out.with_suffix(".json"), "w"), indent=1)
    print(f"[prep g{n}] done {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


# ----------------------------------------------------------------------------- run
def build_pairs(S, semis, cls, n, seed):
    """Per item: removed connections (their chunk in the prompt) matched to added connections. Returns flat arrays
    item, semicolon position, source class, target class; plus the oracle prompt edits (item, pos of A tok, pos of B tok, cells)."""
    rng = np.random.default_rng([seed, n, 17])
    maze, conn_a, conn_b = S["maze"], S["conn_a"], S["conn_b"]
    item, pos, src, tgt = [], [], [], []
    feat_of = cls // 2
    for j, i in enumerate(maze):
        rem = np.flatnonzero(conn_a[i] & ~conn_b[j]); add = np.flatnonzero(conn_b[j] & ~conn_a[i])
        assert len(rem) == len(add)
        if not len(rem): continue
        add = add[rng.permutation(len(add))]; flip = rng.integers(0, 2, len(add))
        where = {int(f): e for e, f in enumerate(feat_of[i])}
        for r, a, fl in zip(rem, add, flip):
            e = where[int(r)]
            item.append(j); pos.append(semis[i, e]); src.append(cls[i, e]); tgt.append(2 * int(a) + int(fl))
    return tuple(np.array(x, dtype=np.int64) for x in (item, pos, src, tgt))


def make_hook(cond, stats, bi, pos, src, tgt, device):
    kind = cond[0]; args = {a[0]: a[1:] for a in cond[1:]}
    layer = int(args.get("L", "0"))
    if kind in ("dom", "patch"):
        mu = stats["mu"][layer].to(device).float(); alpha = float(args.get("a", "1"))
        def hook(resid):
            for s, o in enumerate(OFFSETS):
                x = resid[bi, pos + o]
                resid[bi, pos + o] = (x + alpha * (mu[s][tgt] - mu[s][src])) if kind == "dom" else (x + alpha * (mu[s][tgt] - x))
            return resid
        return layer, hook
    if kind == "sae":
        sd = stats["sae"][args.get("s", "relu-m1.5-l0.03")]; assert sd["layer"] == layer
        sae = sae_lib.build(sd["sd"], device)
        nlat = int(args.get("n", "16")); value = float(args.get("v", "1")); scale = sd["sd"]["scale"]
        md = "_md" if args.get("t", "logit") == "md" else ""                     # latent table: logistic (default) or mean-difference
        lat = sd["lat" + md][:, :, :nlat].to(device); val = sd["val" + md][:, :, :nlat].to(device)
        sites = [SITES.index(x) for x in ("A", "B")] if args.get("o", "all") == "AB" else range(4)
        def hook(resid):
            ar = torch.arange(len(bi), device=device)[:, None]
            for s in sites:
                o = OFFSETS[s]; x = resid[bi, pos + o] / scale; a = sae.encode(x); a2 = a.clone()
                a2[ar, lat[s][src]] = 0.0
                a2[ar, lat[s][tgt]] = value * val[s][tgt]
                resid[bi, pos + o] = (x + (a2 - a) @ sae.W_dec) * scale
            return resid
        return layer, hook
    raise ValueError(kind)


def run(run_dir: str, conds: str, tune: bool = False, out_dir: str = "/workspace/steer_out", batch: int = 1000,
        n_mazes: int = 3000, seed: int = 4321, tune_mazes: int = 300, tune_seed: int = 999, tag: str = "", saes: str = "", sel_steps: int = 300, n_train: int = 12000, part: str = ""):
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run_ = Path(run_dir); n = int(re.search(r"dfs-g(\d+)-", run_.name).group(1)); seed_tag = run_.parent.name
    cfg, vocab, id_to_coord = load_vocab(run_)
    set_path = ED.build(n, str(run_ / "config_holder.json"), n_mazes=tune_mazes if tune else n_mazes, seed=tune_seed if tune else seed)
    S = dict(np.load(set_path)); sha = str(S["sha"])
    stats = torch.load(prep(run_dir, saes=saes, sel_steps=sel_steps, tag=tag, part=part, n_train=n_train), weights_only=False)
    model, _ = load_ckpt(run_ / "model.final.pt", None, device)
    semis, cls = chunk_table(S["prompts"], vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE], vocab[SPECIAL_TOKENS.CONNECTOR], id_to_coord, n)
    item, ppos, src, tgt = build_pairs(S, semis, cls, n, seed)
    coord_to_id = {c: i for i, c in id_to_coord.items()}
    pe_id, pad_id = int(S["path_end_id"]), int(S["pad_id"]); maze, conn_a, conn_b = S["maze"], S["conn_a"], S["conn_b"]
    origin_c, target_c, lv = S["origin"], S["target"], S["level"]; N = len(maze); max_new = n * n + 2
    starts = np.searchsorted(item, np.arange(0, N + 1))                                  # item is sorted
    # KV cache is 12 layers x 2 x 256 floats per position per item: cap the batch to ~45% of this GPU's memory
    per_item = 12 * 2 * 256 * 4 * (int(S["prompts"].shape[1]) + max_new) * 1.6
    batch = max(200, min(int(batch), int(0.45 * torch.cuda.get_device_properties(0).total_memory / per_item)))
    conds = conds if isinstance(conds, (list, tuple)) else str(conds).split(",")
    for cname in conds:
        out = Path(out_dir) / ("tune" if tune else "eval") / (cname.replace(":", "_") + tag) / f"g{n}__base__{seed_tag}.npz"
        if out.exists(): continue
        cond = cname.split(":"); t0 = time.time()
        solved_b = np.zeros(N, bool); solved_a = np.zeros(N, bool); reached = np.zeros(N, bool)
        nbad_b = np.zeros(N, np.int16); nbad_a = np.zeros(N, np.int16); malformed = np.zeros(N, bool)
        for lo in range(0, N, batch):
            hi = min(N, lo + batch); prompts = torch.tensor(S["prompts"][maze[lo:hi]], device=device)
            a, b = starts[lo], starts[hi]
            bi = torch.tensor(item[a:b] - lo, device=device); pp = torch.tensor(ppos[a:b], device=device)
            sc = torch.tensor(src[a:b], device=device); tg = torch.tensor(tgt[a:b], device=device)
            layer, hook = None, None
            if cond[0] == "oracle":
                c1 = np.array([coord_to_id[feat_cells(int(t) // 2, n)[int(t) % 2]] for t in tgt[a:b]], dtype=np.int64)
                c2 = np.array([coord_to_id[feat_cells(int(t) // 2, n)[1 - int(t) % 2]] for t in tgt[a:b]], dtype=np.int64)
                if len(c1):
                    prompts[bi, pp - 3] = torch.tensor(c1, device=device); prompts[bi, pp - 1] = torch.tensor(c2, device=device)
            elif cond[0] != "none" and len(bi):
                layer, hook = make_hook(cond, stats, bi, pp, sc, tg, device)
            gen = greedy_kv_hook(model, prompts, max_new, pe_id, pad_id, layer, hook).cpu().numpy()
            for k in range(hi - lo):
                j = lo + k; i = maze[j]; g = gen[k].tolist()
                for stop in (pe_id, pad_id):
                    if stop in g: g = g[: g.index(stop)]
                malformed[j] = any(t not in id_to_coord for t in g)
                coords = [tuple(origin_c[i])] + [id_to_coord[t] for t in g if t in id_to_coord]; tgt_c = tuple(target_c[i])
                r, bb = ED.judge(coords, conn_b[j], n, tgt_c); _, ba = ED.judge(coords, conn_a[i], n, tgt_c)
                reached[j] = r; nbad_b[j] = bb; nbad_a[j] = ba
                solved_b[j] = r and bb == 0 and not malformed[j]; solved_a[j] = r and ba == 0 and not malformed[j]
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, grid_n=n, cond="base", method=cname, seed_tag=seed_tag, run=run_.name, set_sha=sha,
                            solved_graft=solved_b, solved_token=solved_a, reached=reached, n_bad_graft=nbad_b, n_bad_token=nbad_a, malformed=malformed)
        ch = ~S["path_same"]
        print(f"[g{n} {seed_tag} {cname}] {time.time() - t0:.0f}s | solved vs target maze by level: "
              + " ".join(f"{solved_b[lv == k].mean():.2f}" for k in range(int(lv.max()) + 1))
              + f" | route-changed mean {solved_b[ch & (lv > 0)].mean():.3f}", flush=True)
    print("RUN_DONE", flush=True)


def oracle_cells_doc():
    """Directed class 2f+0 = (upper/left cell first); 2f+1 = flipped. feat_cells(f)[cls % 2] is the first token's cell."""


if __name__ == "__main__":
    fire.Fire(dict(prep=prep, run=run))
