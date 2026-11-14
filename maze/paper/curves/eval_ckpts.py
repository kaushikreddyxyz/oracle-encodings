#!/usr/bin/env python3
"""Score every checkpoint of one training run (torch-native, batched).

Per checkpoint and per world (true; phantom for grafted models):
  teacher-forced along the world's shortest path:
    p_sp       mass on a shortest-path continuation (at the target: <PATH_END>)
    p_valid    mass on other lattice neighbours with no wall between
    p_invalid  1 - p_sp - p_valid
  greedy rollout from the origin token:
    solved     reached the target AND every move contiguous + wall-free
    reached    final coordinate == target
    clean      no wall crossing / jump before the rollout ended
    valid_move_frac, len_ratio (rollout length / shortest length, solved only)
The eval set (n_mazes fresh mazes of this grid size, seed `seed`, + phantom
edits) is built once per grid and cached under --eval_dir.

  python maze/paper/curves/eval_ckpts.py --run_dir <run> --grid_n 6
Outputs <run>/eval/metrics.json (list of rows) and <run>/eval/per_maze.npz.
"""
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401

import json
import re
import time
from collections import deque
from pathlib import Path

import fire
import numpy as np
import torch
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder

from maze.core.data import PretokDataset, pretokenize
from maze.core.graft import as_graft, load_injector
from maze.core.tl_bridge import load_torch_ckpt


# ---- lattice helpers (feature layout as in maze/core/injection.py) ----------
def feat_idx(a, b, n):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return n * n + r1 * n + min(c1, c2)
    return min(r1, r2) * n + c1


def lattice_nbrs(cell, n):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            yield (rr, cc)


def open_nbrs(conn, cell, n):
    return [nb for nb in lattice_nbrs(cell, n) if conn[feat_idx(cell, nb, n)]]


def dist_to(conn, dst, n):
    d = {dst: 0}
    q = deque([dst])
    while q:
        u = q.popleft()
        for v in open_nbrs(conn, u, n):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def shortest_path(conn, src, dst, n):
    d = dist_to(conn, dst, n)
    if src not in d:
        return None
    path = [src]
    while path[-1] != dst:
        cur = path[-1]
        path.append(min(nb for nb in open_nbrs(conn, cur, n) if nb in d and d[nb] == d[cur] - 1))
    return path


def internal_mask(n):
    m = np.zeros(2 * n * n, dtype=bool)
    for f in range(2 * n * n):
        if f < n * n:
            m[f] = (f // n) < n - 1
        else:
            m[f] = ((f - n * n) % n) < n - 1
    return m


def make_phantom(conn, path, rng, n, n_walls=2, n_doors=2, tries=400):
    o, t = path[0], path[-1]
    path_feats = [feat_idx(path[i], path[i + 1], n) for i in range(len(path) - 1)]
    closed = [f for f in np.flatnonzero(internal_mask(n)) if not conn[f]]
    for nw, nd, need_all in ((n_walls, n_doors, True), (n_walls, n_doors, False), (1, 1, True)):
        nw = min(nw, len(path_feats))
        for _ in range(tries):
            walls = rng.choice(path_feats, nw, replace=False)
            doors = rng.choice(closed, nd, replace=False)
            c2 = conn.copy()
            c2[walls] = False
            c2[doors] = True
            p2 = shortest_path(c2, o, t, n)
            if p2 is None:
                continue
            used = {feat_idx(p2[i], p2[i + 1], n) for i in range(len(p2) - 1)}
            if need_all and sum(int(d) in used for d in doors) < nd:
                continue
            return c2, [int(w) for w in walls], [int(d) for d in doors], p2
    return None


# ---- eval set ------------------------------------------------------------------
class EvalSet:
    """Fresh mazes of one grid size + phantom edits, with everything the scorer
    needs precomputed (token tensors, category masks per position)."""

    def __init__(self, cfg: ConfigHolder, grid_n: int, eval_dir: Path, n_mazes: int,
                 seed: int, n_walls: int, n_doors: int, device):
        self.n = grid_n
        self.device = device
        vocab = cfg.tokenizer.get_vocab()
        # masks are over-allocated and sliced to the model's logit width at scoring
        # time (the HF wrapper's get_vocab() can carry an extra id the model lacks)
        self.vocab_size = max(vocab.values()) + 2
        self.pad_id = vocab[SPECIAL_TOKENS.PADDING]
        self.path_start_id = vocab[SPECIAL_TOKENS.PATH_START]
        self.path_end_id = vocab[SPECIAL_TOKENS.PATH_END]
        coord_re = re.compile(r"^\((\d+),(\d+)\)$")
        self.id_to_coord = {i: (int(m.group(1)), int(m.group(2)))
                            for t, i in vocab.items() if (m := coord_re.match(t))}
        self.coord_to_id = {c: i for i, c in self.id_to_coord.items()}

        d = eval_dir / f"g{grid_n}-s{seed}-n{n_mazes}"
        d.mkdir(parents=True, exist_ok=True)
        pre = d / "pretok.pt"
        if not pre.exists():
            ds_cfg = MazeDatasetConfig(name="evalset", grid_n=grid_n, n_mazes=n_mazes,
                                       maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed)
            dataset = MazeDataset.from_config(ds_cfg, do_generate=True, load_local=False,
                                              save_local=False, gen_parallel=False, verbose=False)
            pretokenize(dataset, cfg.maze_tokenizer, vocab, pre,
                        bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
        pretok = PretokDataset(pre)
        rows = [pretok[i] for i in range(len(pretok))]
        self.conn = np.stack([c.numpy() for _, c, _ in rows]).astype(bool)
        self.end_pos = np.array([int(e) for _, _, e in rows])
        self.seqs = [s.tolist() for s, _, _ in rows]
        n = grid_n
        self.origin_pos, self.true_path = [], []
        for s in self.seqs:
            ps = s.index(self.path_start_id)
            pe = s.index(self.path_end_id)
            self.origin_pos.append(ps + 1)
            self.true_path.append([self.id_to_coord[t] for t in s[ps + 1: pe]])
        self.origin_pos = np.array(self.origin_pos)

        edits_f = d / f"phantom_w{n_walls}_d{n_doors}.json"
        if edits_f.exists():
            ed = json.load(open(edits_f))
        else:
            rng = np.random.default_rng(seed)
            ed = {}
            for i in range(len(self.seqs)):
                ph = make_phantom(self.conn[i], self.true_path[i], rng, n, n_walls, n_doors)
                if ph is not None:
                    c2, walls, doors, p2 = ph
                    ed[str(i)] = dict(walls=walls, doors=doors, path=[list(c) for c in p2])
            json.dump(ed, open(edits_f, "w"))
        self.phantom = {}
        for k, v in ed.items():
            i = int(k)
            c2 = self.conn[i].copy()
            c2[v["walls"]] = False
            c2[v["doors"]] = True
            self.phantom[i] = (c2, [tuple(c) for c in v["path"]])
        self.worlds = {
            "true": [(i, self.conn[i], self.true_path[i]) for i in range(len(self.seqs))],
            "phantom": [(i, c2, p2) for i, (c2, p2) in sorted(self.phantom.items())],
        }
        self._tf_cache = {}
        self.semi_id = vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]

    def dummify_(self):
        """`dummy` data variant (maze/core/data.py): blank the adjacency list to `;`
        so the graft is the only source of connectivity. Call before any scoring."""
        assert not self._tf_cache
        for s, e in zip(self.seqs, self.end_pos):
            s[1: int(e)] = [self.semi_id] * (int(e) - 1)

    # ---- teacher-forced batches (built once per world) ----------------------
    def tf_batches(self, world, batch):
        if world in self._tf_cache:
            return self._tf_cache[world]
        out = []
        items = self.worlds[world]
        V = self.vocab_size
        for lo in range(0, len(items), batch):
            chunk = items[lo: lo + batch]
            seqs, metas = [], []
            for i, conn, path in chunk:
                op = int(self.origin_pos[i])
                prompt = self.seqs[i][: op + 1]
                seqs.append(prompt + [self.coord_to_id[c] for c in path[1:]] + [self.path_end_id])
                metas.append((i, op, len(path)))
            L = max(len(s) for s in seqs)
            toks = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
            for k, s in enumerate(seqs):
                toks[k, : len(s)] = torch.tensor(s)
            attn = (torch.arange(L)[None, :] < torch.tensor([len(s) for s in seqs])[:, None]).long()
            conn_t = torch.tensor(np.stack([c for _, c, _ in chunk]), dtype=torch.bool)
            end_t = torch.tensor([int(self.end_pos[i]) for i, _, _ in chunk], dtype=torch.long)
            # category masks per scored position
            bidx, pidx, sp_rows, val_rows, maze_of_pos = [], [], [], [], []
            for k, ((i, conn, path), (_, op, plen)) in enumerate(zip(chunk, metas)):
                target = path[-1]
                dist = dist_to(conn, target, self.n)
                for step, cell in enumerate(path):
                    sp = np.zeros(V, dtype=bool)
                    va = np.zeros(V, dtype=bool)
                    nb_open = open_nbrs(conn, cell, self.n)
                    if cell == target:
                        sp[self.path_end_id] = True
                        for nb in nb_open:
                            va[self.coord_to_id[nb]] = True
                    else:
                        for nb in nb_open:
                            if nb in dist and dist[nb] == dist[cell] - 1:
                                sp[self.coord_to_id[nb]] = True
                            else:
                                va[self.coord_to_id[nb]] = True
                    bidx.append(k); pidx.append(op + step); sp_rows.append(sp); val_rows.append(va)
                    maze_of_pos.append(i)
            out.append(dict(
                toks=toks.to(self.device), attn=attn.to(self.device),
                conn=conn_t.to(self.device), end=end_t.to(self.device),
                bidx=torch.tensor(bidx, device=self.device), pidx=torch.tensor(pidx, device=self.device),
                sp=torch.tensor(np.stack(sp_rows), device=self.device),
                va=torch.tensor(np.stack(val_rows), device=self.device),
                maze_of_pos=np.array(maze_of_pos),
            ))
        self._tf_cache[world] = out
        return out


# ---- model wrapper ----------------------------------------------------------------
class Scorer:
    def __init__(self, model, injector, device):
        self.model, self.injector, self.device = model, injector, device
        self.layer = int(injector.layer) if injector is not None else 0

    @torch.no_grad()
    def logits(self, toks, attn, conn, end):
        delta = None
        if self.injector is not None:
            g = as_graft(self.injector)   # any code: static (edge/onion/vertex) or dynamic (ego)
            delta = g.delta(g.code.features(toks, attn, conn, end))
        return self.model(toks, attention_mask=attn, inject_delta=delta,
                          inject_layer=self.layer, return_type="logits", padding_mode="right")


@torch.no_grad()
def score_tf(scorer, es, world, batch):
    """Per-maze mean (p_sp, p_valid) over that maze's path positions."""
    n_m = len(es.seqs)
    sums = np.zeros((n_m, 2)); cnt = np.zeros(n_m)
    for b in es.tf_batches(world, batch):
        probs = scorer.logits(b["toks"], b["attn"], b["conn"], b["end"]).float().softmax(-1)
        p = probs[b["bidx"], b["pidx"]]  # [P, V]
        V = p.shape[-1]
        psp = (p * b["sp"][:, :V]).sum(-1).cpu().numpy()
        pva = (p * b["va"][:, :V]).sum(-1).cpu().numpy()
        np.add.at(sums[:, 0], b["maze_of_pos"], psp)
        np.add.at(sums[:, 1], b["maze_of_pos"], pva)
        np.add.at(cnt, b["maze_of_pos"], 1)
    m = cnt > 0
    per_maze = np.full((n_m, 2), np.nan)
    per_maze[m] = sums[m] / cnt[m, None]
    return per_maze, int(cnt.sum())


@torch.no_grad()
def rollout(scorer, es, world, batch, max_new, kv=True):
    """Greedy rollout from the prompt (through the origin token) in every maze of
    `world`; judged against that world's connectivity. Returns per-maze arrays."""
    items = es.worlds[world]
    n_m = len(es.seqs)
    solved = np.full(n_m, np.nan); reached = np.full(n_m, np.nan); clean = np.full(n_m, np.nan)
    vmf = np.full(n_m, np.nan); lenr = np.full(n_m, np.nan)
    for lo in range(0, len(items), batch):
        chunk = items[lo: lo + batch]
        B = len(chunk)
        prompts = [es.seqs[i][: int(es.origin_pos[i]) + 1] for i, _, _ in chunk]
        L0 = max(len(p) for p in prompts)
        toks = torch.full((B, L0 + max_new), es.pad_id, dtype=torch.long, device=es.device)
        for k, p in enumerate(prompts):
            toks[k, : len(p)] = torch.tensor(p, device=es.device)
        cur = torch.tensor([len(p) for p in prompts], device=es.device)
        conn_t = torch.tensor(np.stack([c for _, c, _ in chunk]), dtype=torch.bool, device=es.device)
        end_t = torch.tensor([int(es.end_pos[i]) for i, _, _ in chunk], dtype=torch.long, device=es.device)
        done = torch.zeros(B, dtype=torch.bool, device=es.device)
        ar = torch.arange(B, device=es.device)
        same_len = all(len(p) == L0 for p in prompts)
        if kv and same_len:
            gen_kv = greedy_kv(scorer.model, scorer.injector, scorer.layer, toks[:, :L0], conn_t, end_t,
                               max_new, es.path_end_id, es.pad_id)
            toks[:, L0:] = gen_kv
            cur = torch.full((B,), L0 + max_new, device=es.device)
        else:
            for _ in range(max_new):
                W = int(cur.max())
                attn = (torch.arange(W, device=es.device)[None, :] < cur[:, None]).long()
                lg = scorer.logits(toks[:, :W], attn, conn_t, end_t)
                nxt = lg[ar, cur - 1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, es.pad_id), nxt)
                toks[ar, cur] = torch.where(done, toks[ar, cur], nxt)
                cur = cur + (~done).long()
                done = done | (nxt == es.path_end_id)
                if bool(done.all()):
                    break
        toks_c = toks.cpu().numpy(); cur_c = cur.cpu().numpy()
        for k, (i, conn, path) in enumerate(chunk):
            start = len(prompts[k])
            gen = toks_c[k, start: cur_c[k]].tolist()
            if es.path_end_id in gen:
                gen = gen[: gen.index(es.path_end_id)]
            elif es.pad_id in gen:
                gen = gen[: gen.index(es.pad_id)]
            coords = [path[0]]
            ok = True; valid = 0; total = 0
            for t in gen:
                c = es.id_to_coord.get(t)
                if c is None:
                    ok = False; break
                total += 1
                if abs(c[0] - coords[-1][0]) + abs(c[1] - coords[-1][1]) == 1 and conn[feat_idx(coords[-1], c, es.n)]:
                    valid += 1
                else:
                    ok = False
                coords.append(c)
            r = float(coords[-1] == path[-1] and len(coords) > 1)
            reached[i] = r; clean[i] = float(ok); solved[i] = float(ok and r == 1.0)
            vmf[i] = valid / total if total else 0.0
            if ok and r == 1.0:
                lenr[i] = (len(coords) - 1) / max(1, len(path) - 1)
    return dict(solved=solved, reached=reached, clean=clean, valid_move_frac=vmf, len_ratio=lenr)


@torch.no_grad()
def greedy_kv(model, injector, layer, prompt, conn, end_pos, max_new, path_end_id, pad_id):
    """Greedy decoding with a KV cache. `prompt` is [B, P] with NO padding (every
    row the same length, true for spanning-tree mazes of one grid size). Mirrors
    MazeGPT.forward(padding_mode="right") block by block. Returns [B, max_new]
    generated ids (pad after <PATH_END>)."""
    import torch.nn.functional as F
    dev = prompt.device
    B, P = prompt.shape
    H, dh = model.cfg.n_heads, model.cfg.d_head
    g = as_graft(injector)
    pos = torch.arange(P + max_new, device=dev)
    resid = model.embed(prompt) + model.pos_embed(pos[:P])[None]
    if g is not None:
        # the code decides what each position gets (static: rows(conn) from end_pos on;
        # dynamic: frame-dependent); the KV loop then extends it one token at a time
        attn = torch.ones(B, P, dtype=torch.long, device=dev)
        delta_full = g.delta(g.code.features(prompt, attn, conn, end_pos)).to(resid.dtype)
        state = g.code.init_state(prompt, attn, conn, end_pos)
        delta_row = None
    caches = []
    for i, blk in enumerate(model.blocks):
        if g is not None and i == layer:
            resid = resid + delta_full
        x = blk.ln1(resid)
        q, k, v = blk.attn.qkv(x).split(blk.attn.d_inner, dim=-1)
        q = q.view(B, P, H, dh).transpose(1, 2)
        k = k.view(B, P, H, dh).transpose(1, 2)
        v = v.view(B, P, H, dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=blk.attn.scale)
        resid = resid + blk.attn.out(y.transpose(1, 2).reshape(B, P, -1))
        resid = resid + blk.mlp(blk.ln2(resid))
        K = torch.empty(B, H, P + max_new, dh, device=dev, dtype=k.dtype); K[:, :, :P] = k
        V = torch.empty(B, H, P + max_new, dh, device=dev, dtype=v.dtype); V[:, :, :P] = v
        caches.append((K, V))
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
        if g is not None:
            feat, state = g.code.step(state, cur)   # t >= P > end_pos: always active for static codes
            delta_row = g.delta(feat).to(resid.dtype)
        for i, blk in enumerate(model.blocks):
            if g is not None and i == layer:
                resid = resid + delta_row
            x = blk.ln1(resid)
            q, k, v = blk.attn.qkv(x).split(blk.attn.d_inner, dim=-1)
            K, V = caches[i]
            K[:, :, t] = k.view(B, H, dh)
            V[:, :, t] = v.view(B, H, dh)
            y = F.scaled_dot_product_attention(q.view(B, H, 1, dh), K[:, :, : t + 1], V[:, :, : t + 1],
                                               scale=blk.attn.scale)
            resid = resid + blk.attn.out(y.reshape(B, -1))
            resid = resid + blk.mlp(blk.ln2(resid))
        cur = model.unembed(model.ln_final(resid)).argmax(-1)
    return out


def load_ckpt(model_path: Path, injector_path: Path | None, device):
    model, meta = load_torch_ckpt(model_path, device=device)
    injector = None
    if injector_path is not None and injector_path.exists():
        d = torch.load(injector_path, weights_only=True)
        assert d.get("mode", "complete") == "complete", "eval_ckpts supports complete mode only"
        injector = load_injector(injector_path, device)   # legacy W_inj (edge/onion) or GraftInjector (vertex/ego/...)
    return model, injector


def main(run_dir: str, grid_n: int, eval_dir: str = "/workspace/evalsets", n_mazes: int = 2000,
         seed: int = 4321, n_walls: int = 2, n_doors: int = 2, batch: int = 1000,
         rollout_batch: int = 2000, max_new: int | None = None, worlds: str = "auto",
         include_final: bool = True, only_iters: str = "", kv: bool = True, verify_kv: bool = False):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    run = Path(run_dir)
    cfg = ConfigHolder.load(json.load(open(run / "config_holder.json")))
    es = EvalSet(cfg, grid_n, Path(eval_dir), n_mazes, seed, n_walls, n_doors, device)
    inj_final = run / "injector.final.pt"
    if inj_final.exists() and torch.load(inj_final, weights_only=True).get("dummy_data", False):
        es.dummify_()
        print("dummy-data run: adjacency list blanked in every eval prompt", flush=True)
    if max_new is None:
        max_new = grid_n * grid_n + 2
    ck = run / "checkpoints"
    iters = sorted(int(m.group(1)) for p in ck.glob("model.iter_*.pt") if (m := re.match(r"model\.iter_(\d+)\.pt", p.name)))
    if only_iters != "":
        if isinstance(only_iters, int):
            keep = {int(only_iters)}
        elif isinstance(only_iters, (tuple, list)):
            keep = {int(x) for x in only_iters}
        else:
            keep = {int(x) for x in str(only_iters).split(",")}
        iters = [i for i in iters if i in keep]
    jobs = [(it, ck / f"model.iter_{it}.pt", ck / f"injector.iter_{it}.pt") for it in iters]
    if include_final and (run / "model.final.pt").exists():
        jobs.append((-1, run / "model.final.pt", run / "injector.final.pt"))
    out_dir = run / "eval"
    out_dir.mkdir(exist_ok=True)
    rows, per_maze = [], {}
    injected = (run / "injector.final.pt").exists() or any(p.exists() for _, _, p in jobs)
    world_list = ["true", "phantom"] if injected else ["true"]
    if worlds != "auto":
        world_list = worlds.split(",")
    print(f"{run.name}: {len(jobs)} checkpoints, worlds {world_list}, {len(es.seqs)} mazes, "
          f"{len(es.phantom)} phantom worlds", flush=True)
    t0 = time.time()
    for it, mp, ip in jobs:
        model, injector = load_ckpt(mp, ip, device)
        sc = Scorer(model, injector, device)
        for world in world_list:
            tf, n_pos = score_tf(sc, es, world, batch)
            if verify_kv:
                ro_a = rollout(sc, es, world, rollout_batch, max_new, kv=False)
                ro_b = rollout(sc, es, world, rollout_batch, max_new, kv=True)
                for key in ("solved", "reached", "clean"):
                    a, b = ro_a[key], ro_b[key]; m = ~np.isnan(a)
                    print(f"    verify_kv {world} {key}: agree {float((a[m]==b[m]).mean()):.4f} "
                          f"(no-kv {np.nanmean(a):.4f} kv {np.nanmean(b):.4f})", flush=True)
            ro = rollout(sc, es, world, rollout_batch, max_new, kv=kv)
            m = ~np.isnan(tf[:, 0])
            row = dict(iteration=it, world=world, n_mazes=int(m.sum()), n_pos=n_pos,
                       p_sp=float(np.nanmean(tf[:, 0])), p_valid=float(np.nanmean(tf[:, 1])),
                       p_invalid=float(np.nanmean(1 - tf[:, 0] - tf[:, 1])),
                       solved=float(np.nanmean(ro["solved"])), reached=float(np.nanmean(ro["reached"])),
                       clean=float(np.nanmean(ro["clean"])), valid_move_frac=float(np.nanmean(ro["valid_move_frac"])),
                       len_ratio=float(np.nanmean(ro["len_ratio"])) if np.any(~np.isnan(ro["len_ratio"])) else float("nan"))
            rows.append(row)
            per_maze[f"{it}_{world}"] = np.stack([tf[:, 0], tf[:, 1], ro["solved"], ro["reached"], ro["clean"]], 1).astype(np.float32)
            print(f"  iter {it:>5} {world:8s} sp {row['p_sp']:.4f} valid {row['p_valid']:.4f} "
                  f"invalid {row['p_invalid']:.5f} solved {row['solved']:.3f} reached {row['reached']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(run=run.name, grid_n=grid_n, n_mazes=n_mazes, eval_seed=seed,
                       n_walls=n_walls, n_doors=n_doors, rows=rows), open(out_dir / "metrics.json", "w"), indent=1)
        del model, injector, sc
    np.savez_compressed(out_dir / "per_maze.npz", columns=np.array(["p_sp", "p_valid", "solved", "reached", "clean"]), **per_maze)
    print(f"done {run.name}: {len(rows)} rows in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
