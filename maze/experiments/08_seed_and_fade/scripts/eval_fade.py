#!/usr/bin/env python3
"""Score seed-and-fade checkpoints on the STANDARD eval set (distance-stratified,
maze/paper/curves/results/distance/sets/g{n}.npz): prompt = maze A, target world =
maze B. Needs only torch + numpy (the set file is self-contained).

Per checkpoint:
  retention   does the model's OWN residual at the graft site carry the graft's
              code along the graft's directions (fade_lib.retention), at the
              ORIGIN token (headline: the first position where the code is
              needed), at <PATH_START>, and at the first three teacher-forced
              path positions; fit on the first 2/3 of the base mazes,
              accuracies on the last 1/3. The criterion is `gain` (unbiased),
              not cosine: the base model's own maze content at these positions
              is 1-5x the graft amplitude, which crushes cosine even at gain 1.
  conditions  greedy rollouts, judged against B (solved_graft) and A (solved_token)
    off            nothing at the site: the model on autopilot
    on             the graft as trained, carrying B
    delta{s}       + s * (code(B) - code(A)) at the site: steering along the
                   seeded directions, s in --strengths, plus delta_g{m} with
                   s = m * (measured gain at the origin) for m in 0.5, 1, 2
                   (steering is sharply tuned to s ~ gain, so a fixed grid
                   can miss an adopted code of small gain)
    delta_perm     the same edit with the graft's directions PERMUTED across the
                   free connections (strength 1): right subspace, wrong labels
    replace{s}     overwrite the span: mean-ablate it, then write s * code(B) (levels >= 1);
                   replaceA{s}: the same with code(A) (level 0): does the model keep solving
                   its own maze when its own code is swapped for the canonical one
    ablate         per-position mean-ablate the span of the graft's directions (level 0 only)
    ablate_rand    the same for 3 random subspaces of the same rank (mean + each; level 0 only)
  summary per condition: same-maze solve rate (level 0), route-changed solve rate
  (levels >= 1 whose origin->target route differs; the headline steering number),
  and how often the rollout still solves the PROMPT maze on those items.

  python eval_fade.py --run_dir <run> --set_npz <g6.npz> [--iters 1400,4000,-1]
  --edit_layer L: read / steer / ablate at resid_pre of block L instead of the graft's own site
  base model (no injector): add --w_seed S --w_layer L = the fixed graft a run of
  seed S would have had at layer L (unit Gaussian rows, scale 1), or --w_from <injector.pt>
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

# several evals share one host: without this every process spins ~one OpenMP
# thread per core and the small CPU solves in retention() crawl
torch.set_num_threads(4)

from fade_lib import (SiteEditor, free_mask, greedy_kv_edit, random_projector, retention,
                      site_residual, span_projector)
from maze.core.model import MazeGPT, MazeGPTConfig


def load_model(path, device):
    # maze.core.tl_bridge.load_torch_ckpt without its maze_transformer import
    d = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = MazeGPT(MazeGPTConfig.from_dict(d["model_cfg"]))
    model.load_state_dict(d["state_dict"])
    return model.to(device).eval()


def feat_idx(a, b, n):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return n * n + r1 * n + min(c1, c2)
    return min(r1, r2) * n + c1


def shortest_path(conn, src, dst, n):
    """BFS on the lattice (as eval_ckpts.shortest_path). List of cells src..dst."""
    from collections import deque
    def nbrs(cell):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and conn[feat_idx(cell, (rr, cc), n)]:
                yield (rr, cc)
    d = {dst: 0}; q = deque([dst])
    while q:
        u = q.popleft()
        for v in nbrs(u):
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    path = [src]
    while path[-1] != dst:
        path.append(min(v for v in nbrs(path[-1]) if v in d and d[v] == d[path[-1]] - 1))
    return path


def judge(coords, conn, n, target):
    bad = 0
    for a, b in zip(coords[:-1], coords[1:]):
        if not (abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and conn[feat_idx(a, b, n)]):
            bad += 1
    return len(coords) > 1 and coords[-1] == target, bad


class StdSet:
    def __init__(self, npz, n_base, device):
        S = np.load(npz)
        self.n = int(S["grid_n"])
        nb = len(S["conn_a"]) if n_base <= 0 else min(n_base, len(S["conn_a"]))
        keep = S["maze"] < nb
        self.maze, self.level = S["maze"][keep], S["level"][keep]
        self.conn_b, self.path_same = S["conn_b"][keep], S["path_same"][keep]
        self.conn_a = S["conn_a"][:nb]
        self.origin, self.target = S["origin"][:nb], S["target"][:nb]
        self.prompts = torch.tensor(S["prompts"][:nb], device=device)
        self.end_pos = torch.tensor(S["end_pos"][:nb], device=device, dtype=torch.long)
        self.id_to_coord = {int(i): (int(r), int(c)) for i, r, c in S["coord_ids"]}
        self.pe_id, self.pad_id = int(S["path_end_id"]), int(S["pad_id"])
        self.sha = str(S["sha"])
        self.device = device
        # teacher-forced sequences prompt + true path[1:] + <PATH_END>, right-padded
        coord_to_id = {v: k for k, v in self.id_to_coord.items()}
        seqs = []
        for i in range(nb):
            path = shortest_path(self.conn_a[i], tuple(self.origin[i]), tuple(self.target[i]), self.n)
            seqs.append(S["prompts"][i].tolist() + [coord_to_id[c] for c in path[1:]] + [self.pe_id])
        T = max(len(q) for q in seqs)
        tf = torch.full((nb, T), self.pad_id, dtype=torch.long)
        for i, q in enumerate(seqs):
            tf[i, : len(q)] = torch.tensor(q)
        self.tf_tokens = tf.to(device)
        self.tf_len = torch.tensor([len(q) for q in seqs], device=device)


@torch.no_grad()
def run_condition(model, layer, es: StdSet, items, make_editor, batch):
    n, dev = es.n, es.device
    N = len(items)
    solved_b = np.zeros(N, bool); solved_a = np.zeros(N, bool)
    max_new = n * n + 2
    for lo in range(0, N, batch):
        js = items[lo: lo + batch]
        idx = es.maze[js]
        it = torch.tensor(idx, device=dev)
        ca = torch.tensor(es.conn_a[idx], device=dev)
        cb = torch.tensor(es.conn_b[js], device=dev)
        gen = greedy_kv_edit(model, make_editor(ca, cb), layer, es.prompts[it], es.end_pos[it],
                             max_new, es.pe_id, es.pad_id).cpu().numpy()
        for k, (j, i) in enumerate(zip(js, idx)):
            g = gen[k].tolist()
            for stop in (es.pe_id, es.pad_id):
                if stop in g:
                    g = g[: g.index(stop)]
            malformed = any(t not in es.id_to_coord for t in g)
            coords = [tuple(es.origin[i])] + [es.id_to_coord[t] for t in g if t in es.id_to_coord]
            tgt = tuple(es.target[i])
            r, bb = judge(coords, es.conn_b[j], n, tgt)
            _, ba = judge(coords, es.conn_a[i], n, tgt)
            solved_b[lo + k] = r and bb == 0 and not malformed
            solved_a[lo + k] = r and ba == 0 and not malformed
    return solved_b, solved_a


def summarize(es: StdSet, items, solved_b, solved_a):
    lv, same = es.level[items], es.path_same[items]
    rc = (lv >= 1) & ~same
    out = dict(n_items=int(len(items)))
    if (lv == 0).any():
        out["same_maze"] = float(solved_b[lv == 0].mean())
    if rc.any():
        out["route_changed"] = float(solved_b[rc].mean())
        out["route_changed_solves_prompt_maze"] = float(solved_a[rc].mean())
        out["n_route_changed"] = int(rc.sum())
    return out


POSITIONS = ("origin", "path_start", "path1", "path2", "path3")


@torch.no_grad()
def site_stats(model, layer, es: StdSet, batch, max_new):
    """Own site residual along the teacher-forced true path of every base maze.
    -> X: {position: ([N, d] residual, [N] valid mask)} for POSITIONS
       mu: [P + max_new, d] per-position mean over the mazes that reach that
           position (positions past the longest path repeat the last mean)"""
    P = es.prompts.shape[1]
    T = es.tf_tokens.shape[1]
    pos = torch.arange(T, device=es.device)
    sums = torch.zeros(T, model.cfg.d_model, device=es.device); cnt = torch.zeros(T, device=es.device)
    cols = {"path_start": P - 2, "origin": P - 1, "path1": P, "path2": P + 1, "path3": P + 2}
    xs = {k: [] for k in cols}
    for lo in range(0, len(es.tf_tokens), batch):
        tok = es.tf_tokens[lo: lo + batch]
        X = site_residual(model, layer, tok).float()
        valid = (pos[None, :] < es.tf_len[lo: lo + batch, None]).float()
        sums += (X * valid[:, :, None]).sum(0); cnt += valid.sum(0)
        for k, c in cols.items():
            xs[k].append(X[:, c].cpu())
    Xout = {}
    for k, c in cols.items():
        Xout[k] = (torch.cat(xs[k]), (es.tf_len > c).cpu())
    mu = sums / cnt.clamp_min(1)[:, None]
    last = int((cnt > 0).nonzero().max())
    mu[last + 1:] = mu[last]
    if T < P + max_new:
        mu = torch.cat([mu, mu[-1:].expand(P + max_new - T, -1)])
    return Xout, mu[: P + max_new]


def ret_at(X, F_all, W_free, scale, n_fit):
    """retention() on the mazes whose path reaches this position, split by index."""
    x, ok = X
    fit = ok.clone(); fit[n_fit:] = False
    test = ok.clone(); test[:n_fit] = False
    return retention(x[fit], F_all[fit], x[test], F_all[test], W_free, scale)


def main(run_dir: str, set_npz: str, out_dir: str = "", iters: str = "-1", n_base: int = 0,
         strengths: str = "0.1,0.25,0.5,1,2", batch: int = 2000, w_from: str = "", w_seed: int = -1,
         w_layer: int = 0, device: str = "", rand_seeds: str = "1234,1235,1236", edit_layer: int = -1,
         conditions: str = "off,on,delta,delta_perm,ablate,ablate_rand,replace", ret_layers: str = "all"):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run / "eval_fade"
    out.mkdir(parents=True, exist_ok=True)
    es = StdSet(set_npz, n_base, dev)
    es_ret = es if n_base <= 0 else StdSet(set_npz, 0, dev)   # retention always uses every base maze
    n = es.n
    fm = torch.tensor(free_mask(n))
    if str(iters) == "all":
        ck = run / "checkpoints"
        it_list = sorted(int(m.group(1)) for p in ck.glob("model.iter_*.pt")
                         if (m := re.match(r"model\.iter_(\d+)\.pt", p.name))) + [-1]
    else:
        it_list = [int(x) for x in (iters if isinstance(iters, (tuple, list)) else str(iters).split(","))]
    s_list = [float(x) for x in (strengths if isinstance(strengths, (tuple, list)) else str(strengths).split(","))]
    conds = conditions if isinstance(conditions, (tuple, list)) else str(conditions).split(",")
    all_items = np.arange(len(es.maze))
    lvl0 = all_items[es.level == 0]
    n_fit = (2 * len(es_ret.prompts)) // 3
    F_all = torch.tensor(es_ret.conn_a[:, fm.numpy()], dtype=torch.float32) * 2 - 1

    rows = []
    res_f = out / "metrics.json"
    t0 = time.time()
    for it in it_list:
        mp = run / "model.final.pt" if it < 0 else run / "checkpoints" / f"model.iter_{it}.pt"
        ip = run / "injector.final.pt" if it < 0 else run / "checkpoints" / f"injector.iter_{it}.pt"
        if it >= 0 and not mp.exists():                # pod_queue moves the kept checkpoints to kept/
            mp, ip = run / "kept" / mp.name, run / "kept" / ip.name
        if w_from:
            ip = Path(w_from)
        model = load_model(mp, dev)
        if w_seed >= 0:                                # MazeInjector's fixed init, no file needed
            gen = torch.Generator().manual_seed(w_seed)
            w = torch.randn(2 * n * n, model.cfg.d_model, generator=gen)
            inj = dict(W_inj=w / w.norm(dim=-1, keepdim=True), layer=w_layer, scale=1.0)
        else:
            inj = torch.load(ip, map_location="cpu", weights_only=True)
        W = inj["W_inj"].float().to(dev)
        layer, scale = int(inj.get("layer", 0)), float(inj.get("scale", 1.0))
        if edit_layer >= 0:
            layer = edit_layer        # apply the site edits / retention at another layer than the graft's

        X, mu = site_stats(model, layer, es_ret, batch, n * n + 2)
        mu = mu.to(dev)
        ret = {k: ret_at(X[k], F_all, W[fm].cpu(), scale, n_fit) for k in POSITIONS}
        gain0 = ret["origin"]["gain"]
        ret_layers_out = {}
        if str(ret_layers) != "":
            Ls = range(model.cfg.n_layers) if str(ret_layers) == "all" else [
                int(x) for x in (ret_layers if isinstance(ret_layers, (tuple, list)) else str(ret_layers).split(","))]
            for L in Ls:                               # where (if anywhere) does the model write the code
                XL, _ = site_stats(model, L, es_ret, batch, n * n + 2)
                ret_layers_out[str(L)] = {}
                for pos_name in ("origin", "path1"):
                    rl = ret_at(XL[pos_name], F_all, W[fm].cpu(), scale, n_fit)
                    ret_layers_out[str(L)][pos_name] = {k: rl[k] for k in ("cos_gen", "gain", "acc_zero_fit", "acc_probe")}
        P_span = span_projector(W[fm]).to(dev)
        g = torch.Generator().manual_seed(4321)
        free_idx = torch.nonzero(fm).squeeze(1)
        W_perm = W.clone()
        W_perm[free_idx] = W[free_idx[torch.randperm(len(free_idx), generator=g)]]
        rank = int(round(float(torch.trace(P_span))))
        rs = [int(x) for x in (rand_seeds if isinstance(rand_seeds, (tuple, list)) else str(rand_seeds).split(","))]
        P_rands = [(sd, random_projector(rank, W.shape[1], sd, dev)) for sd in rs]

        def editors(name):
            if name == "off":
                return [("off", all_items, lambda ca, cb: SiteEditor("off"))]
            if name == "on":
                return [("on", all_items, lambda ca, cb: SiteEditor("on", W, scale, ca, cb))]
            if name == "delta":
                mk = lambda s: lambda ca, cb: SiteEditor("delta", W, scale, ca, cb, strength=s)
                out = [(f"delta{s:g}", all_items[es.level[all_items] >= 1], mk(s)) for s in s_list]
                if abs(gain0) > 0.02:                      # gain-matched strengths
                    out += [(f"delta_g{m:g}", all_items[es.level[all_items] >= 1], mk(m * gain0)) for m in (0.5, 1, 2)]
                return out
            if name == "delta_perm":
                return [("delta_perm", all_items[es.level[all_items] >= 1],
                         lambda ca, cb: SiteEditor("delta", W_perm, scale, ca, cb, strength=1.0))]
            if name == "replace":
                mk = lambda s: lambda ca, cb: SiteEditor("replace", W, scale, ca, cb, strength=s, P=P_span, mu=mu)
                mkA = lambda s: lambda ca, cb: SiteEditor("replace", W, scale, ca, ca, strength=s, P=P_span, mu=mu)
                return ([(f"replace{s:g}", all_items[es.level[all_items] >= 1], mk(s)) for s in (0.5, 1, 2)]
                        + [(f"replaceA{s:g}", lvl0, mkA(s)) for s in (0.5, 1, 2)])
            if name == "ablate":
                return [("ablate", lvl0, lambda ca, cb: SiteEditor("ablate", P=P_span, mu=mu))]
            if name == "ablate_rand":
                return [(f"ablate_rand{sd}", lvl0, (lambda Pr: lambda ca, cb: SiteEditor("ablate", P=Pr, mu=mu))(Pr))
                        for sd, Pr in P_rands]
            raise ValueError(name)

        row = dict(iteration=it, layer=layer, gain_origin=gain0, retention=ret,
                   retention_by_layer=ret_layers_out, conditions={})
        saved = {}
        for c in conds:
            for cname, items, mk in editors(c):
                sb, sa = run_condition(model, layer, es, items, mk, batch)
                row["conditions"][cname] = summarize(es, items, sb, sa)
                saved[f"{cname}__items"] = items.astype(np.int32)
                saved[f"{cname}__solved_graft"] = sb
                saved[f"{cname}__solved_token"] = sa
        ar = [v["same_maze"] for k, v in row["conditions"].items() if k.startswith("ablate_rand")]
        if ar:
            row["conditions"]["ablate_rand"] = dict(same_maze=float(np.mean(ar)), n_items=len(lvl0), per_seed=ar)
        np.savez_compressed(out / f"items_iter{it}.npz", set_sha=es.sha, **saved)
        rows.append(row)
        json.dump(dict(run=run.name, set_sha=es.sha, n_base=len(es.prompts), rows=rows),
                  open(res_f, "w"), indent=1)
        r = ret["origin"]
        print(f"iter {it:>5} L{layer} | origin: cos {r['cos_gen']:+.3f} gain {r['gain']:+.3f} "
              f"zero-fit {r['acc_zero_fit']:.3f} probe {r['acc_probe']:.3f} (major {r['acc_majority']:.3f}) | "
              + " ".join(f"{k}: same {v.get('same_maze', float('nan')):.3f} rc {v.get('route_changed', float('nan')):.3f}"
                         for k, v in row["conditions"].items())
              + f" [{time.time() - t0:.0f}s]", flush=True)
        del model
    print(f"done -> {res_f}", flush=True)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
