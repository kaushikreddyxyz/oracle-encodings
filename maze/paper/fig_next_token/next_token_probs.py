#!/usr/bin/env python3
"""Next-token probability mass by category, teacher-forced along the shortest
path, on fresh held-out mazes. POD-SIDE script: run from /workspace/maze_injection
(legacy flat layout) so its modules import.

Per maze and per path position the vocabulary is split into
  correct  next cell on a shortest path to the target in the world being scored
           (at the target cell: <PATH_END>)
  valid    other lattice neighbours of the current cell with no wall between
           (contiguous + no wall, but not on a shortest path)
  invalid  everything else: non-adjacent cells, walled-off neighbours, special
           tokens, <PATH_END> before the target
and the softmax mass in each bin is summed (the three sum to 1).

Two worlds per maze (same prompt = the true maze):
  true      the maze as given; channel (if any) carries the true features;
            teacher-forced on the true shortest path
  phantom   n_walls phantom walls on the true shortest path + n_doors phantom
            doors where true walls were (target still reachable); channel
            carries the PHANTOM features; teacher-forced on the phantom
            shortest path; scored against the phantom world. For a base model
            (no channel) this is the control: it cannot see the edit.

  python next_token_probs.py --models '{"baseline-s0": "final_models/baseline", ...}' \
      --out /workspace/runs/next_token --n_mazes 2000 --seed 4321
"""

import json
from collections import deque
from pathlib import Path

import fire
import numpy as np
import torch
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder

from eval_counterfactual import load_checkpoint
from eval_injection import ProbeSet, _forward
from fast_data import PretokDataset, pretokenize

GRID_N = 6
N_CELLS = GRID_N * GRID_N
N_FEAT = 2 * N_CELLS


# ---- lattice helpers (feature layout as in injection.py / evals.py) ---------
def feat_idx(a, b):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return N_CELLS + r1 * GRID_N + min(c1, c2)
    return min(r1, r2) * GRID_N + c1


def lattice_nbrs(cell):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < GRID_N and 0 <= cc < GRID_N:
            yield (rr, cc)


def open_nbrs(conn, cell):
    return [nb for nb in lattice_nbrs(cell) if conn[feat_idx(cell, nb)]]


def dist_to(conn, dst):
    """BFS distance from every reachable cell to dst."""
    d = {dst: 0}
    q = deque([dst])
    while q:
        u = q.popleft()
        for v in open_nbrs(conn, u):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def shortest_path(conn, src, dst):
    d = dist_to(conn, dst)
    if src not in d:
        return None
    path = [src]
    while path[-1] != dst:
        cur = path[-1]
        nxt = min((nb for nb in open_nbrs(conn, cur) if nb in d and d[nb] == d[cur] - 1))
        path.append(nxt)
    return path


INTERNAL = np.array(
    [(f < N_CELLS and f // GRID_N < GRID_N - 1) or (f >= N_CELLS and (f - N_CELLS) % GRID_N < GRID_N - 1)
     for f in range(N_FEAT)]
)


def make_phantom(conn, path, rng, n_walls=2, n_doors=2, tries=400):
    """Grafted world: n_walls edges of the true path closed, n_doors true walls
    opened, target still reachable, both doors used by the new shortest path
    when possible (falls back to >= 1 door used, then to 1 wall + 1 door)."""
    o, t = path[0], path[-1]
    path_feats = [feat_idx(path[i], path[i + 1]) for i in range(len(path) - 1)]
    closed = [f for f in range(N_FEAT) if INTERNAL[f] and not conn[f]]
    for nw, nd, need_all in ((n_walls, n_doors, True), (n_walls, n_doors, False), (1, 1, True)):
        nw = min(nw, len(path_feats))
        for _ in range(tries):
            walls = rng.choice(path_feats, nw, replace=False)
            doors = rng.choice(closed, nd, replace=False)
            c2 = conn.copy()
            c2[walls] = False
            c2[doors] = True
            p2 = shortest_path(c2, o, t)
            if p2 is None:
                continue
            used = {feat_idx(p2[i], p2[i + 1]) for i in range(len(p2) - 1)}
            n_used = sum(int(d) in used for d in doors)
            if need_all and n_used < nd:
                continue
            return c2, [int(w) for w in walls], [int(d) for d in doors], p2
    return None


# ---- scoring -----------------------------------------------------------------
def category_ids(ps, conn, cell, target, dist):
    if cell == target:
        correct = [ps.path_end_id]
        valid = [ps.coord_to_id[nb] for nb in open_nbrs(conn, cell)]
    else:
        nb_open = open_nbrs(conn, cell)
        correct = [ps.coord_to_id[nb] for nb in nb_open if nb in dist and dist[nb] == dist[cell] - 1]
        valid = [ps.coord_to_id[nb] for nb in nb_open if ps.coord_to_id[nb] not in correct]
    return correct, valid


@torch.no_grad()
def score_world(model, injector, ps, worlds, batch=200):
    """worlds: list of dicts {i, conn(np bool 72), path(list of cells)} — one per
    maze. Returns per-position records (maze, step, path_len, p_correct, p_valid)."""
    recs = []
    device = ps.device
    for lo in range(0, len(worlds), batch):
        chunk = worlds[lo: lo + batch]
        seqs, metas = [], []
        for w in chunk:
            i = w["i"]
            op = int(ps.origin_pos[i])
            prompt = ps.tokens[i, : op + 1].tolist()  # through the origin token
            path_ids = [ps.coord_to_id[c] for c in w["path"][1:]] + [ps.path_end_id]
            seqs.append(prompt + path_ids)
            metas.append((i, op, len(w["path"])))
        L = max(len(s) for s in seqs)
        toks = torch.full((len(seqs), L), ps.pad_id, dtype=torch.long, device=device)
        for k, s in enumerate(seqs):
            toks[k, : len(s)] = torch.tensor(s, device=device)
        attn = (torch.arange(L, device=device)[None, :] < torch.tensor([len(s) for s in seqs], device=device)[:, None]).long()
        conn = torch.tensor(np.stack([w["conn"] for w in chunk]), dtype=torch.bool, device=device)
        end_pos = torch.stack([ps.end_pos[w["i"]] for w in chunk])
        logits = _forward(model, injector, toks, attn, conn, end_pos)
        probs = logits.float().softmax(-1).cpu().numpy()
        for k, (w, (i, op, plen)) in enumerate(zip(chunk, metas)):
            target = w["path"][-1]
            dist = dist_to(w["conn"], target)
            for step, cell in enumerate(w["path"]):
                p = probs[k, op + step]
                correct, valid = category_ids(ps, w["conn"], cell, target, dist)
                pc = float(p[correct].sum())
                pv = float(p[valid].sum())
                recs.append((i, step, plen, pc, pv))
    return np.array(recs, dtype=np.float64)


def summarize(recs):
    pc, pv = recs[:, 3], recs[:, 4]
    pi = 1.0 - pc - pv
    first = recs[:, 1] == 0
    last = recs[:, 1] == recs[:, 2] - 1
    def m(mask):
        return dict(p_correct=float(pc[mask].mean()), p_valid=float(pv[mask].mean()),
                    p_invalid=float(pi[mask].mean()), n_pos=int(mask.sum()))
    allm = np.ones(len(recs), dtype=bool)
    return dict(**m(allm), n_mazes=int(len(np.unique(recs[:, 0]))),
                first_move=m(first), at_target=m(last), interior=m(~first & ~last))


# ---- main --------------------------------------------------------------------
def main(
    models: dict,
    out: str = "/workspace/runs/next_token",
    n_mazes: int = 2000,
    seed: int = 4321,
    n_walls: int = 2,
    n_doors: int = 2,
    batch: int = 200,
):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(models, str):  # path to a JSON mapping {name: run_dir} (robust to shell quoting)
        models = json.load(open(models))
    models = {k: Path(v) for k, v in dict(models).items()}

    # held-out mazes (serial generation => reproducible), tokenized once
    first_dir = next(iter(models.values()))
    cfg = ConfigHolder.load(json.load(open(first_dir / "config_holder.json")))
    vocab = cfg.tokenizer.get_vocab()
    ds_cfg = MazeDatasetConfig(name="nexttok", grid_n=GRID_N, n_mazes=n_mazes,
                               maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed)
    dataset = MazeDataset.from_config(ds_cfg, do_generate=True, load_local=False,
                                      save_local=False, gen_parallel=False, verbose=True)
    pretok_file = out_dir / f"pretok-nexttok-s{seed}-n{n_mazes}.pt"
    if not pretok_file.exists():
        pretokenize(dataset, cfg.maze_tokenizer, vocab, pretok_file,
                    bos_id=int(cfg.tokenizer.bos_token_id))
    pretok = PretokDataset(pretok_file)
    ps = ProbeSet([pretok[i] for i in range(len(pretok))], vocab, GRID_N, device)
    conn_np = ps.conn.cpu().numpy()
    print(f"{len(conn_np)} held-out mazes")

    # worlds: true + phantom (same edits for every model: rng seeded once)
    rng = np.random.default_rng(seed)
    true_w, phan_w, edits = [], [], []
    n_fail = 0
    for i in range(len(conn_np)):
        row = ps.tokens[i].tolist()
        op = int(ps.origin_pos[i])
        pe = row.index(ps.path_end_id)
        path = [ps.id_to_coord[t] for t in row[op:pe]]
        assert path == shortest_path(conn_np[i], path[0], path[-1]), "dataset path not the tree path?"
        true_w.append(dict(i=i, conn=conn_np[i], path=path))
        ph = make_phantom(conn_np[i], path, rng, n_walls, n_doors)
        if ph is None:
            n_fail += 1
            continue
        c2, walls, doors, p2 = ph
        phan_w.append(dict(i=i, conn=c2, path=p2))
        edits.append(dict(maze=i, walls=walls, doors=doors, len_true=len(path) - 1, len_phantom=len(p2) - 1))
    print(f"phantom worlds: {len(phan_w)} built, {n_fail} mazes without a valid edit")
    json.dump(dict(seed=seed, n_mazes=n_mazes, n_walls=n_walls, n_doors=n_doors, edits=edits),
              open(out_dir / "phantom_edits.json", "w"))

    summary = {}
    for name, d in models.items():
        model, injector = load_checkpoint(d / "model.final.zanj", d / "injector.final.pt", device)
        summary[name] = dict(run_dir=str(d), injected=injector is not None)
        for setting, worlds in (("true", true_w), ("phantom", phan_w)):
            recs = score_world(model, injector, ps, worlds, batch=batch)
            np.save(out_dir / f"recs_{name}_{setting}.npy", recs)
            summary[name][setting] = summarize(recs)
            s = summary[name][setting]
            print(f"{name:24s} {setting:8s} correct {s['p_correct']:.4f}  valid {s['p_valid']:.4f}  "
                  f"invalid {s['p_invalid']:.5f}  (n_pos {s['n_pos']})")
        del model, injector
        torch.cuda.empty_cache()
        json.dump(summary, open(out_dir / "results.json", "w"), indent=1)
    print("done ->", out_dir / "results.json")


if __name__ == "__main__":
    fire.Fire(main)
