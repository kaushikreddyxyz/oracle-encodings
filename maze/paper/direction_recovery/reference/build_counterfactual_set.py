#!/usr/bin/env python3
"""Build a curated counterfactual edit dataset on fresh (unseen) mazes.

Guarantees (asserted with BFS on every edit):
  - detour edits (block one rollout-traversed true-path edge + open a repair
    wall spanning the cut) leave origin->target SOLVABLE;
  - shortcut opens strictly DECREASE BFS distance (gain = old - new > 0);
  - off-path blocks and zero-gain opens leave the shortest distance unchanged
    (specificity controls: behavior should NOT change).

Edits are anchored to the final fixed-injection model's true-world greedy
rollouts (block edges it actually traverses; prefer shortcut opens adjacent
to its route). Also classifies each maze's true-world rollout into strata
(solved / wall infraction / unsolved without infraction) for figures.

Generation is SERIAL on purpose: maze_dataset's parallel path seeds each pool
worker as `seed + process_id` and hands out indices with `imap`, so the maze
sample depends on the worker count *and* on scheduling order — it does not
reproduce even run-to-run on one machine (verified: two parallel builds of the
same config differ, two serial builds are bit-identical). Sets built with
gen_parallel=True (anything before 2026-08-04) cannot be regenerated.

Output: /workspace/runs/counterfactual_set/set.pt with
  tokens/offsets/conn/end_pos     600 fresh mazes (seed 1234), tokenized
  edits                           list of dicts (maze, family, block, open,
                                  gain, old_dist, new_dist)
  strata                          maze indices per stratum (fixed model)
  rollout_true                    fixed-model true-world rollout token lists
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
from eval_injection import ProbeSet, greedy_rollout
from fast_data import pretokenize, PretokDataset

GRID_N = 6
N_CELLS = GRID_N * GRID_N


def feat_idx(a, b):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return N_CELLS + r1 * GRID_N + min(c1, c2)
    return min(r1, r2) * GRID_N + c1


def feat_cells(f):
    d, rem = divmod(int(f), N_CELLS)
    r, c = divmod(rem, GRID_N)
    if d:  # right edge
        return (r, c), (r, c + 1)
    return (r, c), (r + 1, c)  # down edge


def neighbors(conn, cell):
    r, c = cell
    out = []
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = (r + dr, c + dc)
        if 0 <= nb[0] < GRID_N and 0 <= nb[1] < GRID_N and conn[feat_idx(cell, nb)]:
            out.append(nb)
    return out


def bfs_dist(conn, src, dst):
    seen = {src: 0}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            return seen[cur]
        for nb in neighbors(conn, cur):
            if nb not in seen:
                seen[nb] = seen[cur] + 1
                q.append(nb)
    return None


def component(conn, src):
    seen = {src}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nb in neighbors(conn, cur):
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def all_closed_walls(conn):
    out = []
    for r in range(GRID_N):
        for c in range(GRID_N):
            if r < GRID_N - 1 and not conn[feat_idx((r, c), (r + 1, c))]:
                out.append(feat_idx((r, c), (r + 1, c)))
            if c < GRID_N - 1 and not conn[feat_idx((r, c), (r, c + 1))]:
                out.append(feat_idx((r, c), (r, c + 1)))
    return out


def rollout_to_coords(ps, i, gen):
    o = ps.id_to_coord.get(int(ps.tokens[i, ps.origin_pos[i]]))
    coords = [o]
    for t in gen:
        c = ps.id_to_coord.get(t)
        if c is None:
            break
        coords.append(c)
    return coords


def traversed_edges(coords):
    out = []
    for a, b in zip(coords, coords[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
            out.append(feat_idx(a, b))
    return out


def classify_rollout(ps, i, gen, conn):
    coords = rollout_to_coords(ps, i, gen)
    target = ps.id_to_coord[int(ps.target_id[i])]
    infraction = False
    for a, b in zip(coords, coords[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 or not conn[feat_idx(a, b)]:
            infraction = True
    reached = len(coords) > 1 and coords[-1] == target
    if reached and not infraction:
        return "solved"
    if infraction:
        return "infraction"
    return "unsolved_clean"


def main(
    fixed_run_dir: str = "/workspace/runs/fast/fast-inj-fixed-complete-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000",
    out_dir: str = "/workspace/runs/counterfactual_set",
    n_mazes: int = 600,
    seed: int = 1234,
    gen_parallel: bool = False,  # see module docstring: True is NOT reproducible
):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    cfg = ConfigHolder.load(
        json.load(open(Path(fixed_run_dir) / "config_holder.json"))
    )
    vocab = cfg.tokenizer.get_vocab()

    ds_cfg = MazeDatasetConfig(
        name="cfset", grid_n=GRID_N, n_mazes=n_mazes,
        maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed,
    )
    dataset = MazeDataset.from_config(
        ds_cfg, do_generate=True, load_local=False, save_local=False,
        gen_parallel=gen_parallel, verbose=True,
    )
    pretok_file = out_path / "pretok-cfset.pt"
    pretokenize(
        dataset, cfg.maze_tokenizer, vocab, pretok_file,
        bos_id=int(cfg.tokenizer.bos_token_id),
    )
    pretok = PretokDataset(pretok_file)
    entries = [pretok[i] for i in range(len(pretok))]
    ps = ProbeSet(entries, vocab, GRID_N, device)
    print(f"{len(ps.conn)} mazes in probe structure (of {len(entries)} generated)")

    model, injector = load_checkpoint(
        Path(fixed_run_dir) / "model.final.zanj",
        Path(fixed_run_dir) / "injector.final.pt",
        device,
    )
    rollouts = []
    for lo in range(0, len(ps.conn), 250):
        sub = _slice_ps(ps, lo, min(lo + 250, len(ps.conn)))
        rollouts += greedy_rollout(model, injector, sub, sub.conn)

    conn_np = ps.conn.cpu().numpy()
    edits = []
    strata = {"solved": [], "infraction": [], "unsolved_clean": []}
    n_detour = n_offpath = n_short = n_zero = 0

    for i in range(len(ps.conn)):
        conn = conn_np[i]
        row = ps.tokens[i].tolist()
        origin = ps.id_to_coord[int(ps.tokens[i, ps.origin_pos[i]])]
        target = ps.id_to_coord[int(ps.target_id[i])]
        old_dist = bfs_dist(conn, origin, target)
        assert old_dist is not None

        strata[classify_rollout(ps, i, rollouts[i], conn)].append(i)
        coords = rollout_to_coords(ps, i, rollouts[i])
        route_edges = [f for f in traversed_edges(coords) if conn[f]]
        route_cells = set(coords)

        # true shortest-path edges (from the token sequence's solution)
        pe = row.index(ps.path_end_id)
        path_coords = [
            ps.id_to_coord[t]
            for t in row[int(ps.origin_pos[i]): pe]
            if t in ps.id_to_coord
        ]
        path_edges = set(traversed_edges(path_coords))

        # --- detour edits: block a traversed path edge + open a repair ---
        cands = [f for f in route_edges if f in path_edges]
        picks = []
        if cands:
            thirds = [cands[: len(cands) // 3] or cands,
                      cands[len(cands) // 3: 2 * len(cands) // 3] or cands,
                      cands[2 * len(cands) // 3:] or cands]
            for t in thirds:
                f = t[int(rng.integers(len(t)))]
                if f not in picks:
                    picks.append(f)
        for f in picks:
            blocked = conn.copy()
            blocked[f] = False
            u, v = feat_cells(f)
            comp_u = component(blocked, u)
            mid = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
            repairs = []
            for w in all_closed_walls(conn):
                a, b = feat_cells(w)
                if (a in comp_u) != (b in comp_u):
                    wm = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                    repairs.append(
                        (abs(wm[0] - mid[0]) + abs(wm[1] - mid[1]), w)
                    )
            if not repairs:
                continue
            repairs.sort()
            w = repairs[0][1]
            edited = blocked.copy()
            edited[w] = True
            nd = bfs_dist(edited, origin, target)
            assert nd is not None, "detour edit must stay solvable"
            edits.append(dict(
                maze=i, family="detour", block=int(f), open=int(w),
                gain=old_dist - nd, old_dist=int(old_dist), new_dist=int(nd),
            ))
            n_detour += 1

        # --- off-path block controls ---
        off = [
            f for f in range(2 * N_CELLS)
            if conn[f] and f not in path_edges and f not in route_edges
        ]
        rng.shuffle(off)
        for f in off[:2]:
            blocked = conn.copy()
            blocked[f] = False
            nd = bfs_dist(blocked, origin, target)
            assert nd == old_dist, "off-path block must not affect distance"
            edits.append(dict(
                maze=i, family="offpath_block", block=int(f), open=None,
                gain=0, old_dist=int(old_dist), new_dist=int(nd),
            ))
            n_offpath += 1

        # --- shortcut opens (strict gain) + zero-gain controls ---
        gains = []
        zeros = []
        for w in all_closed_walls(conn):
            opened = conn.copy()
            opened[w] = True
            nd = bfs_dist(opened, origin, target)
            g = old_dist - nd
            if g > 0:
                a, b = feat_cells(w)
                on_route = a in route_cells or b in route_cells
                gains.append((g, on_route, int(w), int(nd)))
            elif g == 0:
                zeros.append(int(w))
        gains.sort(reverse=True)
        chosen = []
        if gains:
            chosen.append(gains[0])                       # max gain
            mid_i = len(gains) // 2
            if gains[mid_i][2] != gains[0][2]:
                chosen.append(gains[mid_i])               # median gain
            for g in gains:
                if g[1] and g[2] not in {c[2] for c in chosen}:
                    chosen.append(g)                      # touches the route
                    break
        for g, on_route, w, nd in chosen:
            assert g > 0
            edits.append(dict(
                maze=i, family="shortcut_open", block=None, open=w,
                gain=int(g), old_dist=int(old_dist), new_dist=int(nd),
                on_route=bool(on_route),
            ))
            n_short += 1
        rng.shuffle(zeros)
        for w in zeros[:1]:
            edits.append(dict(
                maze=i, family="zerogain_open", block=None, open=int(w),
                gain=0, old_dist=int(old_dist), new_dist=int(old_dist),
            ))
            n_zero += 1

    torch.save(dict(
        pretok_file=str(pretok_file),
        edits=edits,
        strata=strata,
        rollout_true_fixed=rollouts,
    ), out_path / "set.pt")
    with open(out_path / "meta.json", "w") as fjson:
        json.dump(dict(
            n_mazes=len(ps.conn), seed=seed, gen_parallel=gen_parallel,
            anchor_run=Path(fixed_run_dir).name,
            n_edits=len(edits), detour=n_detour, offpath_block=n_offpath,
            shortcut_open=n_short, zerogain_open=n_zero,
            strata_counts={k: len(v) for k, v in strata.items()},
        ), fjson, indent=2)
    print(json.dumps(json.load(open(out_path / "meta.json")), indent=2))


def _slice_ps(ps, lo, hi):
    import copy
    sub = copy.copy(ps)
    for f in ("tokens", "attn", "conn", "end_pos", "lens", "origin_pos",
              "first_move", "first_feat", "open_id", "open_feat", "target_id"):
        setattr(sub, f, getattr(ps, f)[lo:hi])
    return sub


if __name__ == "__main__":
    fire.Fire(main)
