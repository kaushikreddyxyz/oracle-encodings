#!/usr/bin/env python3
"""Local (single-decision) eval set: one SWAP of the prompt maze, one decision point. Pure numpy; runs anywhere.

Built from the mazes of the standard distance-stratified set (results/distance/sets/g{n}.npz; read-only). An item is
  (maze A with its origin/target, removed connection e-, added connection e+, decision cell d)
  B = A - e- + e+ is again a spanning tree, e- lies on A's origin->target route (so the route changes; in a tree a
  swap changes the route iff it removes a route edge), and B is a DFS tree of the grid from some root (the training
  family: every non-tree lattice edge joins an ancestor to a descendant), so that B is in-distribution both as a graft
  input and as a rewritten prompt. One swap is the smallest edit that keeps a spanning tree a spanning tree; it is
  simultaneously a BLOCK (e- is a wall invented on the model's route) and an OPEN (e+ is a door the new route must use).
  d = last cell of the common prefix of route_A and route_B. The prefix origin..d is a legal, shortest-route prefix in
  BOTH mazes, and because both are trees there is exactly one correct next move in each: move_A (toward e-) and
  move_B (toward e+), always different cells.
Per-item covariates: lookahead to the wall (route_A steps from d to e-; 0 = the very next move is blocked), lookahead
to the door (route_B steps from d to e+; 0 = the next move goes through the door), route lengths.

Two families per maze: `dfs` (family 0, primary): up to --per_maze swaps drawn uniformly from the DFS-preserving route-changing swaps;
`tree` (family 1, robustness): --per_maze_tree swaps drawn uniformly from ALL route-changing spanning-tree swaps.

  python build_local.py [--per_maze 3] [--per_maze_tree 2] [--seed 4321]      -> results/local/sets/g{n}.npz
"""
import argparse, hashlib, time
from collections import deque
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "results/distance/sets"
OUT = HERE / "results/local/sets"


def feat_idx(a, b, n):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return n * n + r1 * n + min(c1, c2)
    return min(r1, r2) * n + c1


def feat_cells(f, n):
    if f < n * n:
        r, c = divmod(f, n); return (r, c), (r + 1, c)
    r, c = divmod(f - n * n, n); return (r, c), (r, c + 1)


def free_feats(n):
    f = np.arange(n * n)
    return np.concatenate([f[(f // n) < n - 1], n * n + f[(f % n) < n - 1]])


def adjacency(conn, n):
    adj = [[] for _ in range(n * n)]
    for f in np.flatnonzero(conn):
        (r1, c1), (r2, c2) = feat_cells(int(f), n)
        u, v = r1 * n + c1, r2 * n + c2
        adj[u].append(v); adj[v].append(u)
    return adj


def tree_path(adj, src, dst):
    par = {src: -1}; q = deque([src])
    while q:
        u = q.popleft()
        if u == dst: break
        for v in adj[u]:
            if v not in par: par[v] = u; q.append(v)
    if dst not in par: return None
    p = [dst]
    while p[-1] != src: p.append(par[p[-1]])
    return p[::-1]


def is_spanning_tree(conn, n):
    adj = adjacency(conn, n)
    if sum(len(a) for a in adj) != 2 * (n * n - 1): return False
    seen = {0}; q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); q.append(v)
    return len(seen) == n * n


def dfs_roots(conn, n, free):
    """Bool [n^2]: the roots from which this spanning tree is a DFS tree of the n x n lattice (every non-tree lattice
    edge must join an ancestor and a descendant). Root r works for the non-tree edge (x, y) iff r attaches to the tree
    path x..y at x or at y."""
    N = n * n; adj = adjacency(conn, n)
    par = np.full(N, -1); tin = np.zeros(N, int); tout = np.zeros(N, int); t = 0
    stack = [(0, iter(adj[0]))]; tin[0] = 0; t = 1
    while stack:
        u, it = stack[-1]
        for v in it:
            if v != par[u]:
                par[v] = u; tin[v] = t; t += 1; stack.append((v, iter(adj[v]))); break
        else:
            tout[u] = t; stack.pop()
    order = tin                                        # vertex -> euler index; subtree(v) = [tin[v], tout[v])
    ok = np.ones(N, bool)
    for f in free:
        if conn[f]: continue
        (r1, c1), (r2, c2) = feat_cells(int(f), n); x, y = r1 * n + c1, r2 * n + c2
        valid = np.zeros(N, bool)
        for a, b in ((x, y), (y, x)):                  # roots attaching at a: the side of a once the first path edge a->b.. is cut
            if tin[a] <= tin[b] < tout[a]:             # b below a: cut the child of a that contains b
                c = next(c for c in adj[a] if c != par[a] and tin[c] <= tin[b] < tout[c])
                valid |= ~((order >= tin[c]) & (order < tout[c]))
            else:                                      # path leaves a through its parent
                valid |= (order >= tin[a]) & (order < tout[a])
        ok &= valid
        if not ok.any(): break
    return ok


def _job(args):
    i, conn, origin, target, n, seed, per_maze, per_maze_tree = args
    rng = np.random.default_rng([seed, n, i, 31]); free = free_feats(n)
    adj = adjacency(conn, n); o, t = origin[0] * n + origin[1], target[0] * n + target[1]
    route = tree_path(adj, o, t); cands = []
    for k in range(len(route) - 1):
        u, v = route[k], route[k + 1]
        side = {o}; q = deque([o])                      # origin's component of A - (u, v)
        while q:
            a = q.popleft()
            for b in adj[a]:
                if b not in side and {a, b} != {u, v}: side.add(b); q.append(b)
        for f in free:
            if conn[f]: continue
            (r1, c1), (r2, c2) = feat_cells(int(f), n)
            if ((r1 * n + c1) in side) != ((r2 * n + c2) in side): cands.append((k, int(f)))
    n_cand = len(cands); n_tested = 0; n_dfs = 0; out = []

    def item(ci, family):
        k, add = cands[ci]; rem = feat_idx(divmod(route[k], n), divmod(route[k + 1], n), n)
        b = conn.copy(); b[rem] = False; b[add] = True
        dfs_ok = bool(dfs_roots(b, n, free).any())
        if family == 0 and not dfs_ok: return dfs_ok, None
        assert is_spanning_tree(b, n)
        rb = tree_path(adjacency(b, n), o, t)
        j = 0
        while j + 1 < min(len(route), len(rb)) and route[j + 1] == rb[j + 1]: j += 1
        assert route[: j + 1] == rb[: j + 1] and route[j + 1] != rb[j + 1] and j <= k
        door = next(m for m in range(len(rb) - 1) if feat_idx(divmod(rb[m], n), divmod(rb[m + 1], n), n) == add)
        assert door >= j
        return dfs_ok, dict(maze=i, family=family, rem=rem, add=add, conn_b=b, j=j, k_rem=k, k_door=door, move_a=route[j + 1], move_b=rb[j + 1],
                            len_a=len(route) - 1, len_b=len(rb) - 1, route_a=route, route_b=rb, dfs=dfs_ok)

    got = 0                                                # family 0: DFS-preserving swaps, uniformly without replacement
    for ci in rng.permutation(n_cand):
        if got == per_maze: break
        dfs_ok, it = item(ci, 0); n_tested += 1; n_dfs += dfs_ok
        if it is not None: out.append(it); got += 1
    for ci in rng.permutation(n_cand)[:per_maze_tree]:     # family 1: any spanning-tree swap, uniformly (robustness check)
        out.append(item(ci, 1)[1])
    return i, out, n_cand, n_tested, n_dfs


def build(n, per_maze, per_maze_tree, seed, limit=None, procs=None):
    S = np.load(SRC / f"g{n}.npz"); conn_a = S["conn_a"]; N = len(conn_a) if limit is None else limit
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(_job, [(i, conn_a[i], tuple(S["origin"][i]), tuple(S["target"][i]), n, seed, per_maze, per_maze_tree) for i in range(N)], chunksize=8)
    items = [it for _, o, *_ in sorted(res, key=lambda r: r[0]) for it in o]
    n_cand = np.array([r[2] for r in res]); n_tested = np.array([r[3] for r in res]); n_dfs = np.array([r[4] for r in res])
    L = n * n + 1
    def pad(key):
        a = np.full((len(items), L), -1, np.int16)
        for r, it in enumerate(items): a[r, : len(it[key])] = it[key]
        return a
    col = lambda k, dt=np.int32: np.array([it[k] for it in items], dtype=dt)
    arrs = dict(grid_n=n, src_sha=str(S["sha"]), seed=seed, per_maze=per_maze, per_maze_tree=per_maze_tree,
                maze=col("maze"), family=col("family"), rem=col("rem"), add=col("add"), conn_b=np.stack([it["conn_b"] for it in items]),
                j=col("j"), k_rem=col("k_rem"), k_door=col("k_door"), move_a=col("move_a"), move_b=col("move_b"),
                len_a=col("len_a"), len_b=col("len_b"), route_a=pad("route_a"), route_b=pad("route_b"), is_dfs=col("dfs", bool),
                n_candidates_per_maze=n_cand, n_tested_per_maze=n_tested, n_dfs_per_maze=n_dfs)
    arrs["sha"] = hashlib.sha1(arrs["conn_b"].tobytes() + arrs["maze"].tobytes() + arrs["family"].tobytes() + arrs["j"].tobytes() + arrs["move_b"].tobytes()).hexdigest()
    fam = arrs["family"]
    for fi, name in ((0, "dfs"), (1, "tree")):
        m = fam == fi
        print(f"[g{n}] {name}: {m.sum()} items from {len(set(arrs['maze'][m].tolist()))}/{N} mazes | wall lookahead 0: {np.mean((arrs['k_rem'] == arrs['j'])[m]):.2f} "
              f"door lookahead 0: {np.mean((arrs['k_door'] == arrs['j'])[m]):.2f} | shorter route {np.mean((arrs['len_b'] < arrs['len_a'])[m]):.2f} | is_dfs {arrs['is_dfs'][m].mean():.3f}", flush=True)
    print(f"[g{n}] route-changing swaps per maze {n_cand.mean():.1f}; DFS-preserving share of tested {n_dfs.sum() / max(1, n_tested.sum()):.3f}; {time.time() - t0:.0f}s sha {arrs['sha'][:12]}", flush=True)
    return arrs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per_maze", type=int, default=3); ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--grids", default="5,6,7,8,9,10"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--per_maze_tree", type=int, default=2)
    ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for g in [int(x) for x in a.grids.split(",")]:
        arrs = build(g, a.per_maze, a.per_maze_tree, a.seed, limit=a.limit)
        if not a.dry: np.savez_compressed(OUT / f"g{g}.npz", **arrs)
