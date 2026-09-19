#!/usr/bin/env python3
"""Solve rate of grafted models as a function of token-maze / graft-maze dissimilarity.

Every item = (token-space maze A in the prompt, graft-space maze B in the injected
features, A's origin and target). B sits at a controlled Hamming distance from A over
the 2n(n-1) free connection features (maze_distance.py). Levels per base maze:
  0            B == A
  1..K         one DFS-variant per normalised-distance bucket ((k-1)w, kw]
  K+1          an independent fresh random-DFS maze (whatever distance it lands at)
Greedy rollouts are judged against B (the grafted world) and, for reference, against A.

  python eval_distance.py build --grid_n 6 --cfg_json <run>/config_holder.json
  python eval_distance.py run --run_dirs a,b,c --out_dir /workspace/dist_out
"""
import pathlib as _pl, sys as _sys
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(next(p for p in _HERE.parents if (p / "maze" / "core").is_dir())))
_sys.path.insert(0, str(_HERE))
import maze._pathsetup  # noqa
import hashlib, json, os, re, time
from multiprocessing import Pool
from pathlib import Path
import fire, numpy as np, torch
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder
from maze.core.data import PretokDataset, pretokenize
from eval_ckpts import feat_idx, shortest_path, greedy_kv, load_ckpt
from maze_distance import build_variants, fresh_dfs, hamming, n_free, bucket_of

SET_DIR = Path("/workspace/distsets")


def _variants_job(args):
    i, conn, n, seed, K, width = args
    rng = np.random.default_rng([seed, n, i])
    got, _ = build_variants(conn, n, rng, K, width)
    ind = fresh_dfs(rng, n)
    return i, got, ind


def build(grid_n: int, cfg_json: str, n_mazes: int = 3000, seed: int = 4321, n_buckets: int = 12,
          width: float = 0.05, procs: int | None = None):
    n = grid_n
    d = SET_DIR / f"g{n}-s{seed}-n{n_mazes}"
    out = d / f"distset_K{n_buckets}_w{width}.npz"
    if out.exists():
        return out
    d.mkdir(parents=True, exist_ok=True)
    cfg = ConfigHolder.load(json.load(open(cfg_json)))
    vocab = cfg.tokenizer.get_vocab()
    pre = d / "pretok.pt"
    if not pre.exists():
        ds_cfg = MazeDatasetConfig(name="distset", grid_n=n, n_mazes=n_mazes,
                                   maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed)
        dataset = MazeDataset.from_config(ds_cfg, do_generate=True, load_local=False, save_local=False,
                                          gen_parallel=False, verbose=False)
        pretokenize(dataset, cfg.maze_tokenizer, vocab, pre, bos_id=int(cfg.tokenizer.bos_token_id), workers=0)
    pretok = PretokDataset(pre)
    rows = [pretok[i] for i in range(len(pretok))]
    conn = np.stack([c.numpy() for _, c, _ in rows]).astype(bool)
    end_pos = np.array([int(e) for _, _, e in rows])
    ps_id, pe_id = vocab[SPECIAL_TOKENS.PATH_START], vocab[SPECIAL_TOKENS.PATH_END]
    coord_re = re.compile(r"^\((\d+),(\d+)\)$")
    id_to_coord = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items() if (m := coord_re.match(t))}
    seqs = [s.tolist() for s, _, _ in rows]
    prompts, origin, target, len_a = [], [], [], []
    for s in seqs:
        ps, pe = s.index(ps_id), s.index(pe_id)
        prompts.append(s[: ps + 2])
        path = [id_to_coord[t] for t in s[ps + 1: pe]]
        origin.append(path[0]); target.append(path[-1]); len_a.append(len(path) - 1)
    L0 = len(prompts[0]); assert all(len(p) == L0 for p in prompts)
    t0 = time.time()
    with Pool(procs or max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(_variants_job, [(i, conn[i], n, seed, n_buckets, width) for i in range(len(conn))], chunksize=16)
    maze, level, conn_b, dist, eps = [], [], [], [], []
    for i, got, ind in sorted(res, key=lambda r: r[0]):
        maze.append(i); level.append(0); conn_b.append(conn[i]); dist.append(0); eps.append(0.0)
        for k in sorted(got):
            b, dd, e = got[k]
            maze.append(i); level.append(k); conn_b.append(b); dist.append(dd); eps.append(e)
        maze.append(i); level.append(n_buckets + 1); conn_b.append(ind); dist.append(hamming(conn[i], ind, n)); eps.append(1.0)
    maze = np.array(maze); conn_b = np.stack(conn_b)
    # route descriptors (not part of the metric; kept for secondary cuts)
    len_b = np.zeros(len(maze), int); path_same = np.zeros(len(maze), bool)
    for j, i in enumerate(maze):
        pa = shortest_path(conn[i], origin[i], target[i], n)
        pb = shortest_path(conn_b[j], origin[i], target[i], n)
        len_b[j] = len(pb) - 1; path_same[j] = pa == pb
    assert all(bucket_of(dd, n, width) == k for dd, k in zip(dist, level) if 0 < k <= n_buckets)
    sha = hashlib.sha1(conn.tobytes() + conn_b.tobytes() + np.array(prompts).tobytes()).hexdigest()
    np.savez_compressed(out, grid_n=n, conn_a=conn, prompts=np.array(prompts), end_pos=end_pos,
                        origin=np.array(origin), target=np.array(target), len_a=np.array(len_a),
                        maze=maze, level=np.array(level), conn_b=conn_b, dist=np.array(dist), eps=np.array(eps),
                        len_b=len_b, path_same=path_same, n_free=n_free(n), width=width, n_buckets=n_buckets,
                        sha=sha, path_end_id=pe_id, pad_id=vocab[SPECIAL_TOKENS.PADDING],
                        coord_ids=np.array([[i, r, c] for i, (r, c) in id_to_coord.items()]))
    cnt = np.bincount(np.array(level), minlength=n_buckets + 2)
    print(f"[build g{n}] {len(maze)} items, per-level {cnt.tolist()}, variants {time.time() - t0:.0f}s, sha {sha[:12]}", flush=True)
    return out


def judge(coords, conn, n, target):
    bad = 0
    for a, b in zip(coords[:-1], coords[1:]):
        if not (abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and conn[feat_idx(a, b, n)]):
            bad += 1
    reached = len(coords) > 1 and coords[-1] == target
    return reached, bad


def run_one(run_dir: Path, out_dir: Path, device, batch: int, **bk):
    grid_n = int(re.search(r"dfs-g(\d+)-", run_dir.name).group(1))
    seed_tag = run_dir.parent.name
    cond = "fixed" if "inj-fixed" in run_dir.name else "trainable" if "inj-trainable" in run_dir.name else "base"
    out = out_dir / f"g{grid_n}__{cond}__{seed_tag}.npz"
    if out.exists():
        return
    S = np.load(build(grid_n, str(run_dir / "config_holder.json"), **bk))
    n = grid_n
    model, injector = load_ckpt(run_dir / "model.final.pt", run_dir / "injector.final.pt", device)
    assert injector is not None
    layer = int(injector.layer)
    id_to_coord = {int(i): (int(r), int(c)) for i, r, c in S["coord_ids"]}
    pe_id, pad_id = int(S["path_end_id"]), int(S["pad_id"])
    maze, conn_b, conn_a = S["maze"], S["conn_b"], S["conn_a"]
    origin_c, target_c, lv = S["origin"], S["target"], S["level"]
    prompts = torch.tensor(S["prompts"], device=device)
    end_pos = torch.tensor(S["end_pos"], device=device, dtype=torch.long)
    max_new = n * n + 2
    N = len(maze)
    solved_b = np.zeros(N, bool); reached = np.zeros(N, bool); nbad_b = np.zeros(N, np.int16)
    solved_a = np.zeros(N, bool); nbad_a = np.zeros(N, np.int16); rlen = np.zeros(N, np.int16); malformed = np.zeros(N, bool)
    t0 = time.time()
    for lo in range(0, N, batch):
        idx = maze[lo: lo + batch]
        it = torch.tensor(idx, device=device)
        gen = greedy_kv(model, injector, layer, prompts[it], torch.tensor(conn_b[lo: lo + batch], device=device),
                        end_pos[it], max_new, pe_id, pad_id).cpu().numpy()
        for k, i in enumerate(idx):
            g = gen[k].tolist()
            for stop in (pe_id, pad_id):
                if stop in g: g = g[: g.index(stop)]
            j = lo + k
            malformed[j] = any(t not in id_to_coord for t in g)
            coords = [tuple(origin_c[i])] + [id_to_coord[t] for t in g if t in id_to_coord]
            tgt = tuple(target_c[i])
            r, bb = judge(coords, conn_b[j], n, tgt)
            _, ba = judge(coords, conn_a[i], n, tgt)
            reached[j] = r; nbad_b[j] = bb; nbad_a[j] = ba; rlen[j] = len(coords) - 1
            solved_b[j] = r and bb == 0 and not malformed[j]
            solved_a[j] = r and ba == 0 and not malformed[j]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, grid_n=n, cond=cond, seed_tag=seed_tag, run=run_dir.name, set_sha=str(S["sha"]),
                        solved_graft=solved_b, solved_token=solved_a, reached=reached, n_bad_graft=nbad_b,
                        n_bad_token=nbad_a, rollout_len=rlen, malformed=malformed)
    print(f"[g{n} {cond} {seed_tag}] {N} rollouts {time.time() - t0:.0f}s | solved vs graft by level: "
          + " ".join(f"{solved_b[lv == k].mean():.2f}" for k in range(int(lv.max()) + 1)), flush=True)


def run(run_dirs: str, out_dir: str = "/workspace/dist_out", batch: int = 2000, n_mazes: int = 3000, seed: int = 4321,
        n_buckets: int = 12, width: float = 0.05):
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    dirs = run_dirs if isinstance(run_dirs, (list, tuple)) else str(run_dirs).split(",")
    for d in dirs:
        run_one(Path(d), Path(out_dir), device, batch, n_mazes=n_mazes, seed=seed, n_buckets=n_buckets, width=width)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    fire.Fire(dict(build=build, run=run))
