#!/usr/bin/env python3
"""For a grafted run's FINAL checkpoint: roll out in the phantom world (channel = phantom
features) and judge every rollout against BOTH the phantom maze and the true (token-space)
maze, with per-edit counts. Answers "does the grafted model solve the original maze instead?"

  python maze/paper/curves/eval_phantom_vs_true.py --run_dir <run> --grid_n 6
Writes <run>/eval/phantom_vs_true.json + per-maze npz.
"""
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "maze" / "core").is_dir())))
import maze._pathsetup  # noqa
import json, time
from pathlib import Path
import fire, numpy as np, torch
from maze_transformer.training.config import ConfigHolder
from eval_ckpts import EvalSet, Scorer, load_ckpt, greedy_kv, feat_idx


def judge(coords, conn, n, target):
    """coords: rollout cells (starting at origin). Returns dict of flags for world `conn`."""
    ok = True; bad = 0
    for a, b in zip(coords[:-1], coords[1:]):
        adj = abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
        if not (adj and conn[feat_idx(a, b, n)]):
            ok = False; bad += 1
    reached = len(coords) > 1 and coords[-1] == target
    return dict(reached=reached, clean=ok, solved=ok and reached, n_bad=bad)


def main(run_dir: str, grid_n: int, eval_dir: str = "/workspace/evalsets", n_mazes: int = 2000, seed: int = 4321,
         n_walls: int = 2, n_doors: int = 2, batch: int = 2000, max_new: int | None = None):
    device = torch.device("cuda"); torch.set_float32_matmul_precision("high")
    run = Path(run_dir); cfg = ConfigHolder.load(json.load(open(run / "config_holder.json")))
    es = EvalSet(cfg, grid_n, Path(eval_dir), n_mazes, seed, n_walls, n_doors, device)
    if max_new is None: max_new = grid_n * grid_n + 2
    model, injector = load_ckpt(run / "model.final.pt", run / "injector.final.pt", device)
    assert injector is not None, "grafted runs only"
    sc = Scorer(model, injector, device)
    edits = json.load(open(Path(eval_dir) / f"g{grid_n}-s{seed}-n{n_mazes}" / f"phantom_w{n_walls}_d{n_doors}.json"))
    items = es.worlds["phantom"]; n = grid_n
    recs = []
    t0 = time.time()
    for lo in range(0, len(items), batch):
        chunk = items[lo:lo+batch]; B = len(chunk)
        prompts = [es.seqs[i][: int(es.origin_pos[i]) + 1] for i, _, _ in chunk]
        L0 = len(prompts[0]); assert all(len(p) == L0 for p in prompts)
        toks = torch.tensor(prompts, device=device)
        conn_t = torch.tensor(np.stack([c for _, c, _ in chunk]), dtype=torch.bool, device=device)
        end_t = torch.tensor([int(es.end_pos[i]) for i, _, _ in chunk], dtype=torch.long, device=device)
        gen = greedy_kv(sc.model, sc.injector, sc.layer, toks, conn_t, end_t, max_new, es.path_end_id, es.pad_id).cpu().numpy()
        for k, (i, conn_ph, path_ph) in enumerate(chunk):
            g = gen[k].tolist()
            for stop in (es.path_end_id, es.pad_id):
                if stop in g: g = g[: g.index(stop)]
            coords = [path_ph[0]] + [es.id_to_coord[t] for t in g if t in es.id_to_coord]
            malformed = any(t not in es.id_to_coord for t in g)
            target = path_ph[-1]
            jp = judge(coords, conn_ph, n, target); jt = judge(coords, es.conn[i], n, target)
            e = edits[str(i)]; walls = set(e["walls"]); doors = set(e["doors"])
            used = [feat_idx(a, b, n) for a, b in zip(coords[:-1], coords[1:]) if abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1]
            true_path = es.true_path[i]
            recs.append(dict(maze=i, malformed=malformed, reached=jp["reached"],
                             clean_phantom=jp["clean"], solved_phantom=jp["solved"], n_bad_phantom=jp["n_bad"],
                             clean_true=jt["clean"], solved_true=jt["solved"], n_bad_true=jt["n_bad"],
                             crossed_phantom_wall=int(sum(f in walls for f in used)), used_phantom_door=int(sum(f in doors for f in used)),
                             equals_true_path=coords == true_path, equals_phantom_path=coords == path_ph,
                             len_rollout=len(coords)-1, len_true=len(true_path)-1, len_phantom=len(path_ph)-1))
    df = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    m = {k: float(np.mean(v)) for k, v in df.items() if v.dtype != object and k != "maze"}
    both = (df["solved_true"] & ~df["solved_phantom"]).mean()
    summary = dict(run=run.name, grid_n=grid_n, n=len(recs), **m,
                   solved_true_not_phantom=float(both),
                   solved_neither=float((~df["solved_true"] & ~df["solved_phantom"]).mean()),
                   reached_not_clean_phantom=float((df["reached"] & ~df["clean_phantom"]).mean()),
                   of_reached_not_clean__solved_true=float(df["solved_true"][df["reached"] & ~df["clean_phantom"]].mean()) if (df["reached"] & ~df["clean_phantom"]).any() else float("nan"),
                   # among rollouts that break a phantom rule: are ALL the bad moves phantom-wall crossings (i.e. legal in the
                   # original maze), or does the rollout also break rules of the original maze (arbitrary breaks)?
                   breakers=float((~df["clean_phantom"]).mean()),
                   of_breakers__all_breaks_are_phantom_walls=float((df["n_bad_true"] == 0)[~df["clean_phantom"]].mean()) if (~df["clean_phantom"]).any() else float("nan"),
                   of_breakers__some_arbitrary_break=float((df["n_bad_true"] > 0)[~df["clean_phantom"]].mean()) if (~df["clean_phantom"]).any() else float("nan"),
                   of_breakers__mean_phantom_walls_crossed=float(df["crossed_phantom_wall"][~df["clean_phantom"]].mean()) if (~df["clean_phantom"]).any() else float("nan"),
                   arbitrary_break_rate_any=float((df["n_bad_true"] > 0).mean()),
                   crossed_any_wall=float((df["crossed_phantom_wall"] > 0).mean()), used_any_door=float((df["used_phantom_door"] > 0).mean()),
                   used_both_doors=float((df["used_phantom_door"] >= 2).mean()), seconds=time.time()-t0)
    out = run / "eval"; out.mkdir(exist_ok=True)
    json.dump(summary, open(out / "phantom_vs_true.json", "w"), indent=1)
    np.savez_compressed(out / "phantom_vs_true_per_maze.npz", **{k: v for k, v in df.items()})
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}))


if __name__ == "__main__":
    fire.Fire(main)
