#!/usr/bin/env python3
"""Decompose injection reliance by token region and run the random-injection
control on the baseline.

Regions (per sequence): adjacency-list tokens (randomly ordered -> mostly
irreducible entropy), and path tokens (origin/moves/<PATH_END> -- where world
state actually matters). Loss is next-token CE restricted to predicting the
tokens in each region.

Conditions:
  injected run:  features true / zeroed / shuffled-across-mazes
  baseline run:  no hook / true features through a FRESH random projection
                 (never seen in training -> pure perturbation control)

  python maze/core/reliance.py --run_dir <run> --kind injected|baseline
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import json
from pathlib import Path

import fire
import torch
from maze_dataset import SPECIAL_TOKENS
from maze_transformer.training.config import ConfigHolder  # noqa: F401

from maze.core.checkpoints import load_checkpoint
from maze.core.evals import ProbeSet, _forward
from maze.core.data import PretokDataset, find_pretok
from maze.core.injection import MazeInjector


def region_masks(ps: ProbeSet):
    n, L = ps.tokens.shape
    pos = torch.arange(L, device=ps.device)[None, :]
    valid = ps.attn[:, 1:].bool()  # predicting token at position t+1
    tgt_pos = pos[:, 1:]
    path_start_pos = (ps.origin_pos - 1)[:, None]
    adj = (tgt_pos >= 1) & (tgt_pos <= ps.end_pos[:, None]) & valid
    path = (tgt_pos > path_start_pos) & valid
    return {"all": valid, "adjacency": adj, "path": path}


@torch.no_grad()
def losses_by_region(model, injector, ps, conn):
    logits = _forward(model, injector, ps.tokens, ps.attn, conn, ps.end_pos)
    lp = logits[:, :-1].log_softmax(-1)
    ll = lp.gather(-1, ps.tokens[:, 1:][..., None])[..., 0]
    return {
        name: float(-(ll[m]).mean()) for name, m in region_masks(ps).items()
    }


def main(
    run_dir: str,
    kind: str = "injected",  # injected | baseline
    dataset_base_path: str = "/workspace/runs/baseline",
    grid_n: int = 6,
    val_size: int = 100,
    control_seed: int = 1234,
    dummy: bool | None = None,
    out: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    run_path = Path(run_dir)
    model, injector = load_checkpoint(
        run_path / "model.final.zanj",
        run_path / "injector.final.pt" if kind == "injected" else None,
        device,
    )
    model.eval()

    cfg = ConfigHolder.load(json.load(open(run_path / "config_holder.json")))
    vocab = cfg.tokenizer.get_vocab()
    pretok = PretokDataset(find_pretok(dataset_base_path))
    if dummy is None:
        dummy = "dummy" in run_path.name
    if dummy:
        pretok.dummify_(vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE])
    n = len(pretok)
    ps = ProbeSet(
        [pretok[i] for i in range(n - val_size, n)], vocab, grid_n, device
    )

    results = {}
    if kind == "injected":
        results["true"] = losses_by_region(model, injector, ps, ps.conn)
        results["zeroed"] = losses_by_region(
            model, injector, ps, torch.zeros_like(ps.conn)
        )
        perm = torch.randperm(len(ps.conn), device=device)
        results["shuffled"] = losses_by_region(model, injector, ps, ps.conn[perm])
    else:
        results["no_hook"] = losses_by_region(model, None, ps, ps.conn)
        rand_inj = MazeInjector(
            d_model=model.cfg.d_model, grid_n=grid_n, layer=0,
            scale=1.0, seed=control_seed,
        ).to(device)
        results["random_projection_true_features"] = losses_by_region(
            model, rand_inj, ps, ps.conn
        )

    print(f"\n{run_path.name} ({kind}), final checkpoint, {len(ps.conn)} held-out mazes")
    header = f"{'condition':>36} | {'all':>7} | {'adjacency':>9} | {'path':>7}"
    print(header)
    print("-" * len(header))
    for cond, r in results.items():
        print(f"{cond:>36} | {r['all']:7.4f} | {r['adjacency']:9.4f} | {r['path']:7.4f}")

    if out:
        payload = dict(
            run=run_path.name, kind=kind, dummy=bool(dummy),
            n_mazes=int(len(ps.conn)), conditions=results,
        )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
