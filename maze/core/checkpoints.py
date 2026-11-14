#!/usr/bin/env python3
"""Offline counterfactual eval across checkpoints of a (possibly injected) run.

For each checkpoint:
  - channel reliance + counterfactual next-token steering (injection_eval_suite)
  - greedy rollouts under TRUE injected features -> quality metrics
  - greedy rollouts under EDITED features, prompt unchanged:
      * block: the true first path edge is feature-blocked. In a spanning-tree
        maze this disconnects the target, so the honest success criterion is
        AVOIDANCE: the model should not cross the blocked edge.
      * open: a walled-off neighbor of the origin is feature-opened; rollouts
        are judged against the edited connectivity.

Note: for a baseline (non-injected) model the edited-feature conditions are
identical to true-feature conditions (no channel), so cf metrics ~ 0 — that is
the control.

Results: prints a table and writes counterfactual_results.json in run_dir.
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import json
import re
from pathlib import Path

import fire
import torch
from maze_dataset import SPECIAL_TOKENS
from maze_transformer.training.config import ConfigHolder  # noqa: F401  (registers the ZANJ model class)
from zanj import ZANJ

from maze.core.evals import (
    ProbeSet,
    greedy_rollout,
    injection_eval_suite,
    rollout_metrics,
)
from maze.core.data import PretokDataset, find_pretok
from maze.core.injection import InjectionRuntime, MazeInjector


def load_checkpoint(
    model_path: Path,
    injector_path: Path | None,
    device,
    teacher_model: str | None = None,
    teacher_probe: str | None = None,
):
    """Returns (model, injection_runtime|None). The injector checkpoint carries
    its variant metadata (mode / window_k / teacher paths), so every eval path
    wires the right hooks without knowing which run it is looking at."""
    model = ZANJ().read(model_path)
    model.to(device)
    model.eval()
    runtime = None
    if injector_path is not None and injector_path.exists():
        d = torch.load(injector_path, weights_only=True)
        w = d["W_inj"]
        injector = MazeInjector(
            d_model=w.shape[1], grid_n=int((w.shape[0] // 2) ** 0.5),
            layer=int(d.get("layer", 0)), scale=float(d.get("scale", 1.0)),
        )
        injector.W_inj.data = w
        injector.to(device)
        mode = d.get("mode", "complete")
        teacher = None
        if mode == "teacher":
            from maze.core.teacher import load_teacher

            tm = teacher_model or d.get("teacher_model")
            tp = teacher_probe or d.get("teacher_probe")
            assert tm and tp, f"teacher paths missing in {injector_path}"
            teacher = load_teacher(tm, tp, device)
        runtime = InjectionRuntime(
            injector, mode=mode, window_k=int(d.get("window_k", 4)),
            teacher=teacher,
        )
    return model, runtime


def eval_checkpoint(model, injector, ps: ProbeSet) -> dict:
    out = {}
    idx = torch.arange(len(ps.conn), device=ps.device)
    if injector is not None:
        out.update(injection_eval_suite(model, injector, ps))

    rolls_true = greedy_rollout(model, injector, ps, ps.conn)
    out.update({
        f"true/{k.split('/')[-1]}": v
        for k, v in rollout_metrics(ps, rolls_true, ps.conn).items()
    })

    def crossed_blocked(rolls):
        crossed = 0
        for i, gen in enumerate(rolls):
            if gen and gen[0] == int(ps.first_move[i]):
                crossed += 1
        return crossed / max(1, len(rolls))

    out["true/first_move_is_true_move_frac"] = crossed_blocked(rolls_true)

    if injector is not None:
        conn_block = ps.conn.clone()
        conn_block[idx, ps.first_feat] = False
        rolls_block = greedy_rollout(model, injector, ps, conn_block)
        out["block/crossed_blocked_edge_frac"] = crossed_blocked(rolls_block)
        out.update({
            f"block/{k.split('/')[-1]}": v
            for k, v in rollout_metrics(ps, rolls_block, conn_block).items()
        })

        has_open = ps.open_feat >= 0
        conn_open = ps.conn.clone()
        conn_open[idx[has_open], ps.open_feat[has_open]] = True
        rolls_open = greedy_rollout(model, injector, ps, conn_open)
        out.update({
            f"open/{k.split('/')[-1]}": v
            for k, v in rollout_metrics(ps, rolls_open, conn_open).items()
        })
        used = 0
        for i in idx[has_open].tolist():
            if rolls_open[i] and rolls_open[i][0] == int(ps.open_id[i]):
                used += 1
        out["open/first_move_uses_opened_edge_frac"] = used / max(
            1, int(has_open.sum())
        )
    return out


def main(
    run_dir: str,
    dataset_base_path: str = "/workspace/runs/baseline",
    grid_n: int = 6,
    val_size: int = 100,
    only_final: bool = False,
    dummy: bool | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    run_path = Path(run_dir)

    pretok = PretokDataset(find_pretok(dataset_base_path))

    # need vocab: rebuild tokenizer from the run's saved config
    cfg = ConfigHolder.load(json.load(open(run_path / "config_holder.json")))
    vocab = cfg.tokenizer.get_vocab()
    if dummy is None:
        dummy = "dummy" in run_path.name  # variant C models saw blanked prompts
    if dummy:
        pretok.dummify_(vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE])
    n = len(pretok)
    ps = ProbeSet(
        [pretok[i] for i in range(n - val_size, n)], vocab, grid_n, device
    )

    ckpts = []
    if not only_final:
        for p in sorted(
            (run_path / "checkpoints").glob("model.iter_*.zanj"),
            key=lambda p: int(re.search(r"iter_(\d+)", p.name).group(1)),
        ):
            it = int(re.search(r"iter_(\d+)", p.name).group(1))
            ckpts.append((it, p, run_path / "checkpoints" / f"injector.iter_{it}.pt"))
    final_inj = run_path / "injector.final.pt"
    ckpts.append((-1, run_path / "model.final.zanj", final_inj))

    results = {}
    for it, mp, ip in ckpts:
        if not mp.exists():
            continue
        model, injector = load_checkpoint(mp, ip, device)
        res = eval_checkpoint(model, injector, ps)
        key = "final" if it == -1 else str(it)
        results[key] = res
        headline = {
            k: round(v, 4) for k, v in res.items()
            if k in (
                "inj_eval/reliance_shuffled", "inj_eval/cf_block_drop",
                "true/reach_target_frac", "block/crossed_blocked_edge_frac",
                "open/first_move_uses_opened_edge_frac",
            )
        }
        print(f"ckpt {key}: {headline}")
        del model

    out_path = run_path / "counterfactual_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
