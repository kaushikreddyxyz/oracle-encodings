#!/usr/bin/env python3
"""Validate MazeFeatureComputer against ground-truth connection lists.

Checks, over a batch of generated mazes:
  1. "complete" mode: features at every position from <ADJLIST_END> onward
     exactly reconstruct maze.connection_list; all earlier positions are 0.
  2. "incremental" mode: features at <ADJLIST_END> equal the full maze;
     revealed connections accumulate monotonically.
  3. Injected forward pass runs and changes logits (hook is live).
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import numpy as np
import torch
from maze_dataset import MazeDataset, MazeDatasetConfig, SPECIAL_TOKENS
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import BaseGPTConfig, ConfigHolder, TRAINING_CONFIGS

from maze.core.injection import MazeFeatureComputer, MazeInjector

GRID_N = 6
N = 50

cfg = ConfigHolder(
    name="test",
    dataset_cfg=MazeDatasetConfig(
        name="t", grid_n=GRID_N, n_mazes=N,
        maze_ctor=LatticeMazeGenerators.gen_dfs, seed=42,
    ),
    model_cfg=BaseGPTConfig(
        name="tiny", act_fn="gelu", d_model=64, d_head=16, n_layers=2,
    ),
    train_cfg=TRAINING_CONFIGS["test-v1"],
)
dataset = MazeDataset.from_config(
    cfg.dataset_cfg, do_generate=True, load_local=False, save_local=False,
)
tokenizer = cfg.tokenizer
device = torch.device("cpu")
fc = MazeFeatureComputer(tokenizer, GRID_N, device)
vocab = tokenizer.get_vocab()
adjlist_end_id = vocab[SPECIAL_TOKENS.ADJLIST_END]

for maze in dataset:
    toks_str = " ".join(maze.as_tokens(cfg.maze_tokenizer))
    tokens = torch.tensor(
        [tokenizer(toks_str)["input_ids"]], dtype=torch.long
    ) if callable(tokenizer) else None
    if tokens is None:
        raise RuntimeError("tokenizer not callable")

    truth = torch.from_numpy(
        maze.connection_list.astype(np.float32).flatten()
    ) * 2.0 - 1.0

    row = tokens[0].tolist()
    end_pos = row.index(adjlist_end_id)

    feats_c = fc.features(tokens, "complete")[0]
    assert torch.equal(feats_c[end_pos], truth), "complete: mismatch at ADJLIST_END"
    assert torch.equal(feats_c[-1], truth), "complete: mismatch at final pos"
    assert feats_c[:end_pos].abs().sum() == 0, "complete: nonzero before ADJLIST_END"

    feats_i = fc.features(tokens, "incremental")[0]
    assert torch.equal(feats_i[end_pos], truth), "incremental: full state mismatch"
    # monotone: once revealed (+1), never un-revealed
    active = feats_i.abs().sum(dim=-1) > 0
    revealed = (feats_i == 1.0) & active.unsqueeze(-1)
    assert (revealed.int().cummax(dim=0).values >= revealed.int()).all(), (
        "incremental: revealed connection disappeared"
    )

print(f"feature computation OK on {N} mazes")

model = cfg.create_model_zanj().to(device)
injector = MazeInjector(d_model=64, grid_n=GRID_N, layer=0, scale=1.0)
maze = dataset[0]
toks = torch.tensor(
    [tokenizer(" ".join(maze.as_tokens(cfg.maze_tokenizer)))["input_ids"]]
)
logits_plain = model(toks, return_type="logits")
logits_inj = injector.forward_with_injection(model, toks, fc, return_type="logits")
delta = (logits_plain - logits_inj).abs().max().item()
assert delta > 0, "injection had no effect on logits"
print(f"injection hook live, max logit delta {delta:.3f}")
print("ALL TESTS PASSED")
