"""Shared library for the maze injection experiments.

Modules are imported by absolute path, e.g.

    from maze.core.data import PretokDataset, pretokenize
    from maze.core.injection import InjectionRuntime, MazeInjector
    from maze.core.evals import ProbeSet, greedy_rollout
    from maze.core.checkpoints import load_checkpoint
    from maze.core.reliance import region_masks

Nothing is re-exported here on purpose: the submodules pull in torch,
maze_dataset and maze_transformer, and several callers (plot scripts) must
stay importable without them.
"""
