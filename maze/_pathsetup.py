"""sys.path shim: the ONE import mechanism for every runnable script in maze/.

Why it exists
-------------
Shared code lives in ``maze/core/`` and is imported absolutely
(``from maze.core.data import PretokDataset``), which needs the repo root on
sys.path.  Experiment scripts additionally import each other across experiment
folders (03/04/06 all use symbols from 02's ``build_counterfactual_set``), and
the experiment directories are digit-prefixed (``02_counterfactual_edits``) so
they can never be Python packages -- ``python -m`` and dotted imports are out.

So: every runnable script starts with the SAME 5-line header, which locates the
repo root by walking up to the directory containing ``maze/core`` and then
imports this module, which puts the repo root plus every
``experiments/*/scripts``, ``experiments/*/plots`` and ``legacy`` directory on
sys.path.  Cross-experiment imports therefore keep working unchanged, by bare
module name.

    import pathlib as _pl
    import sys as _sys

    _sys.path.insert(0, str(next(
        p for p in _pl.Path(__file__).resolve().parents
        if (p / "maze" / "core").is_dir()
    )))
    import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

Scripts that also need to reach another experiment's results/figures on disk
import ``MAZE_ROOT`` from here instead of the bare module.

Invocation: run scripts by path from anywhere, e.g.

    python3 maze/experiments/02_counterfactual_edits/scripts/build_counterfactual_set.py --help

(``python -m`` is NOT supported -- see the digit-prefix note above.)
"""

import sys as _sys
from pathlib import Path as _Path

MAZE_ROOT = _Path(__file__).resolve().parent
REPO_ROOT = MAZE_ROOT.parent
EXPERIMENTS = MAZE_ROOT / "experiments"


def _candidate_dirs():
    yield REPO_ROOT
    if EXPERIMENTS.is_dir():
        for exp in sorted(EXPERIMENTS.iterdir()):
            if not exp.is_dir():
                continue
            for sub in ("scripts", "plots"):
                if (exp / sub).is_dir():
                    yield exp / sub
    if (MAZE_ROOT / "legacy").is_dir():
        yield MAZE_ROOT / "legacy"


for _d in _candidate_dirs():
    _s = str(_d)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
del _d, _s
