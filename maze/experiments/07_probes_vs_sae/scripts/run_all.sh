#!/bin/bash
# Pod-side driver: probes at the storage site, steering sweep, SAE training + feature steering.
# Run from /workspace/maze_injection (the backup modules must be importable from there).
set -uo pipefail
cd /workspace/maze_injection
export PYTHONUNBUFFERED=1
LOG=/workspace/runs/wm_run.log
mkdir -p /workspace/runs
{
echo "=== $(date) data";     python wm_probes.py data  --n_mazes 6000 --seed 777
echo "=== $(date) acts";     python wm_probes.py acts
echo "=== $(date) train";    python wm_probes.py train
echo "=== $(date) steer sweep (subsample 400/family)"; python wm_probes.py steer --subsample 400 --layers 0,1,2,3,5
echo "=== $(date) sae train L0"; python wm_sae.py train --layer 0 --expansion 8 --l1 3e-3 --steps 20000
echo "=== $(date) sae features L0"; python wm_sae.py features --layer 0
echo "=== $(date) sae steer L0"; python wm_sae.py steer --layer 0 --subsample 400
echo "=== $(date) sae train L1"; python wm_sae.py train --layer 1 --expansion 8 --l1 3e-3 --steps 20000
echo "=== $(date) sae features L1"; python wm_sae.py features --layer 1
echo "=== $(date) sae steer L1"; python wm_sae.py steer --layer 1 --subsample 400
echo "=== $(date) ALL_DONE"
} 2>&1 | grep --line-buffered -vE "Warning|warnings.warn" >> "$LOG"
