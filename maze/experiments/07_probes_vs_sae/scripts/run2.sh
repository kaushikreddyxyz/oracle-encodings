#!/bin/bash
# Pod-side driver v2 (site-aware): probes at A/B/; sites, steering sweep, SAE l1 sweep + feature steering.
set -uo pipefail
cd /workspace/maze_injection
export PYTHONUNBUFFERED=1
LOG=/workspace/runs/wm_run2.log
{
echo "=== $(date) acts";  python wm_probes.py acts --n_use 3000
echo "=== $(date) train"; python wm_probes.py train
echo "=== $(date) steer sweep"; python wm_probes.py steer --subsample 400 --layers 0,1,2,3,6
for L1 in 1e-2 3e-2 1e-1; do
  T="_l1${L1}"
  echo "=== $(date) sae train L0 l1=$L1"; python wm_sae.py train --layer 0 --expansion 8 --l1 $L1 --steps 20000 --tag "$T"
  echo "=== $(date) sae features L0 $T (site B)"; python wm_sae.py features --layer 0 --site B --tag "$T"
  echo "=== $(date) sae features L0 $T (site semi)"; python wm_sae.py features --layer 0 --site semi --tag "$T"
  echo "=== $(date) sae steer L0 $T (site B)"; python wm_sae.py steer --layer 0 --site B --subsample 400 --tag "$T"
done
echo "=== $(date) sae train L1 l1=3e-2"; python wm_sae.py train --layer 1 --expansion 8 --l1 3e-2 --steps 20000 --tag "_l13e-2"
echo "=== $(date) sae features L1"; python wm_sae.py features --layer 1 --site B --tag "_l13e-2"
echo "=== $(date) sae steer L1"; python wm_sae.py steer --layer 1 --site B --subsample 400 --tag "_l13e-2"
echo "=== $(date) ALL_DONE"
} 2>&1 | grep --line-buffered -vE "Warning|warnings.warn|Moving model" >> "$LOG"
