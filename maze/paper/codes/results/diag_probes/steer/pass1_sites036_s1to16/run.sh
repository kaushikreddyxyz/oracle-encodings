#!/bin/bash
cd /workspace
R=/workspace/hf/runs
O=/workspace/steer_out
python maze/paper/codes/steer_probe.py --run_dir $R/g6-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g6-n2000000 --grid_n 6 --label g6_base --out_dir $O
python maze/paper/codes/steer_probe.py --run_dir $R/g6-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 --grid_n 6 --label g6_edge --out_dir $O
python maze/paper/codes/steer_probe.py --run_dir $R/g6-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 --grid_n 6 --label g6_ego --out_dir $O
python maze/paper/codes/steer_probe.py --run_dir $R/g8-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 --grid_n 8 --label g8_edge --out_dir $O
python maze/paper/codes/steer_probe.py --run_dir $R/g8-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 --grid_n 8 --label g8_ego --out_dir $O
echo ALLDONE
