#!/bin/bash
cd /workspace
R=/workspace/hf/runs
O=/workspace/steer_out2
A="--sites 0,6 --s_list 1,2,4,8,16,32,64 --out_dir $O"
python maze/paper/codes/steer_probe.py --run_dir $R/g6-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g6-n2000000 --grid_n 6 --label g6_base $A
python maze/paper/codes/steer_probe.py --run_dir $R/g6-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 --grid_n 6 --label g6_edge $A
python maze/paper/codes/steer_probe.py --run_dir $R/g6-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 --grid_n 6 --label g6_ego $A
echo ALLDONE
