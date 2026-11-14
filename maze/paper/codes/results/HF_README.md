# maze-graft-codes

Final checkpoints of the graft-code campaign (maze-solving transformers, jirpyish d256/12L, 2M random-DFS mazes, prompt intact,
graft injected at block-0 input from `<ORIGIN_END>`). Folder = `g<grid>-s<seed>-<code>-e<epochs>-oe/<run>/` with
`model.final.pt` (torch, `maze/core/tl_bridge.load_torch_ckpt`), `injector.final.pt` (`maze/core/graft.load_injector`) and `config_holder.json`.
Codes: edge = allocentric +-1 per wall; ego = egocentric walls relative to the current cell (dynamic); base = no graft.
e8 = the e4 run continued for 4 more epochs (weights warm-started, AdamW moments reset). Eval results stay in the repo, not here.

| folder | files |
|---|---|
| g10-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, model.final.pt |
| g10-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s0-edge-e8-oe/fast-inj-fixed-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s0-ego-e8-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s1-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s1-edge-e8-oe/fast-inj-fixed-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s1-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s1-ego-e8-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s2-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s2-edge-e8-oe/fast-inj-fixed-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s2-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s2-ego-e8-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s3-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s3-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s4-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g10-s4-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g10-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, model.final.pt |
| g5-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s1-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s1-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s2-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s2-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s3-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s3-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s4-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g5-s4-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g5-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s0-axdeg-e4-oe/fast-inj-fixed-axdeg-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s0-axdeg1h-e4-oe/fast-inj-fixed-axdeg1h-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s0-axdeg1h-e4-oe/fast-inj-trainable-axdeg1h-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, model.final.pt |
| g6-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s0-ego-e4-oe/fast-inj-trainable-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s1-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s1-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s2-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s2-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s3-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s3-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s4-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g6-s4-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g6-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, model.final.pt |
| g7-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s1-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s1-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s2-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s2-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s3-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s3-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s4-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g7-s4-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g7-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, model.final.pt |
| g8-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s1-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s1-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s2-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s2-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s3-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s3-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s4-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g8-s4-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g8-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s0-base-e4-oe/fast-baseline_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, model.final.pt |
| g9-s0-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s0-edge-e8-oe/fast-inj-fixed-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s0-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s0-ego-e8-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s1-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s1-edge-e8-oe/fast-inj-fixed-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s1-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s1-ego-e8-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s2-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s2-edge-e8-oe/fast-inj-fixed-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s2-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s2-ego-e8-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0-cont1_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s3-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s3-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s4-edge-e4-oe/fast-inj-fixed-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
| g9-s4-ego-e4-oe/fast-inj-fixed-ego-complete-oe-L0-s1.0_jirpyish-d256-l12_dfs-g9-n2000000 | config_holder.json, injector.final.pt, model.final.pt |
