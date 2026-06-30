# STATE — overnight concept-probes run (resumable checklist)

`state.json` is the machine source of truth; this mirrors it for humans. On startup the
orchestrator reads both and resumes from the first incomplete unit. Status: ⬜ todo /
🟡 running / ✅ done+pushed / ❌ failed / ⏭️ skipped (reason).

**Run start (UTC):** 2026-06-27T01:58:58Z · **deadline:** +9h · **budget stop:** $320 · **start balance:** $415.22

## ⚠️ BLOCKER (needs the user — 30s manual action)
Both Gemma repos are license-gated and **not accepted** by `kaushikreddyxyz`:
- ⬜ `google/gemma-2-9b` — accept at https://huggingface.co/google/gemma-2-9b  (probe target, Steps 2-4)
- ⬜ `google/gemma-3-27b-it` — accept at https://huggingface.co/google/gemma-3-27b-it  (judge, Step 1)

Until accepted: labeling falls back to a non-gated judge (Qwen2.5-32B-Instruct-AWQ);
probing is **blocked** (Gemma Scope SAEs only exist for gemma-2 → no clean substitute).
Orchestrator re-checks access on a timer and switches to the designed Gemma path the
moment access appears.

## Pre-GPU (cheap, local) — DONE
- ✅ runpodctl auth + balance ($415.22, no pods running)
- ✅ HF auth via cached login (account kaushikreddyxyz)
- ✅ repo-id resolution (gemma-2-9b/gemma-3-27b-it exist but gated; gemma-4 absent; SAEs+climbmix OK)
- ✅ shard provenance: train used ~0-183, val=6542 → disjoint = shards 300-309
- ✅ concept registry (concepts.py): 68 presence + 11 scalar probes
- ✅ config.py, SPEC.md, state.json scaffolded

## Stage 1 — labeling
- ⬜ build_candidates (months, days, numbers10, numbers100, color_wheel, seasons, directions, moon_phases, scalars)
- ⬜ judge_prompts (bespoke per concept/class + scalars)
- ⬜ label (judge N=5, aggregate, filter, push)

## Stage 2 — probes
- ⬜ probe (attention probe per concept/class × layer; AUROC / Spearman+R²; push weights+metrics)

## Stage 3 — attribution
- ⬜ attribute (per-token probe activations on disjoint shards; mean + reliable-only means)

## Stage 4 — geometry (priority order; Tier-1 is headline)
- ⬜ tier1_z12 (months/colors/moon principal angles + planarity + phase alignment; Z/4 seasons vs directions)
- ⬜ tier2_nesting (seasons⊂months Fourier; base10⊂base100)
- ⬜ tier3_magnitude (shared axis PCA; cross-domain transfer; linear-vs-log)
- ⬜ tier4_worldmap (Procrustes to lat/long; compass frame alignment)
- ⬜ tier5_antipodal (indoor=-outdoor? love vs harm angle; 1D check)
- ⬜ tier6_causal_sae (DAS/patching + Gemma Scope cross-check — budget permitting)

## Stage 5 — infra speedup + writeup
- ⬜ GPU workflow speedup (bake template / persist uv cache); measure cold-start delta
- ⬜ final report.md + geometry.md pushed to HF; pods torn down; summary to user
