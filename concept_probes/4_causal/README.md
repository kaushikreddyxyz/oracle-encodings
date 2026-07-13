# 4_causal (was `stage6_1`)

## Purpose

Causal validation of the 2_probes/3_validation directions on gemma-2-2b base:
steering and ablation experiments testing whether learned directions are
causally responsible for a concept, not just correlated readers. Reports never
gate — 3_validation's deploy/caveat/reject status is untouched by causal
results; probe cards carry both axes.

## Pipeline

Working directory: `4_causal/code/` (repo-cloned on pod). Pod-driven via
`pod_run.sh` with env vars `FAMILIES`, `SCRIPTS`, `UPLOAD=1`.

```
1. python plumbing_gate.py --device cuda [--family months --cls january --layer 12 --alpha 1.0]
       # sanity gate: steer at alpha=1 must shift probe score by 1.0+-2%, hook
       # removal must exactly restore baseline. Must PASS before any fleet spend.
2. python e0_geometry.py [--probes-root ../../2_probes/probes --probe-cards ...]
       # local, CPU, ~2s — direction geometry only (cos(ridge,DoM), cos(ridge,LDA),
       # cross-layer decay). No GPU/pod needed.
3. python e1_attrib.py --families <f> --classes <c> [--n-pairs 100 --device cuda]
       # attribution patching + cross-layer Jacobian (candidate salient-layer localization)
4. python e2_cloze.py --families <f> --classes <c> --arms ridge,dom,rand [--factors ...]
       # forced-choice dose-response + specificity
5. python e2_ppl.py --families <f> --classes <c> --arms ridge,dom,rand
       # ActAdd perplexity-ratio (free-text likelihood)
6. python e4_ablate.py --families <f> --classes <c> --arms ridge,dom,rand,other [--space std|grad]
       # everywhere-ablation necessity + causal-rank sweep (family top-k subspace erasure)
7. python e5_propagation.py --families <f> --arms ridge [--frozen] [--patch]
       # single-layer ablation, copy matrix, frozen-attention control (salient/write-layer + half-life)
8. python e3_generate.py --families <f> --arms ridge,dom,rand [--factors ... --max-new-tokens 128 --n-prefixes 10]
       # steered generation (GPU; needs E2 selection or defaults)
9. python e3_judge.py --out out [--cap-usd 20]       # --smoke first (no API cost)
       # mercury-2 judges generations, LOCAL (needs OPENROUTER_API_KEY, no GPU)
10. python analyze.py --roots <pod out dirs...> --cards <probe_cards.json> [--all-concepts]
       # aggregates everything into out/analysis/causal_cards.json + causal_rollup.md
```

Orchestration: `pod_setup.sh` (installs deps, pulls data), `pod_run.sh`
(env-var driven multi-script runner with resume/upload).

## Inputs & Outputs

- Inputs: 2_probes ridge/DoM/LDA rows (`2_probes/probes/<family>/probes_l{L}.npz`),
  3_validation natscores (`3_validation/data/natscores/<family>.natscores.npz`) for
  dose calibration and natural positives, 1_dataset-generated matched pairs for
  E1/E5, prompt banks in `4_causal/prompts/<family>.{cloze,ordinal,tokens}.json`.
- Local outputs (gitignored): `out/{e0,e1,e2_cloze,e2_ppl,e3,e4,e5}/*`,
  `out/analysis/causal_cards.json` (per-concept card: sufficiency slopes+CIs,
  anti-steerable fraction, necessity per arm, specificity, salient/write layers,
  copy-matrix summary, family causal rank, ridge+DoM verdicts),
  `out/analysis/figures/` (per-concept 4-panel pages + `causal_rollup.md/.html`).
- HF mirror: `hf.co/kaushikreddyxyz/concept-probes-gemma2-2b` under path
  `stage6_1/out/*` (old-stage-name path kept intentionally), pushed after every
  script when `UPLOAD=1`.
- **`out/analysis/causal_cards.json` is the handoff artifact consumed downstream
  by `../../attribution/select_probes.py`** (gold-54 selection reads both this
  and 3_validation's `probe_cards.json`).

## Design decisions that bind

- **Steer with DoM, read with ridge.** Ridge (whitened mean-diff, cos(ridge,LDA)
  = 0.920) wins detection/reading; DoM wins necessity (ablation) and judged free
  generation by a wide margin.
- Every reported effect must clear a matched-norm random-direction control
  (slope~0, anti-steerable~50%, ablation damage~0) — controls held in all cases.
- Ablation target is the natural mean, never zero: `z' = z - (w.z - t)*w`.
- Salient causal layer (E5, corrected for last-layer artifact) is bimodal at
  L8/L12 and **disagrees with the correlationally-chosen deployment layer for
  61/64 concepts** — deployment-layer choice is not causal localization.
- Direction rotation across layers is not evidence of recomputation
  (copy-vs-cosine correlation only r~0.28) — later layers mostly copy
  (identity-path share ~0.97 at l+1, decaying to ~0.72 at L25).
- `glorptitude` (nonsense control) has no natural dose calibration -> excluded
  from the steering fleet by design; only E0/E1/E5 screening ran on it.

## Results

(REPORT_6_1.md's results were absorbed into this README 2026-07-13 — full text
in git history; its verdict counts were cross-checked against
`out/analysis/causal_cards.json` — they match)

- **Ridge: 52 causal / 10 read-only / 2 artifact-suspect** (of 64). **DoM: 59 /
  4 / 1.**
- Median steering cloze dose-slope: ridge 0.201, DoM 0.192, matched-random 0.001.
- Median ablation delta diagnostic-token log-prob: ridge -0.12, **DoM -1.90**
  (15x more damage), random -0.01.
- E0 geometry: median cos(ridge,DoM) = 0.291 (real concepts); cos(ridge,LDA) =
  0.920 — ridge ~ whitened DoM.
- E3 judged steered generation at factor 8: DoM mean overall 0.297 (22/64
  concepts >0.5), ridge 0.059, random 0.034, baseline 0.027; only the 3
  intensity scalars (physical_size 0.55, harmfulness 0.56, lovingness 0.56) are
  ridge generation-steerers.
- Natural-text reading comparison (3_validation natscores): ridge detection
  AUROC median 0.975 (44/64 concept wins) vs DoM 0.949 (20 wins); token-level
  Spearman near-tie leaning DoM (49 wins vs 15).
- Family causal rank (erasing top-k DoM subspace): weekdays/moon_phases k50=1,
  color_wheel k50=1 but k90=8 (hardest/highest-rank family, matches its
  1_dataset status).
- Cost: ~$35-40 GPU (5 H100 pods) + $8.09 judge API ~ **$45-50 total**.

## Gotchas

- Pods: **H100 SXM SECURE, template `runpod-torch-v240`**, created/torn down
  via the runpod-spinup skill — `pod_setup.sh`/`pod_run.sh` never create/delete
  pods themselves.
- Secrets only over stdin (HF token, GitHub token); `OPENROUTER_API_KEY` stays
  local (`.env`) and never goes to a pod — E3 judging runs locally, not on-pod.
- Pod-side state dies on Stop->Start — don't stop-and-keep pods overnight;
  everything needed is on HF + GitHub.
- One-time step-0 pre-uploads needed before pod setup (`random_pool.jsonl`,
  `judged_nat.jsonl` per family) since the HF repos don't have them by default.
- 4-pod fleet split by class count (16/16/16/16); `glorptitude` excluded from
  the steering fleet.
- `runpodctl pod get` is the source of truth for SSH port — creation-time port
  can be stale.
- Budget guard: check `runpodctl me` before/after each wave; GPU target
  $60-90.
