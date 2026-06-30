# Concept Probes, Attribution & Representation Geometry — Run Report

Autonomous overnight run. Appended per stage. Source spec: `knowledge/overnight_brief.md`.

**Run start (UTC):** 2026-06-27T01:58:58Z

---

## 0. Setup, model & data choices (and why)

### Models
- **Probe target = `google/gemma-2-9b`** (designed). Chosen by the brief because **Gemma
  Scope** pretrained SAEs exist for Gemma-2-9B, which Step-4 Tier-6 needs; 9B forward
  passes fit a single 24GB GPU. Fallback of last resort `Qwen/Qwen2.5-7B` (non-gated) —
  but that forfeits the SAE cross-check, so it is used only if the license stays unaccepted.
- **Judge = `google/gemma-3-27b-it`** (designed). `gemma-4-31b-it` does **not** resolve to a
  clean repo on HF (307 redirect, no released Gemma 4) → not used. Fallback judge
  `Qwen/Qwen2.5-32B-Instruct-AWQ` (non-gated, AWQ for vLLM throughput), which the brief's
  "good-enough" clause explicitly permits if Gemma stays blocked.
- **SAEs = `google/gemma-scope-9b-pt-res`** — verified accessible (not gated).

### ⚠️ Gating blocker (recorded at run start)
Both Gemma repos are `gated=manual` and the account `kaushikreddyxyz` has **not accepted**
the license (verified by a tiny config.json download test → `GatedRepoError`). Accepting is
a manual web click only the user can do. Decision: do all gemma-independent work
autonomously, fall back to a non-gated judge for labeling if needed, and **block probing**
(Steps 2-4) until access appears (re-checked on a timer). Labels are model-agnostic, so
Step-1 output is valuable regardless of which probe target is ultimately used.

### Data / shard provenance
- Corpus = `karpathy/climbmix-400b-shuffle` (CC-BY-NC, research-only; verified accessible).
  This is the corpus the nanochat baselines actually trained on (not fineweb-edu, despite
  `dataset.py`'s default `BASE_URL`).
- **Shards nanochat training consumed:** downloaded pool shards 0-273; sequentially read
  ~0-183 (<1 epoch; from the dataloader `pq_idx` at the end of the run); validation = shard
  6542. Authoritative record: `meta_008352.json` on `kaushikreddyxyz/oracle_baseline_d24_fp8`.
- **Shards used here (disjoint):** `shard_00300`-`shard_00309` — clear of the entire
  downloaded pool (0-273) and the val shard, so probe data cannot overlap training data.

### Infra
RunPod balance $415.22 at start; $80/hr cap; no pods running. HF auth via CLI-cached login.
`.env` is moved to pods by scp only (local `.env` reads are blocked by a permission hook).

---

## Assumptions / decisions log
- Moon phases: the brief says "12" but there is no canonical 12-phase set → using the **8**
  standard phases (new, waxing crescent, first quarter, waxing gibbous, full, waning gibbous,
  last quarter, waning crescent); illumination fraction recorded per phase for Tier-3 bridge.
- Directions: using the **4 cardinal** directions for the Z/4 geometry (intercardinals
  optional); single-letter/substring forms need directional-sense confirmation by the judge.
- Numbers serve double duty: presence classes {0..10} + base-100 buckets, AND the magnitude
  scalar (digit value as external label) for Tier-3.

*(stage sections appended below as work completes)*

## Decision log update (t+1.1h) — judge substitution
**Judge changed gemma-3-27b-it → `Qwen/Qwen2.5-32B-Instruct-AWQ`.** Reason: gemma-3-27b-it
fails to load under vLLM 0.11 (`rope_scaling should have a 'rope_type' key` — a known
vLLM/transformers version incompatibility for Gemma 3's config; also multimodal/heavier).
Chasing exact version pins would burn wall-clock for a component the brief explicitly
allows substituting. Qwen2.5-32B-Instruct-AWQ is non-gated, loads cleanly under vLLM, and
is a strong instruction-follower for 0-5 scoring. **The scientifically important model is
unchanged: the probe target remains gemma-2-9b** (loaded via transformers, not vLLM, so
unaffected by this). Labels are model-agnostic; labeling can be re-run with gemma-3-27b-it
later (pin transformers to vLLM 0.11's expected version) if a gemma judge is desired.
Earlier infra fix logged: vLLM pinned to 0.11.0 because the newest (0.23) pulled torch+CUDA-13
which the A100 host driver (CUDA 12.8) rejects.

## Step-3 attribution requirement (user, t+1.8h)
Per token position, record BOTH (a) the probe activation at that token (attention probes:
per-token alignment score query·h_t AND the attention weight a_t; plus the pooled seq score),
AND (b) the residual-stream activation norm ‖h_t‖ at that token+layer — so probe activation
is interpretable relative to raw activation magnitude (high score vs merely high-norm token).
Also (per brief): every probe's per-token activation, mean attention-probe activation across
concepts, and mean over reliable probes (metric>0.9).

## Stage 1 — labeling COMPLETE (judge Qwen2.5-32B-Instruct-AWQ, ~28,342 calls, N=5)
All 28,336 candidates (shards 300-309, disjoint) labeled + pushed to hf.co/datasets/kaushikreddyxyz/concept-probes-overnight.
Per-concept kept / pos / neg / discarded / judge-vs-pseudo-gold agreement:
- color_wheel 1242 (621/621) disc 2915, agree 0.81
- days        472  (236/236) disc 2328, agree 0.80
- directions  520  (260/260) disc 1080, agree 0.54  (hardest concept: substring/sense traps, per brief)
- months      212* (106/106) disc 4588, agree 0.90
- moon_phases 160  (88/72)   disc 419,  agree 0.60  (intermediate phases rare in web text)
- numbers10   2660 (1330/1330) disc 1740, agree 0.60
- numbers100  2934 (1467/1467) disc 1066, agree 0.51
- scalars     4331 (regression; 69 high-variance discarded)
- seasons     590  (295/295) disc 1010, agree 0.74
*NOTE: the "kept" counts reflect aggressive 50/50 balance-capping. The full positive
pool is far larger (e.g. months has 4204 label==1 rows). For Step 2/4 the probe loader
was changed to KEEP balance-capped positives (capped at MAX_POS_PER_CLASS=250/class) so
geometry clouds are rich; probe training caps negatives at 4x positives to stay balanced.
Quirk: presence concepts whose surface form is almost always the concept (months, days,
seasons) have scarce true negatives -> low pseudo-gold "agreement" is partly the judge
correctly rejecting wrong-sense/ambiguous surface hits, not error.

## Stage 2 — probes (RUNNING): gemma-2-9b, attention probes, LAYER_STRIDE=3 (14-layer depth sweep)
Reusing the same A100 (vLLM judge killed to free the GPU). AUROC (presence) / Spearman+R2 (scalar).

## Step-4 note: Tier-4 (world map) skipped
Continents (europe/america/africa) were labeled as SCALAR concepts (how-strongly-European
ratings), not geo-located city/country clouds with lat/long, so the Procrustes-to-map test
has no coordinate ground truth to align against. run_geometry.py passes places=None (tier4
skipped) rather than fabricate coordinates. Tiers 1-3 + 5 run; Tier-1 (Z/12 collision) is
the headline. Re-enable Tier-4 later by adding a place->lat/long table.

## Stage 2 — probe performance debugging (resolved)
Probe training was initially ~50s/probe (would be ~150h for 79 probes x sweep). Root causes
found by isolated timing (no model load) on the A100 pod:
1. **GPU-resident training** — original `_collate` rebuilt a padded tensor + per-element
   CPU->GPU copies every batch every epoch. Rewrote to stack each probe's train/val set
   into ONE GPU tensor once, train by slicing on-GPU. (training itself: 0.93s for 60 epochs.)
2. **The real killer — CPU tensor padding on a 252-core host.** Building the padded batch
   on CPU (manual loop OR torch pad_sequence) took ~69s/probe: PyTorch thrashes the tiny
   per-example copies across 252 threads. Moving each cached (T,d) tensor to GPU FIRST then
   padding on GPU = 0.15s/probe (measured a/b=69s vs c=0.15s). Also set_num_threads(8).
Net: ~1s/probe (50x). Probe stage is now extraction-bound (gemma-2-9b fwd, ~350s/layer over
~12k examples; the per-layer re-forward is the documented memory-safe tradeoff).
Infra notes: nvidia-smi reports HOST-namespace PIDs; `kill` inside the container needs
CONTAINER PIDs (from ps) — earlier GPU-process kills silently failed, leaving duplicate
probes contending (the apparent "leaked VRAM" was this, not a true leak). A pod restart
cleanly reset GPU state once. Config: LAYER_STRIDE=7 (6 layers spanning depth),
MAX_POS_PER_CLASS=120, MAX_PER_SCALAR=300 — chosen for predictable completion in budget.

## Stage 2 — probes COMPLETE & pushed
474 probes (79 concepts x 6 layers [1,8,15,22,29,36]), 0 skipped, ~0.4s/probe after the fix.
Weights + per-probe metrics.json + index -> hf.co/kaushikreddyxyz/concept-probes-weights (44MB).
Highlights (best layer per probe): presence months/days/color AUROC up to 0.97-1.0;
moon_phases 0.61-1.0 (intermediate phases hardest). Scalars (Spearman / R2):
africa 0.77/0.53, outdoors 0.72/0.49, europe 0.69/0.46, america 0.65/0.51, harmfulness 0.64/0.52,
duration 0.62/0.37, costliness 0.47, physical_size 0.39, lovingness 0.47/0.48.
Caveat: numbers-magnitude scalar fit poorly at L36 (best-layer selection mitigates).

## Stage 3 — attribution COMPLETE & pushed
attribute.py (after fixing a device-placement bug: inputs were left on CPU while the model
was on cuda) over 40 sampled snippets from the disjoint shards, at each probe's best layer.
Records per (snippet, token, probe): s_t (query.h_t alignment), a_t (attention weight),
pooled_score, AND ||h_t|| (residual-stream norm at that token+layer) — both bits per the
user requirement. Plus token_aggregates (mean a/s across concepts, mean over reliable
probes, norms-by-layer). Pushed -> hf.co/datasets/kaushikreddyxyz/concept-probes-overnight/attribution
(attribution.jsonl ~208MB, token_aggregates.jsonl, summary.json).

## Stage 4 — geometry COMPLETE & pushed (headline layer L25/42, 4-layer sweep [1,13,25,37])
Pushed -> hf.co/datasets/kaushikreddyxyz/concept-probes-overnight (tier{1,2,3,5}.json + per-layer
+ figures/ + geometry.md). N_BOOT=200 bootstrap 95% CIs. (tier5 cloud-capped to 200 pts;
thread-capped to avoid a 252-core numpy thrash that hung the first attempt.)

### Tier 1 — Z/12 collision study (HEADLINE result)
**Cyclic concepts occupy DISTINCT residual-stream subspaces — they do NOT collide onto a
shared Z/12 plane.** Principal angles between cycle planes are near-orthogonal:
months/color_wheel theta=(88.3,89.3)deg; months/moon_phases (85.8,89.5)deg;
color_wheel/moon_phases (81.4,88.4)deg; Z/4 seasons/directions theta2=86.7deg.
Each cycle is individually planar+ordered (months cyclic order_score=1.0, seasons
uniformity 0.88) but the model uses a SEPARATE plane per concept, not one reused "clock".
(moon_phases is Z/8, thinner cloud -> wider CIs.)

### Tier 2 — Harmonic nesting (positive)
Seasons are the FUNDAMENTAL Fourier mode of the months cycle (1st-harmonic energy 0.46;
coarse-grain cosine consistent). Base-10 buckets ~ mean of their unit members (cosine 0.991):
multiscale magnitude coding (numbers10 nested in numbers100).

### Tier 3 — Abstract magnitude axis (partial)
A weak shared magnitude axis (0.35 of variance across numbers/costliness/physical_size/duration,
mean pairwise cosine 0.12). Cross-domain transfer from numbers is weak->moderate
(duration Spearman 0.26). Magnitude is partly, not fully, a shared code.

### Tier 5 — Antipodal / opponent structure
indoors vs outdoors: 63deg (partially aligned, shared place axis). lovingness vs harmfulness:
96deg = ORTHOGONAL (independent features, not a single good-bad axis). Tier 4 (world map)
skipped: continents were scalar-rated, not geo-located clouds.

## RUN COMPLETE — deliverables on HuggingFace
- Probe weights (474 = 79 concepts x 6 layers): hf.co/kaushikreddyxyz/concept-probes-weights
- Labels + candidates + attribution + geometry + this report:
  hf.co/datasets/kaushikreddyxyz/concept-probes-overnight
All four steps executed end-to-end on gemma-2-9b (probe target) with a Qwen2.5-32B-AWQ judge
(gemma-3 substitution, brief-blessed). See decision/debugging logs above.
