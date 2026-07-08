# Stage 7-Oracle — overnight spec: probe corpus scan → oracle encoder → nanochat injection

Written 2026-07-08 ~1:45 AM by the pre-compaction instance, amending the user's
proposal (amendments marked ⚠AMENDED with rationale). The post-compaction
instance implements this. User wakes ~11 AM. This spec is the contract.

## Hard operating constraints (read first, re-read after compaction)

- **Budget**: RunPod balance $773; target total spend ≤ $400, hard stop $550.
  OpenRouter auto-tops-up (judge not needed tonight — no judging in this task).
- **Token quota**: at spec time 13% of 5h / 36% of weekly Fable. **Delegate
  aggressively**: every one-off implementation, debugging, monitoring, or
  data-checking task goes to a subagent; use `model: "sonnet"` (or opus) for
  mechanical/code tasks, reserve own-model reasoning for design decisions and
  gate calls. Compact when possible. Keep tool output out of orchestrator
  context (background agents, output files).
- **Never** put HF/GitHub tokens in argv (stdin pipe pattern, see
  `stage6_1/code/pod_setup.sh` + FLEET.md). Tear down every pod when its phase
  is done; verify `runpodctl pod list` empty at the end.
- All long scripts: tqdm + heartbeat log (user requirement, standing).
- Gates below are GO/NO-GO: never train stage N+1 on stage N output that
  failed its gate. Record every gate result in `stage7_oracle/STATE.md`.

## Objective (user's, restated)

1. Score 2–3B held-out ClimbMix tokens with the best available concept probes
   (3 gemma-2-2b layers × K concepts).
2. **Exp A**: train Qwen3-0.6B encoder (+MLP head) to predict per-token probe
   scores on ~1B tokens; test generalization on ~0.5B. MLP-only first; unfreeze
   encoder only if needed.
3. **Exp B (dual)**: ablate gemma's residual along the probe directions; train
   encoder+MLP to emit a d=2304 vector that, added to the ablated residual,
   minimizes divergence from the clean residual.
4. If A (and B) pass gates: use the encoder as the **oracle encoding** and
   pretrain a nanochat d24 with the signal injected (low-rank, circular
   manifolds for cyclic families, small noise), vs a matched no-value-embeddings
   baseline. Keep tokenizers constant where possible.

## Phase 0 — Probe selection (CPU, $0, do first)

Data: `stage6/data/natscores/<fam>.natscores.npz` (preds_{ridge,dom,lda,logistic}
[12, T, C], y, token2ex, ex_nat_split), `stage6/artifacts/probe_cards.json`,
`stage6_1/out/analysis/causal_cards.json`. All local.

1. For every (concept, layer, arm ∈ {ridge, dom, lda, logistic}) compute on the
   natural TEST half: example-level detection AUROC (max-pool, label ymax≥0.34)
   and token Spearman. (Stage-6.1 established: ridge best AUROC 0.975 vs dom
   0.949; dom slightly better token-ρ. Expect ridge to win most reading cells —
   the user's "usually DoM outperforms ridge" holds for *causal* tasks, and DoM
   is used for the Exp-B ablation directions regardless.)
2. Pick **3 layers** maximizing mean best-arm AUROC across concepts, tie-broken
   toward the causally salient band L8–L12 (6.1 result) and spread (don't pick
   3 adjacent). Expected outcome ≈ {8, 12, 16} or similar mid-band.
3. Per concept per layer pick the best arm. **Selection threshold** *(tunable)*:
   example AUROC ≥ 0.90 AND token ρ_rel indication not pathological. Same
   concept set across all 3 layers (a concept must clear the bar at ALL 3).
4. Per user caveat: if a strong family has one weak class at one layer,
   optionally retrain that (class, layer) — multi-seed ridge refit needs that
   family's activation re-extraction on a pod (stage5 pipeline, ~20–30 min);
   do this ONLY if it rescues ≥2 concepts total, else drop the class.
5. If <20 concepts survive the all-3-layers constraint, relax to layer-optimal
   sets (user-approved fallback). Expect ~40–55 survivors (44 deploy + caveat
   pool); intensity scalars will likely fail the reading bar — dropping them is
   fine and expected.
6. Output: `stage7_oracle/out/probe_set.json` — {layer: {concept: {arm, w(unit,
   std-space), b, t_nat, s95}}} + the DoM direction per concept for Exp B +
   selection table (audit). Gate **G0**: ≥20 concepts, table reviewed.

## Phase 1 — Corpus scoring (GPU, the big cost — engineer before spending)

- **Shards**: ClimbMix shards **≥ 320** (0–273 consumed by old overnight runs;
  310–316 used by Stage 5/6 natstats + natural pool — see memory
  `overnight-probes-handoff`). Verify emptiness of overlap by shard id.
- **⚠AMENDED — throughput first**: Stage-5 extraction ran ~4–7k tok/s (eager,
  small batches) → 2.5B tokens would be ~$450. Unacceptable. Before scoring:
  (a) benchmark gemma-2-2b forward-only bf16 at seq 2048 / large batch with
  `attn_implementation="sdpa"` vs `"eager"` on one H100; (b) **parity check**:
  probe scores under sdpa vs eager on ~100k tokens — accept sdpa if per-probe
  score deltas ≪ probe noise (|Δ| p99 < 0.05·std). Softcapping differences are
  the concern; measure, don't assume. Target ≥30k tok/s; if <15k tok/s after
  tuning, **shrink corpus** to 1.0B train + 0.3B val (budget gate, user's
  volumes are targets not laws).
- **Scoring output** (this is what makes Exps A+B cheap):
  per token store `token_id` (int32) + probe scores s [3 layers × K] quantized
  **int8** with per-(layer,concept) scale/zero (calibrated on first 10M
  tokens), memmap shards + index. ~(4 + 3K) bytes/token → 2B tokens ×
  K≈45 ≈ 280GB. Keep on pod NVMe (100–200GB/pod × fleet); DO NOT try to push
  through HF; consolidate to the encoder-training pod via direct scp/rsync
  pod-to-pod (or score on the same big pod that later trains).
- **⚠AMENDED — dataset norms** (user's medium-confidence item, adopted):
  recompute per-probe score mean/std on the scoring corpus itself (streaming);
  use these (not natscores stats) to standardize Exp-A targets and later the
  injection coords. Store in probe_set.json. Rationale: removes shard-drift
  scale bias; cheap.
- Fleet: 2–4 H100 pods, shard-split, `runpod-torch-v240`, reuse
  stage6_1 pod_setup.sh (needs only gemma + probe_set.json + shard download).
  ClimbMix download: HF `nvidia/ClimbMix` (or the source used in stage6
  `mine_natural.py` — READ that file for the exact repo/format).
- Gate **G1** (before encoder training): score distributions per probe vs
  natural-pool stats (quantile match); spot-check 5 concepts' top-100 firing
  tokens (lexical concepts must fire on their surface forms/associates);
  january-vs-march correlation sanity (within-family correlated but < 0.9).

## Phase 2 — Exp A: probe-prediction encoder

- **Architecture (one stack for A and B, interpretable bottleneck):**
  Qwen3-0.6B (`Qwen/Qwen3-0.6B-Base`, causal, hidden 1024) → linear **up**
  (1024 → n_targets = 3·K, no bias constraint; this activation IS the probe
  prediction, one neuron per (layer, concept)) → [Exp B only: linear **down**
  (3·K → 2304)]. ⚠AMENDED from "4× transformer MLP": the user's own
  requirement "hidden dim = number of probes, each neuron ↔ one probe" defines
  the bottleneck; a 4×-MLP variant (1024→4096→n_targets) is the fallback
  ablation ONLY if the linear head underfits (train it on 100M tokens to
  compare before deciding).
- **Loss**: MSE on corpus-standardized scores, all tokens. Report per-probe R²
  (train + heldout 0.5B). Baselines: predict-mean (R²=0), bag-of-last-token
  embedding linear probe (sonnet-agent can add), shuffled-text control.
- **Schedule**: MLP-only (frozen Qwen) first — if heldout median R² ≥ 0.8,
  STOP (user rule). Else full fine-tune (0.6B fits easily; lr ~1e-4 cosine,
  bsz to fill H100). 1B tokens ≈ 3–6 H100-h fine-tuned; MLP-only much faster
  (can cache Qwen hiddens? No — 1024 fp16 × 1B = 2TB; recompute).
- **Tokenizer bridge Gemma→Qwen — ⚠AMENDED, and answers the user's question:**
  The user's proposal (split unknown gemma tokens into qwen sub-tokens,
  mean-pool, never use qwen tokens absent from gemma) is *half* right: the
  intersection restriction hurts (qwen would see unnatural fragments
  everywhere, degrading its encoding) and sub-token splitting can't handle the
  reverse case where a natural qwen token *merges across* a gemma boundary.
  **Primary scheme — char-offset prefix-state alignment**: tokenize the raw
  text with EACH tokenizer independently, keep char offset maps; the encoder
  representation for gemma token t = Qwen's hidden state at the **last qwen
  token whose char span ends ≤ end-char(t)** (a causal prefix state at that
  char boundary — no future leakage, no vocab games). Where no qwen token ends
  exactly at t's boundary (crossing), that fallback state lags by ≤ a few
  chars; **measure the crossing rate first** (expected low; both BPEs split on
  spaces). If crossing rate > ~10% of positions or A underperforms, fall back
  to **boundary-constrained encoding** (= user's proposal, cleaned up):
  re-tokenize each gemma token's string with qwen in-context and mean-pool its
  sub-token states. Build this as ONE module (`align.py`) with both modes —
  the SAME module later bridges Qwen→nanochat.
- Gate **G2**: heldout median per-probe R² ≥ 0.6 = GO for nanochat
  (0.4–0.6 = marginal, proceed only with user-visible caveat in STATE.md;
  < 0.4 = NO-GO for phase 4, report). Additional REAL-data check: run the
  encoder over the Stage-6 natural eval texts and compute example-AUROC of its
  predictions against judge labels — encoder should retain ≥90% of the gemma
  probes' AUROC. Also check R² on a 50M-token fineweb-edu sample scored by
  gemma (the nanochat corpus is fineweb-family, not ClimbMix — cheap
  distribution-shift check before phase 4).

## Phase 3 — Exp B: ablation-repair dual

- **⚠AMENDED — B has a closed-form target; no extra teacher passes and no
  extra samples needed for the MSE version.** Ablating the K DoM directions
  {d_c = σ⊙w_c^dom} at layer l to natural means t is an affine map of the
  clean residual; the repair vector is exactly
  `v* = D · G⁻¹ · (s − t)` where D = [d_c] (2304×K), G = DᵀD (Gram), s = the
  clean DoM probe scores at layer l — all recoverable from the Phase-1 stored
  scores (store DoM scores for the primary layer too if arm choice differed).
  So Exp B trains from the SAME memmaps: loss = ‖(up/down)(qwen(x))_t − v*_t‖².
  Two variants: (i) **fixed decoder** D·G⁻¹ (train up-proj only — this is
  literally Exp A in different units); (ii) **learnable decoder** — check the
  learned down-proj columns recover the DoM directions (cosine per concept —
  a free interpretability result). A live-teacher KL-repair variant (ablate,
  add v, continue forward, minimize KL vs clean) is STRETCH ONLY if ahead of
  schedule — it's the behaviorally meaningful version but needs online gemma.
- Ablation layer: ONE layer — the causal-salient consensus (L8 or L12 per
  causal_cards `e5_salient_layer_corrected` histogram). Directions: **DoM**
  (6.1: necessity −1.90 nats vs ridge −0.12 — user's "probes need to be
  capable of concept ablation" is satisfied only by the DoM arm).
- Gate **G3**: heldout R² on v* ≥ 0.5 and (variant ii) median direction-recovery
  cosine ≥ 0.7. B failing does NOT block phase 4 (A is the injection signal);
  it's the dual/validation experiment.

## Phase 4 — nanochat d24 with oracle-encoding injection (conditional on G2)

- **Baseline discipline**: baselines exist on HF
  (`kaushikreddyxyz/modular-addition-checkpoints`? NO — nanochat baselines:
  A/B d24 pair, CORE 0.2777/0.2711; see memory `oracle-encodings-baseline`,
  `runs/oracle_runs/baseline.sh` for exact flags incl. seed/save-optimizer).
  **Value embeddings**: memory `nanochat-tricks-vs-oracle` says VEs are the
  serious confound — the injected model must train with VEs DISABLED
  (freeze/zero wte+value_embeds path per that memory). CHECK whether the HF
  d24 baselines already have VEs disabled; if not, train a matching **no-VE
  baseline concurrently** on a second node (same recipe, same seed policy,
  same token budget) — the comparison is injected-no-VE vs plain-no-VE, both
  else-identical to baseline.sh.
- **Injection signal**: encoder (frozen, best A checkpoint) run over the
  nanochat pretraining corpus (fineweb-edu shards per baseline.sh). Per
  nanochat token (char-offset alignment module, qwen-side prefix states):
  probe predictions ŝ ∈ R^{3K} → **structured low-rank embedding**:
  - cyclic families → 2-D circle coords per family:
    (Σ_c ŝ_c cos θ_c, Σ_c ŝ_c sin θ_c), θ_c = class phase (month k → 2πk/12…);
  - presence/categorical non-cyclic (continents, location_type) → per-class
    coords as-is (or family-PCA-2D if >4 classes);
  - any surviving intensity scalars → 1-D each.
  Total rank r ≈ 15–30. Coords standardized (corpus stats), **plus gaussian
  noise σ = 0.15·RMS** *(tunable)* — the user's anti-trivial-manifold measure.
  Embed into d_model=1536 via a FIXED random orthonormal 1536×r matrix, scale
  to **β = 0.05 of residual RMS at the injection site** *(tunable; measure
  RMS on a baseline checkpoint forward)*; add to the residual stream at ONE
  mid-band site (after block 8 of 24 — mirror the gemma finding that mid-band
  is where concepts causally live).
- **Storage vs inline**: precompute injection coords (r fp16/int8 per token)
  for the full token budget → r=24 int8 ≈ 24B/token; 12–15B tokens ≈ 300–360GB
  → fits node NVMe. Precompute on 1–2 H100s (qwen 0.6B ~100k tok/s ≈ 35–40
  GPU-h — START THIS EARLY, it's the long pole) OR inline qwen per-rank during
  training (adds ~10% step time, zero storage, more engineering risk). Choose
  by clock: precompute must begin by ~4 AM to matter; otherwise inline.
  ⚠ If the full token budget is unaffordable in time/money, it is acceptable
  to inject for the FIRST N billion tokens and anneal β→0 after (documented),
  or to shorten both runs' token budget equally — comparability > scale.
- **Tokenizers**: nanochat keeps its own tokenizer (retraining nanochat's
  tokenizer to gemma's 256k is not sensible for d24); constancy holds between
  the two nanochat runs, which is the comparison that matters. The bridge
  handles cross-tokenizer alignment. (This answers the user's "keep tokenizers
  constant… reuse?" — reuse nanochat's existing tokenizer artifacts from the
  baseline runs; do NOT switch nanochat to a different vocab overnight.)
- Compute: 8×H100 node(s) (~$26/hr). Injected run ≈ baseline recipe duration
  (baseline pair took ~? — read baseline.sh / HF card; assume 5–8h). If a new
  no-VE baseline is needed, run both nodes concurrently. Budget both ≈ $300
  worst case — check remaining balance before launch; if tight, d20 fallback
  with a freshly trained d20 no-VE baseline pair is NOT cheaper — prefer
  shortening token budget.
- Gate **G4** (during): loss curve sane vs baseline within first 2k steps
  (divergence > 5% bpb = kill, inspect β). Final: CORE + val bpb both runs;
  post-hoc probe: linear-decode the injected concepts from the injected
  model's residuals vs baseline (did the model *use* the oracle subspace?).

## Timeline (1:45 AM start, user awake 11 AM)

- 2:00 Phase 0 (subagent, CPU) + throughput/parity benchmark pod up.
- 2:30–3:30 scoring code + align.py + encoder code (3 parallel sonnet agents,
  smoke-tested like stage6_1: tiny-model fixtures).
- 3:30–8:00 Phase 1 fleet scoring (2–4 pods). In parallel: fineweb qwen
  precompute decision + nanochat prep agent (reads baseline.sh, VE status,
  prepares run configs).
- 7:00–10:00 Exp A (starts on partial data if scoring is streaming-ordered;
  else 8:00). Exp B right behind on same pod.
- ~10:00–11:00 G2 call; if GO and coords ready → launch nanochat node(s);
  they finish mid-afternoon. Write STATE.md summary for the user at 11.
- Everything that can slip: B's learnable-decoder variant, KL-stretch,
  per-class retraining, extra corpus.

## Risk register (pre-answered decisions)

1. sdpa parity fails → eager + shrink corpus (gate above), do NOT eat $450.
2. Qwen3-0.6B gated/unavailable → Qwen2.5-0.5B; record swap.
3. ClimbMix ≥320 shards don't exist / format drift → fall back to largest
   unused shard ids; check `stage6/code/mine_natural.py` for loader.
4. Encoder R² marginal only for intensity/implicit concepts → fine; report
   per-family; lexical concepts carry the injection.
5. nanochat run can't finish before user wakes → expected; leave running with
   monitor + STATE.md instructions; do NOT babysit with own context (cron or
   long-interval monitor).
6. Compaction mid-night → this SPEC + STATE.md + memory pointer are the
   recovery path; keep STATE.md current after every gate.
