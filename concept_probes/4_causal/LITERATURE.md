# Stage 6.1 literature survey — causal evaluation of concept directions

Compiled 2026-07-02 by three research subagents (steering evals / erasure-ablation
evals / layer attribution). These notes ground the Stage 6.1 spec in
`knowledge/concept_probes/task.md`. Verification status is flagged inline by the
surveyors; AxBench details were verified against the paper's LaTeX source, most
other figures against arXiv HTML — re-check any single number before quoting it
in a publication.

---

# Survey 1: Evaluating steering vectors / activation addition

Scope: rigorous, replicable eval protocols for causal testing of concept directions, oriented to our setting — 64 concepts × 12 layers of ridge-probe rows (unit-normed, on standardized activations) on **gemma-2-2b BASE** resid_post, token-level salience concepts (months, colors, compass, intensity scalars), not behavioral traits.

## 1. AxBench (Wu et al. 2025, arXiv:2501.17148, ICML 2025) — highest priority; verified from LaTeX source

**Protocol summary.** Takes natural-language concept descriptions (sampled from Neuronpedia's GemmaScope SAE feature labels — so mostly token-level concepts, like ours), synthesizes labelled data with `gpt-4o-mini-2024-07-18`, and evaluates every method on two axes: concept **detection** (AUROC on held-out labelled data with hard negatives) and **steering** (LLM-judged open-ended generation). Concept500 = 500 concepts per site; sites = resid layers 10/20 of **Gemma-2-2B-it** and 20/31 of **Gemma-2-9B-it** (note: *instruction-tuned*, though 2B SAEs are the base-model GemmaScope SAEs). 144 train examples/concept.

**Steering eval, exact mechanics:**
- Intervention: in-place activation addition at **one residual layer, all token positions**: `h ← h + α·w`, with `w` **unit-normed** for all rank-1 methods (DiffMean, Probe, PCA, LAT, ReFT-r1), via pyvene.
- Magnitude: `α = steering_factor × max_activation`, where max_activation = max projection `h·w` over that concept's detection eval set (for SAEs, Neuronpedia's max activation). **14 factors swept: {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0}.**
- Generation: 10 Alpaca-Eval instructions per concept, ≤128 new tokens, temperature 1.0, **no repetition penalty** (they explicitly argue penalties are an unfair, non-standard crutch common in prior steering work).
- Judge: `gpt-4o-mini-2024-07-18`, three independent rubrics each scored **0/1/2**:
  1. **Concept score** — concept absent (0) / minimally-awkwardly incorporated (1) / fully & naturally incorporated (2). Judged "without regard for grammar or logical coherence."
  2. **Instruct score** — response unrelated (0) / minimally related (1) / clearly related (2) to the instruction, topic-relatedness only.
  3. **Fluency score** — highly unnatural/repetitive (0) / noticeable errors (1) / fluent (2).
  Each prompt asks for an explanation then "Rating: [[score]]".
- **Overall = harmonic mean of the three subscores** (range 0–2; a zero on any axis zeroes the overall).
- **No-leakage factor selection**: the 10 instructions are split 5/5; the best factor per concept×method is chosen on one half by overall score and *reported on the held-out half*. Secondary metric: **winrate vs SAE** (ties = 0.5 win).

**Key numbers (mean overall steering score, avg over 4 sites):** Prompt 0.894 > LoReFT 0.741 > SFT 0.676 > LoRA 0.615 > ReFT-r1 0.543 >> **DiffMean 0.239** > SAE 0.165 > SAE-A 0.157 > LAT 0.127 > PCA 0.105 > **Probe 0.098** > SSV 0.026. Detection AUROC (avg): DiffMean 0.942 ≈ Probe 0.940 ≈ ReFT-r1 0.938 (no significant pairwise difference) >> SAE 0.695.

**Dose-response finding:** raising the factor monotonically *lowers* instruct score everywhere; concept score **rises-then-falls at early layers** but rises ~monotonically at late layers. ReFT-r1 traces the Pareto frontier of concept-vs-instruct.

**Applicability to us:** closest existing protocol — same model family, rank-1 resid-stream addition, token-level open-vocabulary concepts, per-concept factor calibration via max projection. Two critical caveats: (a) their **linear probe (BCE, unit-normed) was among the best detectors and the worst steerers** — a direct warning that our high-AUROC ridge rows may not steer; DiffMean must be our comparison arm; (b) they used -it models with Alpaca-Eval instructions — the instruct score is not meaningful for our base model.

## 2. ActAdd (Turner et al., arXiv:2308.10248) and CAA (Panickssery/Rimsky et al., arXiv:2312.06681)

### ActAdd
**Protocol.** Steering vector = activation difference of a *single* prompt pair (e.g. "Love"−"Hate") at layer *l*, scaled by coefficient *c*, added during the forward pass. Models: **GPT-2-XL (a base model)** for topic/perplexity experiments; OPT-6.7B and LLaMA-3-8B for detox/sentiment.
**Metrics:**
- **Perplexity ratio**: mean per-token perplexity of steered vs unsteered model on OpenWebText documents *split by topic relevance*. Success = perplexity ↓ on concept-relevant text (~0.875 ratio for weddings) while ≈1.0 (0.994) on unrelated text. Fully automatic, generation-free, base-model-native.
- Topic-shift success rate on free-form continuations of generic prefixes, K=3 completions.
- Detox: Perspective API on RealToxicityPrompts; sentiment: SiEBERT on IMDb continuations (66.9% neg→pos on LLaMA-3).
- Fluency: conditional perplexity under an external LM; relevance: embedding cosine.
- Hyperparameters by grid search, c ∈ [3, 20], layer ∈ [6, 24]; middle layers most effective (layer 6 peak ≈90% wedding success on GPT-2-XL).
Caveat: some qualitative demos acknowledged cherry-picked; quantitative perplexity/detox/sentiment results are the replicable part.
**Applicability:** the **topical perplexity-ratio eval is the single most base-model-appropriate metric in the literature** — no instructions, no judge; directly measures "does adding w make concept-relevant text more likely without damaging unrelated text." Bucket by judge/lexicon labels, never by the probe itself (circularity).

### CAA
**Protocol.** Steering vector = **mean difference of resid activations at the answer-letter token** over hundreds of A/B contrast-pair prompts (sycophancy, corrigibility, hallucination, …); added at **all positions after the user prompt** at one layer. Llama-2-7B/13B-Chat; best layer 13 (7B).
**Metrics:** (i) multiple-choice: mean probability on behavior-matching answer vs multiplier; (ii) open-ended: GPT-4 1–10 rubric on held-out questions; (iii) capabilities: MMLU ≈ unchanged under steering.
**Findings:** bidirectional probability deltas (0.03–0.3); effects concentrated in middle layers; steering stacks with system prompts/finetuning.
**Applicability:** the **forced-choice logprob delta** transfers directly to our categorical families (Δ log P of concept-consistent vs inconsistent continuations); the ±multiplier symmetry test is worth copying.

## 3. Tan et al. 2024, "Analysing the Generalisation and Reliability of Steering Vectors" (arXiv:2407.12404, NeurIPS 2024)

**Protocol.** CAA-style vectors from Model-Written Evals (36–40 behavioral datasets), Llama-2-7b-Chat (L13) and Qwen-1.5-14b-Chat (L21). **Propensity** per example: logit difference m_LD = logit(y+) − logit(y−). **Steerability** = slope of a linear fit of propensity over multipliers λ ∈ {−1.5…+1.5} — per-example and per-dataset. Released as `steering-bench`.

**Key findings.**
- **Steerability is highly variable across examples within a concept**: in many datasets ~40–50% of individual examples are *anti-steerable* (negative slope) even when the dataset mean looks fine.
- **Spurious steerability bias**: A/B position and Yes/No token artifacts explain ~20–60% of per-sample steerability variance despite balanced training data.
- **ID→OOD**: steerability correlates between prompt settings (ρ ≈ 0.89 Llama, 0.69 Qwen) but degrades OOD; vectors transfer best where the model already behaves as desired.

**Applicability:** (1) report **per-example dose-response slopes and the anti-steerable fraction**, not just mean deltas; (2) audit spurious drivers (token frequency, position, lexical co-occurrence); (3) evaluate on prompts distributionally distinct from probe-training text.

## 4. Representation Engineering (Zou et al. 2023, arXiv:2310.01405)

Reading vs control split: *RepReading* (LAT: stimulus templates, PCA on differences of contrastive pairs) vs *RepControl* (linear combination, piece-wise conditional, projection operators; contrast vectors applied layer-by-layer across many layers). Honesty control raised TruthfulQA MC1 31.0→55.0 (7B), 35.9→50.3 (13B), ~59 (70B; approximate). Coefficient selection not standardized — the gap AxBench criticizes. LAT reading vectors: AxBench detection AUROC 0.712, steering 0.127 — reading ≠ control operationalized. RepE's multi-layer injection contrasts with AxBench/CAA single-layer; with 12 probed layers we can test both.

## 5. Probe-derived vs difference-of-means vs SAE rows

- **AxBench (verified):** DiffMean steers ~2.4× better than BCE linear probes (0.239 vs 0.098) and beats SAE decoder rows (0.165); probes steer worse than SAEs despite being co-best detectors.
- **Im & Li 2025 (arXiv:2502.02716):** theory: under L(v)=E‖h₊−h₋−v‖², optimum is exactly v* = E[h₊−h₋] (mean difference). Diagnosis: classifier weights approximate the right *direction* (up to whitening) but wrong *scale*; PCA directions can be nearly orthogonal to the separation axis. Empirics: MeanDiff wins nearly everywhere (refusal APC 87.5 vs 79.2 classifier vs 64.5 PCA-diff).
- **Why probes mis-steer:** a max-margin direction discounts high-variance nuisance components that the generative distribution of concept-present activations actually contains; mean-difference includes them. Ridge on standardized activations ≈ (Σ+λI)⁻¹-scaled class-mean-difference — closer to whitened mean-diff than logistic weights; our w lives in diagonally-whitened coordinates, so raw-space steering must choose σ⊙w vs w⊘σ — run both.
- Secondary (not fully verified): Mayne et al. arXiv:2411.08790; Chalnev et al. arXiv:2411.02193.

## 6. Dose methodology synthesis

Consensus: (i) **normalize α per concept** by the natural activation scale along w (AxBench max-projection); (ii) **sweep, don't pick** — report the concept-vs-fluency Pareto; select single reported factor on a held-out split; (iii) mid layers steer best for topical content (ActAdd L6/48; AxBench L10>L20 on 2B for DiffMean; CAA L13/32); (iv) always pair target metric with a **degradation metric** — concept score alone is maximized by degenerate repetition. Anthropic (Durmus et al. 2024, "Evaluating Feature Steering"): factor −20…20 sweep, MMLU+PubMedQA capability guard, sweet spot −5…+5, sizable **off-target effects** (a gender-bias feature raised age bias 13%) → our 64-probe battery doubles as the off-target instrument.

## 7. BASE-model pitfalls

- ActAdd is the main base-model precedent: generic prefixes + corpus perplexity ratios, no instructions.
- AxBench's instruct score assumes an -it model; replace with **prefix-topicality**; keep concept + fluency.
- At temp 1.0 with no penalties, small base models are disfluent at α=0 → **always report the α=0 judged baseline**; deltas, not absolutes. Logit-slope and perplexity-ratio metrics are immune and carry the headline.
- Pres et al. 2024 (arXiv:2410.17245): prefer **model likelihoods** (baseline-referenced Δ log-likelihood of behavior-consistent vs inconsistent continuations) over sampled text or MC; under their pipeline CAA underperforms prior reports.
- No paper quantifies base-vs-instruct Gemma-2 steering differences — our base-model adaptation is a contribution, not a replication.

## Ranked shortlist (survey 1)

1. **AxBench steering protocol, base-model-adapted** (judge-based; concept / prefix-topicality / fluency, harmonic mean, no-leakage factor split, α=0 row, DiffMean arm).
2. **ActAdd perplexity-ratio** (automatic, cheap, runs on the full grid).
3. **Forced-choice logprob deltas within concept families** + Tan-style per-example slopes and anti-steerable fraction; monotone-ordinal test for intensity scalars.
4. **Off-target / selectivity audit** via the probe battery + a base-model capability anchor.

---

# Survey 2: Concept erasure / directional ablation and measuring causal necessity

## 1. Amnesic Probing — Elazar et al. 2021 (TACL, arXiv:2006.00995)

**Protocol.** Train iterative linear SVMs to predict a property from BERT layer-k representations; project onto the nullspace of each classifier (INLP); repeat until probes hit majority-class accuracy; run modified representations through remaining layers; measure behavior change.
**Metrics.** ΔLM accuracy on masked-word prediction (coarse-POS: 94.12% → 61.92%); KL divergence of token distribution before/after.
**Controls.** **Rand**: remove the same number of random directions. **Selectivity**: concatenate an explicit 32-d encoding of the erased property back and fine-tune; recovery = the intervention removed mostly the target property.
**Key findings.** Probe accuracy uncorrelated with behavioral importance (Spearman ρ = 0.085, p = 0.871) — the canonical decodable-≠-used result.
**Caveats.** INLP removed 264 directions for coarse POS, 585 fine POS, 738 dependencies — huge rank reductions damage regardless of content. Erasure at early layers sometimes **recovers at later layers** (early observation of self-repair).
**Applicability.** Rand + selectivity controls and ΔKL/Δaccuracy dual metrics transfer directly. The rank caveat bites for our multiclass families: removing one ridge direction per class is not "erasing the concept."

## 2. LEACE — Belrose et al. 2023 (NeurIPS, arXiv:2306.03819)

Closed-form, least-squares-optimal affine edit r(x) = x − W⁺ P_{WΣ_XZ} W(x − E[X]) guaranteeing **linear guardedness** (no linear classifier beats constant). Oracle LEACE (per-example labels at edit time) = upper bound on surgical erasure. **Concept scrubbing**: LEACE at every layer of Pythia/LLaMA to erase POS; metric = increase in LM loss (bits/byte): LEACE +1.73–3.57; cruder SAL +3.16–4.69; **matched-rank random-subspace: negligible** → LMs causally rely on linearly-encoded POS. Mean-difference projection is exactly right for *binary* concepts; multiclass needs the whitened cross-covariance column space (optimal projection is *oblique*); LEACE rank ≈ #classes−1 in one shot (rank 17 for POS vs INLP's ~360).

## 3. Refusal direction ablation — Arditi et al. 2024 (arXiv:2406.11717)

Difference-in-means direction per (layer, post-instruction position); best candidate chosen on val by bypass/induce scores subject to **KL < 0.1 on harmless prompts** (do-no-harm constraint), excluding layers ≥ 0.8L. **Directional ablation:** x′ ← x − r̂r̂ᵀx at **every layer and every position** — the direction can never be re-written (forecloses self-repair). Addition at the single source layer induces refusal. Metrics: refusal substring score, Llama Guard 2 safety score, MMLU/ARC/GSM8K/… capability suite. Refusal → ~0% across 13 models with ≤~1–2.4% capability change. **Why gold-standard:** bidirectional causality, everywhere-ablation, explicit KL specificity constraint, independent judge, 13-model replication.

## 4. Diff-in-means is causal; reading ≠ steering

- **Marks & Tegmark (arXiv:2310.06824):** Normalized Indirect Effect NIE = (PD*− − PD−)/(PD+ − PD−); mass-mean NIE 0.95/1.41 vs LR probe 0.66/0.58 (LLaMA-13B L7–13). MCI hypothesis: probes latch onto correlated confounds.
- **ITI (Li et al. 2023, arXiv:2306.03341):** mass-mean shift beats probe-weight direction for steering (42.3% vs 34.8% true×informative).
- **Belrose (diff-in-means blog):** every admissible linear predictor β satisfies ⟨β, δ⟩ > 0; neutralizing span(δ) is **necessary and sufficient** to erase linearly available information about a *binary* concept.
- Unverified-in-detail: 2026 preprint (arXiv:2606.02907) reports centroid-difference steering ≈ matched-norm random in their setting (p = 0.286).

## 5. Metrics & controls table

| Metric | Used by | Notes |
|---|---|---|
| Δ task accuracy (masked LM) | Elazar 2021 | vanilla vs amnesic |
| KL(p_clean ‖ p_ablated) next-token | Elazar (effect); Arditi (specificity constraint <0.1) | dual use |
| Δ LM loss, bits/byte | Belrose 2023 | collateral + effect on natural text |
| Δlog-prob / NIE | Marks & Tegmark | normalized indirect effect |
| Judge/classifier score on generations | Arditi | behavior-level outcome |
| Capability benchmarks | Arditi | global-damage guard |
| Direct vs total effect | McGrath 2023 | see §7 |

Controls: matched-rank/matched-norm random directions; other-concept directions (a full 64×64 ablate-i-measure-j matrix would be a novel strong version); selectivity restore.

## 6. Rank of the concept subspace

INLP greedy over-removal (20–738 dims) with real collateral damage; **RLACE (arXiv:2201.12091)**: adversarial **rank-1 projection suffices** for binary gender (~50% classifier accuracy, 78.86% main-task preserved); LEACE rank ≈ k−1 one-shot. For binary concepts single-direction necessity is well-posed; for k-class families expect causal rank up to k−1 — a per-class direction sweep vs full-subspace LEACE measures the effective causal rank.

## 7. Propagation & self-repair

- **Hydra effect (McGrath et al., arXiv:2307.15771):** resample-ablate attention layers (15 alternative prompts — in-distribution; preferred over zero/mean ablation); downstream layers compensate — at L23/32 compensation explains 92% of variance in downstream changes, restoring ~70% of the ablated contribution; late MLPs act as suppressors/erasers. Direct effect (downstream frozen) vs total effect (ablation propagates).
- **Rushing & Nanda 2024 (arXiv:2402.15390):** self-repair holds on the full training distribution, partly LayerNorm-scale mediated, noisy/imperfect.
- Consequence: single-layer projection is a weak necessity test; "no behavior change after ablating at l" ≠ "layer l causally irrelevant."

## Ranked shortlist (survey 2)

1. **Arditi-style everywhere directional ablation** with three direction arms (ridge w, DoM δ, LEACE rank-1) + matched-norm random + other-concept controls; outcomes = Δ judge-scored concept metrics + Δlog-prob of concept tokens; KL guard on off-concept text.
2. **LEACE subspace scrubbing per family** + rank sweep → effective causal rank; matched-rank random-subspace control mandatory.
3. **Layer attribution with explicit self-repair control**: ablate at l only vs from-l-onward; direct vs total effects; erasure-propagation curves.
4. **Amnesic selectivity restore** adapted to inference time (add back α·δ after erasure; recovery ⇒ specific information was the cause).

Cross-cutting: base-model next-token log-prob of judged concept tokens is a cleaner primary effect size than judged generations; NIE-style normalization for cross-concept comparability.

---

# Survey 3: Localizing the causally salient layer; cross-layer persistence

## 1. Activation patching / causal tracing

Clean vs corrupted prompt; patch residual/MLP/attn at (layer, position) across runs. **Denoising** (clean→corrupt) tests sufficiency — ROME's causal tracing found the early site (mid-layer MLPs at subject token) and late site (late attention at last token); ROME needed **windows of ~10 adjacent MLP layers** — single-layer interventions under-report distributed/persistent features. **Noising** (corrupt→clean) tests necessity. They are **not complements** under redundancy (Heimersheim & Nanda, arXiv:2404.15255). Metric: normalized logit difference m = (LD_patched − LD_corrupt)/(LD_clean − LD_corrupt); avoid raw probability; use ≥2 metrics. Interchange interventions (Geiger et al./DAS) on the probe subspace only = cleanest match to our setup (cited from memory; verify details).

## 2. Attribution patching

First-order Taylor: Δmetric ≈ Σ (clean_x − corrupt_x) ⊙ ∂metric/∂x |_corrupt. **Two forwards + one backward** give attributions for every layer×position simultaneously (Nanda; Syed et al. EAP matches/beats ACDC). Failure modes: LayerNorm curvature, saturated attention; early-layer residual attributions least reliable. With metric = probe readout at l′, ∂(w_l′·h_l′)/∂h_l is a **cross-layer influence Jacobian** — all l < l′ in one backward pass.

## 3. Residual stream as shared memory

- **Tuned lens (Belrose et al., arXiv:2303.08112):** representations rotate/scale across depth; per-layer affine translators needed — "same feature ≠ fixed direction across layers."
- **Norm growth (Heimersheim & Turner):** residual norm grows ~exponentially (~×1.045/layer GPT2-XL) — raw cross-layer dot products not comparable; steering magnitudes must scale with local norm.
- **Crosscoders (Anthropic 2024):** per-layer SAEs re-learn duplicated persistent features; a crosscoder latent's per-layer decoder-norm profile is a persistence signature.
- **Multi-layer SAEs (Lawson et al., arXiv:2409.04185):** despite high adjacent-layer residual cosine, a latent tends to fire at one layer per prompt; persistence is less clean than the cartoon.
- **Feature flow graphs (Balcells et al., arXiv:2410.08869):** cross-layer feature matching via decoder cosine; features persist/merge/split; semantic and directional similarity can decouple.
- **Stages of inference (Lad et al., arXiv:2406.19384):** deleting/swapping adjacent middle layers retains 72–95% accuracy — mid layers partially interchangeable; expect causal-salience plateaus, not sharp peaks.

## 4. Cross-layer direction cosine — caveats

1. Raw cosine confounded by anisotropy (few dominant high-variance dims) — compute in whitened space too (Mahalanobis-style probe comparison, arXiv:2606.19603, skim-only).
2. Rotation ≠ recomputation; recomputation can land in the same direction. **Only intervention distinguishes copy from recompute.**
3. Norm growth: track projection/stream-norm SNR across layers.

## 5. Self-repair — the confound

McGrath: naive ablate-at-l-measure-output **underestimates** l's importance, worst at mid layers. Recipe to distinguish copy vs recompute: **exact path decomposition** of the layer-l′ residual into identity path vs summed outputs of blocks l+1…l′; **resample-ablate, never zero-ablate**; **freeze trick** (freeze downstream attention patterns to clean values — if the feature still recovers, recovery came through the identity path).

## 6. Steering-layer-choice findings

CAA: sharp peak at L15–17/32 (~50% depth), near-zero very early/late. AxBench: L10 & L20 on 2B **by convention, not sweep**. Anthropic Golden Gate: middle layer (exact layer undisclosed). "Predicting Where Steering Vectors Succeed" (arXiv:2604.15557, 2026, skim-only): per-concept optimal layer varies strongly; a logit-lens "Linear Accessibility Profile" predicts best steering layer (ρ≈0.6–0.9). Our 64-concept × 12-layer causal sweep fills a real gap.

## 7. Attention-mediated propagation across positions

Geva et al. (arXiv:2304.14767): attention knockout over **windows of layers** (single-layer knockouts leak); subject enrichment early-mid at subject token; transfer to final token via mid-upper attention. Steering mechanics (arXiv:2604.08524, skim-only): injected vectors propagate mainly as attention **values** through existing patterns, not by re-routing attention. Intervening at (t, l) and reading at (t′>t, l′>l) mixes residual copy and attention transport — separate by same-position vs later-position readout.

## 8. Circularity mitigations

Behavioral anchors (Δ logprob on concept-diagnostic tokens; judged generations); LEACE/proper erasure rather than naive project-out; held-out readout family (disjoint-data probes or DoM as meter); tuned-lens/unembedding readout as a probe-free intermediate meter.

## Recommended ≤3-experiment design (survey 3)

E1 attribution-patching salience map + cross-layer Jacobians (screening, hours). E2 verified read/write localization: denoise + LEACE-erase per layer, at concept-token vs all positions → write layer & read window (~300k short forwards, ~1 H100-day upper bound with batching). E3 copy-vs-recompute decomposition: identity-path vs downstream-output contributions + frozen-downstream control → 12×12 copy matrix C[l,l′] vs the cosine matrix.

---

# Flags / unverified items (union)

- 2026 papers arXiv:2604.15557, 2604.08524, 2606.19603, 2606.02907: abstract/skim only.
- DAS/interchange-intervention specifics: cited from memory.
- RepE 70B TruthfulQA figure approximate; Anthropic steering layer undisclosed.
- AxBench cached LaTeX at `~/.cache/oracle-encodings/knowledge/2501.17148/` (surveyor note).
