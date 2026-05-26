# What Do LLM Representations Contain Beyond Sparse Linear Features?
## Core Question
Sparse autoencoders (SAEs) decompose LLM activations into sparse sums of linear feature directions. This decomposition is incomplete — a residual $r(x) = x - \text{SAE}(x)$ persists at scale and approaches some fixed asymptote. We treat the SAE as a **filter** that absorbs everything consistent with the sparse linear representation hypothesis, and study what passes through that filter. **The question is not about SAEs, it is about LLM representations: How do non-sparse features manifest in LLM representations?**
## Motivation
The linear representation hypothesis (LRH) claims that LLM activations decompose into sparse sums of linear feature directions. SAEs operationalize this assumption for one-dimensional features. But SAE reconstruction error has a nonzero floor that does not shrink with dictionary size (Gao et al. 2024, Engels et al. 2025). This floor might indicate that some component of LLM activations is **not sparse-linear** — it is structured in a way that the sparse linear decomposition can't capture at any scale. In addition, Sun et al., 2025 showed that dense SAE latents correlate with principal components of the LLM residual stream (more than sparsely-activating latents). Furthermore, SAEs trained on residual stream vectors that have been projected onto the orthogonal complement of these dense latents no longer learn dense latents of their own. This fact indicates that dense latents are learning some subspace of the residual stream rather than arising from some abnormality in training. 
## Central Hypothesis
There are specific, identifiable, systematic types of information that are present in $r(x)$ and absent from (or degraded in) $\text{SAE}(x)$. These components reflect ways that LLMs encode information that are fundamentally incompatible with sparse linear decomposition.
## Key Definitions
- $x$: activation vector at a given layer for a given token
- $\text{SAE}(x)$: sparse linear reconstruction of $x$ (top-$k$ features from a trained SAE)
- $r(x) = x - \text{SAE}(x)$: the SAE residual — everything not captured by the sparse linear decomposition
- **Dense component**: any structure in $x$ that requires dense (non-sparse) activation patterns to represent, i.e., cannot be captured by activating a small number of dictionary directions
## Background and Related Work
### SAE Error and Dark Matter
**Gao et al. (2024)** — "Scaling and evaluating sparse autoencoders." Observed that SAE reconstruction MSE follows a power law with a nonzero asymptote as dictionary size scales. This constant error floor is the original "dark matter" observation.
**Engels, Smith, Tegmark (2025)** — "Decomposing the dark matter of sparse autoencoders." (TMLR) Decomposed SAE error into a linearly-predictable component ($\text{LinearError} = b^* \cdot x$, mostly unlearned sparse features) and a non-linearly-predictable component ($\text{NonlinearError}$, persistent across scale). Proposed a theoretical model (Appendix B) where the activation decomposes as $x = \sum w_i y_i + \text{Dense}(x)$, leading to $\text{SaeError} = \text{Dense}(x) + \text{Introduced}(x) + \text{unlearned features}$ (Eq. 20). Validated this decomposition synthetically (Appendix C.3, Table 1) but never measured $\text{Dense}(x)$ vs. $\text{Introduced}(x)$ in real models. Key findings:
- $\sim$50% of error vector and $>$90% of error norm are linearly predictable from input activations
- $\text{FVU}_{\text{nonlinear}}$ is roughly constant at fixed sparsity as SAE width scales (Fig. 1, 2)
- $\text{NonlinearError}$ has lower norm predictability, fewer interpretable features, but proportional CE loss contribution (Fig. 6, 7, 8)
- Gradient pursuit (better encoder) reduces total FVU by 3-5% but leaves $\text{FVU}_{\text{nonlinear}}$ unchanged (Section 6.1)
- Prior-layer SAE outputs explain up to 50% of nonlinear error variance (Section 6.2)
### Dense Structure Inside SAEs
**Stolfo, Wu, Sachan (2025)** — "Antipodal pairing and mechanistic signals in dense SAE latents." (ICLR SLLM Workshop) Established that densely-activating SAE latents ($>$10% firing rate) are concentrated in the top-PC subspace of the residual stream ($\rho_k$ metric), align with the $W_U$ quasi-nullspace, and form antipodal pairs (cosine sim $\approx -1$) that encode signed scalar quantities using two nonneg latents. Found that one such pair tracks the last singular vector of $W_U$ and correlates with output entropy.

**Sun, Stolfo, Engels, Wu, Rajamanoharan, Sachan, Tegmark (2025)** — "Dense SAE latents are features, not bugs." (NeurIPS 2025) Extended the above with causal ablations: removing the dense-latent subspace from activations and retraining the SAE eliminates dense latents; removing an equal sparse-latent subspace does not. Confirms dense latents track intrinsic directions in the residual stream. Proposed a taxonomy of six dense latent types: position-tracking, context-binding, $W_U$-nullspace/entropy, alphabet, meaningful-word, and PCA latents. Notably, the taxonomy explains $<$ 50% of all dense latents.

### Linear Representation Hypothesis and Its Limits

**Park, Choe, Veitch (2023)** — "The linear representation hypothesis and the geometry of large language models." Formalizes the LRH.

**Gorton (2025)** — "The origins of representation manifolds in large language models." Characterized representation manifolds — structured multi-dimensional feature geometries in LLM activations.

**Mendel (2024)** — "SAE feature geometry is outside the superposition hypothesis." Argues the multi-dimensional structure of SAE latents contradicts the standard superposition model.

### Other Relevant Work

**Gurnee (2024)** — "SAE reconstruction errors are (empirically) pathological." SAE errors have a larger effect on CE loss than random perturbations of equal norm, suggesting the errors are structured, not random.

**Bussmann et al. (2024)** — "Stitching SAEs of different sizes." Larger SAEs learn new types of dictionary vectors: features absent in smaller SAEs and finer-grained splits of existing features.

## Relationship to Prior Work

Sun et al. (2025) study dense structure that SAEs **do** capture (dense latents in the dictionary). We study dense structure that SAEs **don't** capture (the residual $r(x)$). These are complementary:

- Dense SAE latents represent directions the SAE learned but that fire frequently. The SAE allocates dictionary capacity to them.
- $r(x)$ represents structure the SAE failed to capture at all — either because it ran out of capacity, because the structure isn't representable as 1D directions, or because the sparsity constraint prevents activating enough features simultaneously.

A key open question is whether $r(x)$ contains the same types of information as dense SAE latents (position, context-binding, entropy) or qualitatively different types. If the former, the SAE partially captures dense structure and $r(x)$ is the overflow. If the latter, there are aspects of LLM representations that the SAE framework fundamentally cannot access.


### The key mechanistic question
What property of certain information makes it dense? Possible answers (non-exhaustive, to be investigated):
- **Multi-dimensional feature geometry**: the information lives on a manifold (circle, sphere, etc.) that no finite set of 1D directions can capture without error
- **Distributed coding**: the information is encoded across many dimensions jointly, with no privileged sparse basis
- **Context-dependent binding**: the same subspace carries different information in different contexts (Sun et al.'s "context-binding" latents hint at this), and the binding operation itself is inherently dense
- **Interference/superposition artifacts**: dense patterns arise from the interaction of many sparse features, producing structure in the sum that isn't in any individual term
- **Normalization/scaling signals**: information encoded in the overall geometry of the activation vector ($\|x\|$, angular position relative to subspaces) rather than in individual directions

Understanding *which* of these mechanisms is at play — and whether different types of dense information use different mechanisms — is the core contribution.

## Experimental Plan

**Setup** (shared across phases): Gemma 2 9B, Gemma Scope SAEs (largest available width per layer, up to 1M), $\sim$100k tokens from the Pile (following Engels et al.'s dataset: 300 contexts of 1024 tokens, filtered to positions > 200).

### Phase 1: Establish that $r(x)$ contains structured, identifiable information

**Goal**: Produce results that demonstrate $r(x)$ is not noise — it contains specific, nameable types of information that $\text{SAE}(x)$ loses. This is the "hey, this is interesting" result.

**Experiment 1.1: Comparative probing.** For a battery of known properties $P$ (token position, POS tag, named entity type, next-token entropy, syntactic depth, document topic), train linear probes on three inputs independently:
- $x$ (full activation — ceiling)
- $\text{SAE}(x)$ (sparse reconstruction)
- $r(x)$ (residual)

Report probe accuracy for each. The key finding we're looking for: properties where probe accuracy on $r(x)$ is substantially above chance while probe accuracy on $\text{SAE}(x)$ is substantially *below* probe accuracy on $x$. This means the SAE filter systematically loses this information and it ends up in $r(x)$.

Run across layers 5, 12, 20, 25 of Gemma 2 9B to see whether the "lost" information types vary by layer.

**Experiment 1.2: Unsupervised structure in $r(x)$.** Compute the covariance matrix of $r(x)$ across tokens. Measure:
- Effective dimensionality via participation ratio: $d_{\text{eff}} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$ where $\lambda_i$ are eigenvalues
- Subspace stability: compute top-$k$ PCA directions on two random data splits, measure principal angles between the subspaces. If the subspace is stable, $r(x)$ has consistent geometric structure across tokens.
- For each of the top 50 PC directions of $r(x)$, compute correlations with: token position, $\|x\|$, next-token entropy, $W_U$ nullspace composition ($\alpha_k$ from Sun et al.). Flag any direction with $|\rho| > 0.3$ on any of these for further investigation.

**Experiment 1.3: $r(x)$ vs. dense SAE latent subspace.** Sun et al. identified dense latents in Gemma Scope SAEs. Compute the subspace spanned by their decoder directions. Project $r(x)$ onto this subspace vs. its orthogonal complement. This directly measures: is $r(x)$ "more of the same dense structure the SAE partially captured" or "something the SAE dictionary doesn't point toward at all"?

**Milestone**: at least one specific information type that is degraded in $\text{SAE}(x)$ but present in $r(x)$, plus a geometric characterization (dimensionality, stability) of $r(x)$ that shows it is structured.

### Phase 2: Build confidence and pick apart the Phase 1 findings

**Goal**: For each information type identified in Phase 1, run ablations and fine-grained experiments to determine (a) how robust the finding is, and (b) whether the information is *necessarily* dense or just accidentally missed by this SAE.

**Experiment 2.1: SAE width scaling.** Repeat the probing from 1.1 using Gemma Scope SAEs at multiple widths (16k, 65k, 131k, 262k, 524k, 1M) on the same layer. For each property $P$:
- If probe accuracy on $r(x)$ decreases with SAE width $\to$ the SAE is gradually learning to capture $P$; it's sparse-compatible and just underlearned. Not interesting for our purposes.
- If probe accuracy on $r(x)$ stays constant or increases with SAE width $\to$ scaling the SAE does not help. $P$ is plausibly dense. This is the signal we want.

This is the key experiment that separates "dense by nature" from "sparse but unlearned." It directly mirrors Engels et al.'s finding that $\text{FVU}_{\text{nonlinear}}$ is constant across scale, but now we're asking *which specific information types* contribute to that constant floor.

**Experiment 2.2: Cross-architecture consistency.** Repeat probing on $r(x)$ from a different SAE architecture (e.g., train JumpReLU SAEs using the eleuther_sae_modified codebase, or use SAEs at different sparsity levels $k$). If the same information types end up in $r(x)$ regardless of SAE architecture, they're intrinsic to the activation, not an artifact of TopK or a specific training run.

**Experiment 2.3: Causal importance.** For each information type found to be persistent in $r(x)$ across scale:
- Replace $x$ with $\text{SAE}(x)$ during the forward pass (patching out $r(x)$) and measure task-specific performance (not just aggregate CE loss, which Engels et al. already measured in Fig. 8). Use targeted benchmarks: e.g., if positional information is dense, test on tasks requiring position sensitivity. If semantic information is dense, test on tasks requiring semantic reasoning.
- Compare to a control: replace $x$ with $\text{SAE}(x) + r_{\text{shuffled}}(x)$ (add the residual from a different, randomly selected token). If performance is similar to patching with $\text{SAE}(x)$ alone, the residual's contribution is token-specific. If performance is worse, the residual carries generic structure (like norm information).

**Experiment 2.4: Geometric deep dive.** For the most promising information type (the one that is persistent across scale, consistent across architectures, and causally important):
- What is the dimensionality of the subspace it occupies in $r(x)$? (project $r(x)$ onto directions that predict property $P$, measure how many dimensions are needed)
- Is there manifold structure? (local dimensionality estimation, e.g., using nearest-neighbor methods on the $r(x)$ vectors conditioned on property $P$)
- How does this subspace relate to the SAE decoder directions? (is it approximately orthogonal, or does it overlap with many decoder directions — requiring dense combinations to represent?)

**Milestone**: A clear picture of which Phase 1 findings survive scrutiny. For at least one information type: it's persistent across SAE scale, consistent across architectures, causally important, and we know its geometric properties. We now have enough to decide what paper to write.

### Phase 3: Commit to a narrative

This is the fork. Based on Phase 2 results, choose one of the following directions and run the experiments it requires:

**If the persistent information has identifiable manifold/multi-dimensional structure:**
The paper is about how LLMs encode specific computations on geometric manifolds that are invisible to sparse linear methods. Deep dive into the geometry, connect to the multi-dimensional features literature (Engels et al. 2024, Gorton 2025), show that the manifold structure is *used* by the model (causal evidence), and discuss implications for interpretability methods beyond SAEs.

**If the persistent information is distributed/holographic (no clean low-dimensional structure):**
The paper is about a fundamentally different encoding strategy in LLMs — one that operates alongside sparse features but uses a qualitatively different representational format. Characterize the encoding, show it carries specific information, and argue that interpretability requires tools beyond dictionary learning.

**If different information types use different mechanisms:**
The paper maps out the "representational zoo" — LLMs use multiple encoding strategies simultaneously (sparse 1D features, dense subspaces, manifolds, etc.), and we provide the first empirical characterization of which strategies encode which types of information. This is Paper A from the trajectories below.

## Potential Paper Trajectories

*Paper A (broad)*: "Beyond sparse linear features: unsupervised discovery of dense representational structure in LLMs." Establishes that LLMs encode specific types of information in ways incompatible with sparse linear decomposition. Provides methods for discovering what those types are.

*Paper B (deep dive)*: "Property X is encoded densely in LLM representations." Zooms in on one specific type of information found in $r(x)$, characterizes its geometry, causal role, layer-wise trajectory, and explains why sparse methods can't capture it.

**If $r(x)$ is mostly unlearned sparse features:**

The finding is still useful (it tells the field that SAE scaling will eventually capture everything), but the paper is less interesting. I think this scenario is quite unlikely. Perhaps use this as a pivot to studying why specific sparse features are harder to learn.

## Open Questions

- What is the right methodology for Phase 2.4? How do you determine whether information is "necessarily dense" vs. "dense by accident"? Experiment 2.1 (scaling) is the main lever, but it's indirect — it shows the SAE *doesn't* learn it at scale, not that it *can't*.
- How do we study the geometry of information in $r(x)$ without assuming it aligns with high-variance directions? PCA is a natural starting point but may miss low-variance structured subspaces.
- What tools exist for characterizing manifold structure, distributed codes, or other non-sparse representational formats in high-dimensional neural network activations?
- Sun et al.'s dense latent taxonomy explains $<$ 50% of dense latents. Is the unexplained fraction related to what we find in $r(x)$?
