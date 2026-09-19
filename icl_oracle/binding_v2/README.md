# Multiclass and compositional latent binding

Status: main-model development, fresh-seed evaluation, and random-label controls are complete. See [REPORT.md](REPORT.md) for the findings, [results/main_summary.md](results/main_summary.md) for full main-task tables, and [results/main_summary.png](results/main_summary.png) for the comparison figure. This document records the design and reproduction commands.

This follow-up to [`../REPORT_bind.md`](../REPORT_bind.md) asks whether the injected channel supports more than binary lookup, and whether predictions on unseen combinations exceed explicit similarity-retrieval baselines. Model weights remain frozen. Grid-walk and Markov-graft experiments are outside this experiment.

## Implemented tasks

Every prompt samples an arbitrary permutation from eight classes to eight single-token label words. Item words are distinct within the prompt and independent of the hidden signal. No task instruction or explanation of the hidden rule is given: the model sees demonstrations followed by one query ending at `:`.

| Family | Input and rule | Query interpretation |
| --- | --- | --- |
| `codebook` | One of K orthogonal unit codes maps to an arbitrary label; K defaults to 8 and is configurable. | Repeated symbols with new item words; capacity/association calibration. |
| `selection` | Eight binary features; select three distinct feature coordinates, then map their three-bit pattern to a label. | Input combinations never shown in demonstrations. |
| `parity` | Eight binary features; three disjoint feature pairs produce three XOR bits, then an arbitrary label. | Input combinations never shown in demonstrations. |
| `random` | Balanced random labels assigned to all 256 eight-bit combinations. | Negative control for systematic generalization to unseen combinations. |

Feature tasks reserve 64 of the 256 combinations as a heldout pool. Demonstrations are drawn from the other 192 combinations; all eight output labels are guaranteed to appear. Sampling is without replacement until that pool is exhausted. A 256-demonstration run therefore includes repeated combinations and records the unique count. Queries are sampled uniformly without replacement from the heldout pool, before inference. `--include-seen` additionally evaluates a separate set drawn from demonstrated combinations; results must distinguish these subsets.

Feature codes are `(2*x - 1)/sqrt(8)`. A seed-specific orthonormal basis maps these codes into the residual stream, so every example's injected vector has the same total norm. Codebook vectors are likewise unit norm. A fixed basis is used for the demonstrations and queries of a task seed. Injection occurs after decoder block `layer-1`, at the item token, with amplitude `beta * norm_ref`. The reference is the median centered residual norm at demonstration item positions in the uninjected prompt.

The three model conditions are:

- `token`: show the eight bits as space-separated text, or a single marker word for a codebook symbol.
- `none`: no hidden-feature information appears in the text or activations.
- `inject`: supply hidden features only through the residual edit.

Each query is evaluated independently against the same demonstration prefix. The runner caches that prefix once, copies its KV cache for each query batch, and never reveals another query's answer. It projects only query readout states to vocabulary logits.

## Metrics and comparison models

Primary accuracy is the fraction whose most likely **label** is correct. Also save probability of the correct label after renormalization over labels, full-vocabulary probability, total label probability mass, and full-vocabulary top-1 accuracy. Eight-way nominal chance is 12.5%; compare the matched no-signal control as well because finite prompts need not have perfectly balanced labels.

`baseline.py` receives the exact clean feature vectors, demonstrations, and queries. It evaluates nearest cosine neighbor, temperature-weighted cosine voting, and ridge classifiers on linear or quadratic features. Temperature and ridge settings are selected on disjoint development seeds. `analyze.py` can regenerate the exact observed model queries and attach matched baseline scores; it fails if raw query IDs, labels, or seen status do not match.

Selection is **not** constructed to defeat similarity: retrieval can generalize well on this task. Parity is a harder test of the proposed explanation, but no behavioral result alone identifies a particular attention circuit. A successful textual baseline establishes whether this prompt format and model can learn the task; it is an empirical comparison rather than a mathematical ceiling.

## Fast-run protocol

The main GPU development runs used `google/gemma-2-2b` at hidden-state layer 8 and `Qwen/Qwen2.5-7B` at layer 4. These layer choices came from the earlier binary experiments, before this task's evaluation results. The two models ran independently in parallel, with:

- Families: codebook-8, selection, and parity.
- Development seeds: 0, 1, 2, 3.
- 128 demonstrations and 32 independent unseen queries per feature-task seed, plus 32 seen-combination queries using `--include-seen`. Codebook tasks have 32 seen-symbol queries.
- Text, no-signal, and injected conditions; injection doses 0.25, 0.5, 1, and 2.
- Select a dose separately per model and task using development accuracy, then freeze it before evaluating fresh seeds 100–107. Ties choose the smaller dose.

The selected doses were Gemma beta 2 for every task; Qwen beta 1 for codebook and selection, and beta 2 for parity. Selection manifests are saved as `results/{gemma,qwen}_heldout.selection.json`. Evaluation expanded codebook size to 4, 8, 16, and 32 while retaining the dose selected at K=8. Each model produced 6,144 main heldout records. Random-label controls then used the same evaluation seeds and fixed beta 2, inherited from parity without random-task dose tuning; these produced 1,536 records per model. Both GPU pods were terminated after collecting results.

The codebook-size sweep fixes the total demonstrations at 128, giving 32, 16, 8, and 4 examples per symbol respectively. Consequently its decline cannot be interpreted as a pure representational-capacity limit: number of labels, evidence per label, and code geometry change together, and no K-specific layer or dose retuning was performed.

The main-seed results must remain separate from development sweeps. Record selected settings and any protocol changes when reporting results. A failure on a task whose textual comparison also fails is ambiguous about the injected channel; report that limitation rather than treating it as evidence against binding.

Example development commands from the repository root (use separate machines/processes with sufficient memory):

```bash
python -m icl_oracle.binding_v2.runner \
  --model google/gemma-2-2b --device cuda --layer 8 \
  --task codebook,selection,parity --seeds 0,1,2,3 \
  --shots 128 --queries 32 --classes 8 \
  --modes token,none,inject --beta 0.25,0.5,1,2 --bs 8 --include-seen \
  --output icl_oracle/binding_v2/results/gemma_development.jsonl

python -m icl_oracle.binding_v2.runner \
  --model Qwen/Qwen2.5-7B --device cuda --layer 4 \
  --task codebook,selection,parity --seeds 0,1,2,3 \
  --shots 128 --queries 32 --classes 8 \
  --modes token,none,inject --beta 0.25,0.5,1,2 --bs 8 --include-seen \
  --output icl_oracle/binding_v2/results/qwen_development.jsonl
```

For evaluation, use one task per invocation when its selected dose differs, set `--seeds 100,101,102,103,104,105,106,107`, pass only the chosen `--beta`, and write a new evaluation file. Output is append-only: restarting an entire command against the same filename can create duplicates. The analyzer rejects duplicates by default; `--drop-identical` permits only exact repeated records.

For codebook evaluation, repeat with `--classes 4`, `8`, `16`, and `32`; all codebook queries are seen-symbol queries regardless of `--include-seen`. For the random control, use `--task random --classes 8 --beta 2 --include-seen` with the same eight evaluation seeds and three modes.

Clean-vector baseline sweep, with independent hyperparameter selection:

```bash
python -m icl_oracle.binding_v2.baseline \
  --families selection parity random --n-demo 64 128 256 \
  --n-query 32 --include-seen \
  --development-seeds 0 1 2 3 4 5 6 7 \
  --evaluation-seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 \
  --out icl_oracle/binding_v2/results/baselines.json

python -m icl_oracle.binding_v2.analyze \
  icl_oracle/binding_v2/results/gemma_development.jsonl \
  --baseline-summary icl_oracle/binding_v2/results/baselines_summary.json \
  --output icl_oracle/binding_v2/results/gemma_development_summary.json
```

The broader baseline sweep uses 8 development and 16 evaluation seeds; model-matched baseline comparisons use the model's actual evaluated seeds and queries, not the broader baseline aggregate.

## Local development pilots

`results/local_pilot.jsonl`, `results/local_codebook_dose.jsonl`, and `results/local_text128.jsonl` contain exploratory runs on the already-cached `Qwen/Qwen3-0.6B-Base`. They use only seeds 0 and 1, 16 queries, and different shot/layer/dose settings. Configuration fields in each row are authoritative. These runs establish executable inference and help choose a dose sweep; they are not fresh-seed confirmation.

The 0.6B model is substantially smaller than the models used in the earlier binding results. Its outcomes cannot establish what Gemma-2-2B or Qwen2.5-7B can do. Keep these pilots separate from the main GPU report, including when their textual feature-task performance is weak.

## Source validation and statistical limitations

Run `python -m unittest icl_oracle.binding_v2.test_binding_v2 -v`. Four tests passed during implementation. They cover task determinism, label coverage, unit norms, seen/unseen separation, and exact selection/parity rule outputs. A tiny randomly initialized Qwen3 model checks cached injected and uninjected query predictions against independent full-prefix forwards, across multiple batches, to an absolute/relative tolerance of `1e-6`; probability mass and predicted labels are also checked. This test needs no downloaded checkpoint or tokenizer.

Aggregate each task seed first, then give seeds equal weight. Standard errors in the analyzer are across seed means, not across individually correlated queries. A one-seed group has no reported standard error. For comparisons, paired differences across the same task seeds are more informative than treating conditions as independent.

The main GPUs used bfloat16 weights. Exact tied logits can resolve differently for vocabulary-order top-1 versus label-list-order top-1; full-vocabulary accuracy can therefore occasionally be slightly higher than within-label accuracy. These are distinct recorded metrics. At large codebook sizes, label probability mass falls substantially, so restricted accuracy alone does not describe unrestricted generation quality.

Changing demonstration count changes sampling and query draws; changing query count changes item and label-word assignments. Shot sweeps therefore do not constitute strictly nested acquisition trajectories. Fixed settings regenerate identical tasks, and the matched-baseline analyzer checks the actual query identities.

Injected directions, rule, label permutation, split, and sampled words vary together with task seed; the present pilot does not independently estimate each source of variation. Full-space random directions are tested; activation-PC directions and alternative injection locations are not included in this first pass. The token condition also has a longer context and a different representation from the injection condition.

## Main-model results

The results are reported in [REPORT.md](REPORT.md). The main frozen-model runs do not outperform matched clean-vector cosine voting on unseen feature combinations. Textual parity also fails, limiting conclusions about the injection channel specifically. Raw outputs preserve restricted and full-vocabulary metrics, seen/unseen flags, and task metadata for independent analysis; no query subset was selected after observing predictions.
