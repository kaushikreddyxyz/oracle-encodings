# Stage 6.1 fleet runbook (orchestrator commands)

Pods: **H100 SXM SECURE, template `runpod-torch-v240`**, created/deleted via the
runpod-spinup skill (`create-pod.sh` / `cleanup-pod.sh`) — pod lifecycle is the
orchestrator's job, none of these scripts create or delete pods.
`<pod>` below = the ssh alias/command the skill prints (prefer direct SSH).

Secrets discipline: HF + GitHub tokens travel **only over stdin pipes** (never
argv), OPENROUTER_API_KEY stays in the **local** repo-root `.env` and never
goes to a pod (E3 judging runs locally).

---

## Step 0 — one-time local pre-uploads (data pods need but HF lacks)

Verified 2026-07-02: the HF model repo has `families/*` + `natscores/*` but NOT
`random_pool.jsonl`; the dataset repo has `judged/judged.jsonl` but NOT
`judged_nat.jsonl`. `natstats26.npz` and `probe_cards.json` are git-tracked and
arrive with the clone. So, from the repo root, once:

```bash
hf upload kaushikreddyxyz/concept-probes-gemma2-2b \
   concept_probes/stage6/data/natural/random_pool.jsonl \
   stage6_1/inputs/random_pool.jsonl --commit-message "stage6_1 E3 prefix pool"

for FAM in color_wheel continents costliness directions duration harmfulness \
           location_type lovingness months moon_phases physical_size seasons weekdays; do
  hf upload kaushikreddyxyz/concept-probes-stage4-data \
     concept_probes/stage4/data/$FAM/judged/judged_nat.jsonl \
     data/$FAM/judged/judged_nat.jsonl --repo-type dataset \
     --commit-message "stage6_1: natural judged eval for pods"
done
```

(glorptitude has no judged_nat — nonsense control, never scored on natural text.)

**Also: commit + push the stage6_1 code and prompts/ to GitHub before any pod
setup** — pods get code via `git clone`, not scp.

---

## Step 1 — pilot pod (plumbing gate + 3-concept pilot)

```bash
# 1. create pod (runpod-spinup skill), then:
scp concept_probes/stage6_1/code/pod_setup.sh <pod>:/workspace/

# 2. setup: HF token line 1, GitHub token line 2, both over stdin
{ cat ~/.cache/huggingface/token; gh auth token; } | \
  ssh <pod> 'bash /workspace/pod_setup.sh'

# 3. PLUMBING GATE (task.md §6.1.8): steer january@L12 alpha=1 on the real
#    model; PASS iff mean probe-score shift within 2% of 1 (p95 within 5%)
#    and hook removal restores the baseline bit-identically. Exit code 0/1.
ssh <pod> 'cd /workspace/oracle-encodings/concept_probes/stage6_1/code && \
           python plumbing_gate.py --device cuda'

# 4. 3-concept pilot (january / harmfulness / europe) through E1-E2-E4-mini:
ssh <pod> 'cd /workspace/oracle-encodings/concept_probes/stage6_1/code && \
  nohup env FAMILIES="months,harmfulness,continents" \
    SCRIPTS="e1,e2_cloze,e2_ppl,e4" \
    ARGS_e1="--classes january,harmfulness,europe" \
    ARGS_e2_cloze="--classes january,harmfulness,europe" \
    ARGS_e2_ppl="--classes january,harmfulness,europe" \
    ARGS_e4="--classes january,harmfulness,europe" \
    UPLOAD=1 bash pod_run.sh > /workspace/pilot_run.out 2>&1 & echo started'

# monitor (heartbeats + per-script logs + exit codes):
ssh <pod> 'tail -5 /workspace/oracle-encodings/concept_probes/stage6_1/out/progress_*.log; \
           cat /workspace/oracle-encodings/concept_probes/stage6_1/out/logs/status.tsv 2>/dev/null'
```

**Decision gate before spending on the fleet:** ≥1 pilot concept shows a clean
monotone dose-response beating its random control, AND the plumbing gate
passed. If yes → keep the pilot pod as fleet pod A (it is already set up).

---

## Step 2 — 4-pod fleet for E1/E2/E4/E5 (balanced by class count, 16/16/16/16)

| Pod | FAMILIES | classes |
|-----|----------|---------|
| A | `months,seasons` | 12+4 = 16 |
| B | `color_wheel,location_type,costliness,physical_size` | 12+2+1+1 = 16 |
| C | `weekdays,moon_phases,duration` | 7+8+1 = 16 |
| D | `directions,continents,lovingness,harmfulness` | 8+6+1+1 = 16 |

glorptitude is EXCLUDED from the steering fleet (no natscores → no dose
calibration; `common.dose_calib` raises for it by design). If E4/E5 want it as
an extra control, run it separately with explicit doses — open issue.

Per pod (after `pod_setup.sh` as in step 1; setup can pass
`FAMILIES="months seasons"` to download only that pod's data, or default all):

```bash
ssh <podX> 'cd /workspace/oracle-encodings/concept_probes/stage6_1/code && \
  nohup env FAMILIES="<row above>" SCRIPTS="e1,e2_cloze,e2_ppl,e4,e5" \
    UPLOAD=1 bash pod_run.sh > /workspace/pod_run.out 2>&1 & echo started'
```

`UPLOAD=1` pushes `out/` (results + logs + status.tsv) to
`hf.co/kaushikreddyxyz/concept-probes-gemma2-2b` under `stage6_1/out` after
every script, so partial results survive pod loss.

**Expected wall-clock per pod** (16 classes; budget §6.1.8 says 20–30 H100-h
TOTAL, i.e. ~5–7 h/pod):
- e1_attrib: ~100 pairs x 16 concepts x 2 metrics, 2 fwd + 1 bwd each →
  ~10k fwd-equivalents, short texts — **[PLACEHOLDER ~0.5–1 h]**
- e2_cloze: 16 concepts x 12 layers x 12 doses x 3 arms x ~30–50 short cloze
  prompts (biggest grid) — **[PLACEHOLDER ~1–3 h]**
- e2_ppl: 3 layers x 6 doses x 3 arms x 2 buckets of natural text —
  **[PLACEHOLDER ~0.5–1 h]**
- e4_ablate: everywhere-ablation x arms + causal-rank sweep —
  **[PLACEHOLDER ~1–2 h]**
- e5_propagation: 12 ablation layers x 12 readout layers + copy-matrix
  decomposition + frozen-attn control — **[PLACEHOLDER ~1–2 h]**
Calibrate all of these from the pilot pod's status.tsv timestamps before
launching pods B–D; abort/trim grids if the pilot extrapolates past budget.

---

## Step 3 — E3 steered generation (1 pod, after E2 selection exists)

E3 reads `out/e2_cloze/selection.json` for the selected dose (falls back to
factor 2.0 + 1.0 if absent). **e2_cloze does NOT write selection.json itself**
(it writes dose-response curves + summary.jsonl): the orchestrator derives it
in the analysis pass — schema E3 accepts (tolerant reader in e3_generate.py):
`{family: {class: {"factor": f, "layer": l}}}` or `{concept: {...}}`, with
`factor`/`selected_factor`/`best_factor` and optional `layer` keys. Running
E3 with the 2.0/1.0 defaults is acceptable if E2 analysis is not done yet.
Pull the fleet's merged out/ from HF first:

```bash
ssh <podE3> 'cd /workspace/oracle-encodings/concept_probes/stage6_1 && \
  hf download kaushikreddyxyz/concept-probes-gemma2-2b \
    --include "stage6_1/out/e2_cloze/*" --local-dir /workspace/hf_staging --quiet && \
  mkdir -p out/e2_cloze && \
  cp -a /workspace/hf_staging/stage6_1/out/e2_cloze/. out/e2_cloze/'

# all 13 calibratable families on one pod; 64 concepts x 7 configs
# (baseline + {ridge,dom,rand} x {selected, lower}) x 10 prefixes x 128 toks
# ≈ 4.5k generations ≈ 57k batched decode steps -> ~0.5-1.5 h on H100
ssh <podE3> 'cd /workspace/oracle-encodings/concept_probes/stage6_1/code && \
  nohup env FAMILIES="months,seasons,color_wheel,location_type,costliness,physical_size,weekdays,moon_phases,duration,directions,continents,lovingness,harmfulness" \
    SCRIPTS="e3" UPLOAD=1 bash pod_run.sh > /workspace/e3_run.out 2>&1 & echo started'
```

(E3 can also ride on fleet pod A after its queue drains — it is
resume-safe per (concept, arm, factor, prefix).)

## Step 4 — E3 judging (LOCAL, needs .env OPENROUTER_API_KEY; no GPU)

```bash
# pull generations from HF (uploaded by pod_run UPLOAD=1)
hf download kaushikreddyxyz/concept-probes-gemma2-2b \
  --include "stage6_1/out/e3/generations_*.jsonl" --local-dir /tmp/e3pull --quiet
mkdir -p concept_probes/stage6_1/out/e3
cp -a /tmp/e3pull/stage6_1/out/e3/. concept_probes/stage6_1/out/e3/

# smoke first (no API): 
python concept_probes/stage6_1/code/e3_judge.py --smoke --out concept_probes/stage6_1/out

# real run: mercury-2 via OpenRouter, K=3 paraphrased rubrics, ~4.5k gens / 6
# per call x 3 variants ≈ 2.2k calls ≈ $2-5 at stage-4 mercury rates; resume-
# safe (audit-log cache), hard cap:
python concept_probes/stage6_1/code/e3_judge.py --out concept_probes/stage6_1/out --cap-usd 20

# then push judged results back to HF:
hf upload kaushikreddyxyz/concept-probes-gemma2-2b \
   concept_probes/stage6_1/out/e3 stage6_1/out/e3 \
   --commit-message "stage6_1 E3 judged scores"
```

---

## Teardown reminder

Pods bill while idle. After each pod's status.tsv shows every script rc=0 AND
the final `upload after final: ok` line: `./cleanup-pod.sh <pod-id>` (runpod
skill). Do NOT stop-and-keep pods overnight — Stage-5 lesson: pod-side state
dies on Stop→Start anyway; everything needed is on HF + GitHub.

Budget guard: `runpodctl me` before/after each wave; total GPU target
$60–90 (§6.1.8).
