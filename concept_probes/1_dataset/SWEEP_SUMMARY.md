# Stage 4 sweep — closing summary (2026-07-02)

COMPLETE, including post-sweep repairs. Final cost **$136.41**.

Post-sweep repairs (verified):
- **Hazard hardfix**: 4,815 mis-surfaced color_wheel hard negatives flagged out,
  4,833 corrected ones merged; the family's hard-negative strengths dropped to
  mean 0.174 / p90 0.222 (was p90 0.556) — now at the faint-echo level.
- **Form-holdout supplements**: +4,278 conditioned examples merged; final §6.2
  lexical-holdout pool = **12,608 rows**. The only classes still under 50 are the
  documented single-form cases (may; orange, yellow; the four moon phases with no
  surface variant) whose G-ratio falls back to the implicit slice by convention.

| family | gen_ok | judged | probe classes | dataset rows |
|---|---|---|---|---|
| months | 29,082 | 29,082 | 12 | 47,543 |
| weekdays | 19,167 | 19,167 | 7 | 28,722 |
| seasons | 12,254 | 12,254 | 4 | 16,914 |
| color_wheel | 30,681 | 30,681 | 12 | 49,094 |
| directions | 21,424 | 21,424 | 8 | 32,777 |
| moon_phases | 17,474 | 17,474 | 8 | 28,589 |
| continents | 16,085 | 16,085 | 6 | 23,865 |
| location_type | 7,617 | 7,617 | 2 | 8,204 |
| costliness | 3,712 | 3,712 | 1 | 3,709 |
| physical_size | 3,708 | 3,708 | 1 | 3,708 |
| lovingness | 3,717 | 3,717 | 1 | 3,715 |
| duration | 3,687 | 3,687 | 1 | 3,687 |
| harmfulness | 3,716 | 3,716 | 1 | 3,715 |
| **total** | **172,324** | **172,324** | **64** | **254,242** |

- **Cost**: $131.06 all-in (vs $121 pre-sweep estimate; includes pilot, re-runs,
  and the mis-surfaced-hazard sunk cost). Judging dominated (~$115).
- **Disk**: 1.9 GB (audit logs dominate; datasets ~450 MB).
- **Quality invariants held across families**: explicit strengths ~0.6–0.75 mean,
  hard negatives at the faint-echo level (~0.17 ≈ score 1), neutrals 91–98% zero,
  intensity level→judged-strength curves perfectly monotone (see reports/).
- Known caveats: color_wheel implicit 34% judged-zero (tertiary hues are hard to
  pin implicitly); moon_phases 0.3% unmatched quotes; ~74 seasons form-holdout
  rows have 2-of-3 judge votes (rate-limit era).

## Operational lessons (for anyone re-running)
1. OpenRouter's `inception/mercury-2` is served by Inception — pushing OR harder
   saturates the same pool. OR concurrency 96 is the sweet spot; 224 stalls.
2. Provider rate limits debit RESERVED max_tokens (incl. hidden reasoning
   tokens), not actual completions — budget accordingly (or_client `limits`).
3. Deterministic task→provider assignment + resume = leftover work pinned to the
   slow provider; use capacity-based adaptive selection (judge.py).
4. gpt-oss needs `reasoning_effort: low` + generous max_tokens or it burns the
   completion budget on hidden reasoning and returns empty content.
5. Every model response shape WILL eventually be malformed somewhere: strict
   json_schema kills most drift; the rest needs type guards at every field.
