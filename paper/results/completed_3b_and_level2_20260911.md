# The 3B MaxRL comparison is complete

**Qwen2.5-3B now has all 50/50 completed, admitted terminal runs:** five domains × MaxRL/ReplayMaxRL × five paired seeds (70–74). All five domain comparisons are complete. Across the three models, E118 is complete at **150/150 runs and 15/15 five-seed comparisons**.

This September 11, 2026 completion update supersedes the snapshot collected at 06:43–06:45 UTC that still showed missing 3B cells. The final 3B training receipt arrived at **18:54 UTC**. The refreshed [endpoint audit](../audits/results_completion_20260911/latest_endpoints.json) and [numerical snapshot](latest_results_20260911.json) record the current collection times and source provenance. Training completion and exact step-3072 evaluation were checked separately; every admitted endpoint contains the four registered sampled draws.

## Finished 3B results

All rows use the same five paired seeds. Absolute results are **MaxRL → ReplayMaxRL**. Pass@8 is shown as a percentage; distinct@8 counts correct modes per prompt. **B8 = distinct@8 − pass@8** measures correct modes beyond the first, using pass@8 as a probability.

| Domain | Paired seeds | Pass@8 (%) | Distinct@8 | ΔB8 [95% interval] |
|---|---:|---:|---:|---:|
| Graph | 5/5 | 82.42 → 84.77 | 1.514 → 1.700 | +0.163 [0.021, 0.305] |
| Countdown | 5/5 | 60.43 → 75.39 | 0.626 → 0.927 | +0.152 [0.119, 0.185] |
| Python | 5/5 | 35.63 → 63.28 | 0.356 → 0.748 | +0.115 [0.080, 0.149] |
| MathIR | 5/5 | 26.37 → 51.29 | 0.264 → 0.519 | +0.0059 [0.0010, 0.0107] |
| PantryPlan | 5/5 | 68.75 → 75.78 | 0.963 → 1.642 | +0.609 [0.467, 0.751] |

ReplayMaxRL has positive mean effects on pass@8, distinct@8, and B8 in every 3B domain. All five B8 intervals lie above zero, although MathIR's extra-mode gain is tiny. The pass@8 effect remains inconclusive for Graph and Python because their intervals span zero; Countdown, MathIR, and PantryPlan have positive pass@8 intervals. These are descriptive, unadjusted paired Student-t 95% intervals, with no pooled effect across domains or correction for multiple comparisons.

## Level 2: four complete domains

**Graph, Countdown, Python, and MathIR are complete at 80/80 runs:** all four methods—Dr.GRPO, ReplayDr.GRPO, MaxRL, and ReplayMaxRL—have five terminal seeds in each domain. Their finished comparisons are presented in the [current campaign report](current_campaign_results_20260911.md#e119-level-2-factorial).

The full Level-2 campaign is **90/100 runs**. PantryPlan remains **10/20**: Dr.GRPO has 3/5 endpoints, ReplayDr.GRPO 4/5, MaxRL 1/5, and ReplayMaxRL 2/5. This yields **2/5 paired Dr.GRPO comparisons** (seeds 43 and 46), **1/5 paired MaxRL comparison** (seed 43), and **1/5 common four-method seeds**. PantryPlan is partial and is excluded from the four-complete-domain summary.

The earlier request to finish the missing **3B** cells is resolved. For **Level 2**, the remaining completion gap is confined to PantryPlan.
