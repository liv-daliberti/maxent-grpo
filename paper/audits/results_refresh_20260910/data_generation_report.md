# September 10 endpoint data generation

The source census was collected from 2026-09-10 16:49:41.871296 to 16:53:47.899239 UTC. It admits only exact step-3072 endpoints with all four registered evaluation draws and no selected conflicting retry. Every retained endpoint value from September 9 is unchanged; none disappeared. No experiment, scheduler, launcher, or analysis helper was changed.

| Campaign | Admitted endpoints (change) | Matched seed contrasts (change) | Complete blocks (change) |
|---|---:|---:|---:|
| E118 | 137/150 (+11) | 66/75 (+5) | 12/15 (+1) |
| E119 | 85/100 (+2) | 41/50 (+1) | 4/5 (unchanged) |
| E120-R1 | 41/45 (+6) | 41/45 (+6) | 7/9 (+1) |

Training receipt counts equal admitted endpoint counts in this census. E118 blocks require two MaxRL arms, E119 blocks all four methods, and E120 blocks both weighting arms. E119 contrasts count the two pairwise comparisons separately. E120 endpoint counts cover frequency treatments; uniform comparator endpoints are reused. All 41 E120 pairs pass mechanism eligibility, including seven complete blocks.

The resulting report includes all 34 planned contrasts, 33 with observed pairs and 27 with five seeds. Compared with September 9, there are 19 additional treatment endpoints, 12 additional matched seed contrasts, and two additional complete blocks.

## Added evidence

- E118 Qwen-3B Graph now completes five seeds. ReplayMaxRL minus MaxRL yields delta pass@8 +0.0234375 (95% descriptive paired Student-t interval [-0.047825697, +0.094700697]), delta distinct@8 +0.186328125 ([-0.022363709, +0.395019959]), and delta B8 +0.162890625 ([+0.020802305, +0.304978945]). Its correctness and raw distinct-mode intervals include zero.
- E118 Qwen-3B Countdown now has paired seeds 70 and 74: delta pass@8 +0.17578125, delta B8 +0.1494140625. MathIR now has paired seeds 70, 71, 74: delta pass@8 +0.220703125, delta B8 +0.0071614583. These partial estimates have no interval. Pantry retains only paired seed 72 despite an additional unpaired replay endpoint.
- E119 PantryPlan adds the first ReplayDr.GRPO/Dr.GRPO pair, seed 46: delta pass@8 +0.15625, delta distinct@8 +0.232421875, and delta B8 +0.076171875. There is no MaxRL pair and no four-arm Pantry seed. The strict four-arm factorial estimates and intervals are numerically unchanged from September 9.
- E120 Falcon-1B PantryPlan now completes five mechanism-eligible pairs. Uniform minus frequency replay gives delta pass@8 -0.021484375 (exhaustive paired percentile-bootstrap 95% interval [-0.062890625, +0.02578125]), delta distinct@8 -0.1671875 ([-0.626953125, +0.43984375]), and delta B8 -0.145703125 ([-0.5640625, +0.4140625]). All intervals include zero, so this complete result remains inconclusive about direction.
- E120 Qwen-3B Graph now has four pairs, seeds 70–73: delta pass@8 -0.00634765625, delta B8 +0.06982421875. PantryPlan adds two pairs, seeds 73–74: delta pass@8 +0.0068359375, delta B8 +0.2060546875. All currently available pairs are mechanism eligible; partial blocks have no interval and are not complete mechanism blocks.

B8 = distinct@8 minus pass@8 measures additional correct modes beyond the first and remains coupled to correctness. Complete-block intervals are descriptive, unadjusted for multiplicity or repeated dated looks. There is no missing-cell imputation, no model/domain pooling, and no universal replay-weighting superiority claim. The September 4 E120 primary analysis remains byte-identical, including its original 25 Qwen treatment endpoints. The standard latest-effects table retains its original comparison baseline of September 6; the changes above are explicitly versus September 9.

## Artifacts and validation

- Frozen source: `paper/audits/results_refresh_20260910/latest_endpoints.json` (SHA-256 `a27e6762a3f05d6f3fa6b5f6659cf400d4a929a67db892c69f2f6c96c424a8e3`).
- Census summary: `paper/results/latest_results_20260910.json` (SHA-256 `dbaaf327e1765cbf2dc2d39d64599b69b17333487efb44762ffc09709f4b9d60`).
- Readable report, full-precision CSV/JSON, and four TeX table bodies: `paper/results/current_campaign_results_20260910*`.
- Overview figure: `paper/figures/current_campaign_results_20260910.pdf` and `.png`.
- Strict E119 factorial: `paper/results/level2_factorial_contrasts_20260910.json` and its table body.
- `data_generation_validation.json` independently checks 69 contrasts and 276 metric summaries against admitted audit rows, exact seed intersections, t/bootstrap arithmetic, unchanged retained endpoints, stable frozen ledgers, and 26 preserved historical/helper file hashes.
- Three existing census-gate regression tests passed (`data_builder_tests.stdout`).
- All 14 generated result/overview outputs reproduced byte-identically from the frozen audit (`data_reproduction_verification.json`). E120 mechanism seed sets were independently reconciled, and the full overview plot was visually inspected for readable rows, exact denominators, unclipped axes/source footer, and correct partial/full interval encoding.

The existing builders were invoked unchanged. The exact commands and pre-generation hashes are in `data_generation_before.json`; the arithmetic checker is retained as `validate_data_generation.py`. The root agent owns manuscript text, README integration, and manuscript build checks; a separate agent owns Figures 5 and 6.
