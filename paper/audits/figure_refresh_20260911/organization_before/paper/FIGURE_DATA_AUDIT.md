# Figure and completed-data audit

Evidence refresh: 2026-08-26. The canonical 750-cell registry and DAPO state are
audited through 12:06 EDT. PointMaze and PointMaze Tour are
outside this audit. “Available” means a terminal sampled endpoint or a
preregistered mechanism record exists; it does not turn a partial prefix into a
five-seed estimate.

## 1. Data that belongs in each current plot

| Plot | Completed data that should be represented | Current action |
|---|---|---|
| `modecollapse_story` | The frozen, mechanically selected Qwen2.5-3B Graph example and its matched Dr.GRPO/ReplayDr.GRPO trajectory | Keep the illustrative selection frozen. The fifth completed Graph pair belongs in the all-endpoint figures, not in a post-outcome reselection of the example. |
| `baseline_collapse_precheck` | All 150 registered pass-0 and pass-8 cells of the two verifier-only objectives: E78/E79/E80-R1 Dr.GRPO controls and the E95 plain-GRPO cohort with the E114 Qwen2.5-3B seed extension, five domains x five seeds x three scales | Complete; every cell has all four fixed-seed draws at both endpoints, so no panel is a prefix. Retained-breadth percentages are withheld below a .05 pass-0 extra-modes floor, which only Graph and PantryPlan clear. Pass 0 is read per arm because E80-R1 seeds 71--74 do not share MathIR and PantryPlan evaluation requests with the plain-GRPO arm. |
| `modebench_examples` | Canonical executable examples for Graph, Countdown, Python, MathIR, and Pantry | No run-result update; this is benchmark/verifier evidence. |
| `verified_replay_mechanism` | Verify, canonicalize, store, and recurrently replay the retained successful answer | No endpoint update; this is the conceptual ReplayDr.GRPO mechanism. |
| `terminal_pass8_distinct8_frontier` | Standardized terminal endpoints for the current Dr.GRPO, GRPO, ReplayDr.GRPO, UCPO, and RLEP-Dr methods, with exact method-specific `n` | Refreshed as the current-method Figure 3. Dr.GRPO, GRPO, and ReplayDr.GRPO are `n=5` throughout; UCPO is `n=5` in all ten smaller-model cells; RLEP-Dr is `n=5` in nine smaller-model cells and `n=2` in Falcon Python. Unsupported Qwen2.5-3B UCPO/RLEP-Dr cells stay blank. Retired Semantic-MaxEnt is appendix-only, and DAPO is excluded until standardized pass@8/breadth endpoints exist. |
| `xmode_adaptive_cross_scale_distinct_at_k` | Seed-matched Dr.GRPO versus adaptive Semantic MaxEnt + ReplayDr.GRPO from E89/E91/E92 for K=1,2,4,8,16,32 | Current. E80 has additional 3B controls, but E92 has only seed 70, so the valid treatment intersection remains `n=1` at 3B. |
| `xmode_adaptive_cross_scale_pass_at_k` | The same exact E89/E91/E92 seed intersections and K budgets, reporting sampling correctness | Current for the same reason as the distinct-at-K companion. |
| `direct_baseline_learning_curves_static_strip` | All terminal UCPO blocks/prefixes from E97/E99/E115 and all terminal sparse RLEP-Dr prefixes from E98-R1/E100/E116, with exact `n` | Refreshed. Includes all ten complete smaller-model UCPO blocks and nine complete RLEP-Dr blocks, including newly complete Qwen Countdown and MathIR; unsupported cells remain blank. GRPO is handled by the endpoint forest and core trajectory grids. |
| `replay_mechanism_telemetry_qwen05b` | All 25 E78 ReplayDr.GRPO runs and all 25 exact-zero controls across all 3,072 updates: bank occupancy, capacity hits, realized dose, and paired update time | Current and terminal. Do not infer identity-level survival because those snapshots were not retained. |
| `bank_occupancy_retained_breadth` | All 25 E78 replay seeds joined to normalized trajectory AUC and terminal correctness-adjusted breadth | Current and terminal; keep this descriptive and unpooled. |
| `sustained_auc_effects_qwen05b` | All five E78 domains at paired `n=5`, with correctness, distinct-mode, and correctness-adjusted trajectory-AUC effects over 17 checkpoints | Current and terminal. |
| `fixed_semantic_factorial_cross_scale_strip` | Complete Qwen2.5-0.5B and Falcon3-1B four-arm factorials from E78/E79/E81/E82/E83/E85/E86; exact seed-70 3B Dr.GRPO, ReplayDr.GRPO, and fixed semantic-on-replay from E87 | Current. The absent 3B semantic-only arm must remain blank. |
| `fixed_semantic_factorial_effects` | The complete Qwen2.5-0.5B and Falcon3-1B five-domain paired main effects and interactions at `n=5` | Current and terminal. This is evidence for the legacy fixed-semantic estimator, not the repaired v6 estimator. |
| `adaptive_semantic_gate_e88` | All nine E88 controller records for the unreachable 5% target | Current mechanism-only failure diagnostic. |
| `adaptive_mechanism_outcomes_qwen05b` | All 25 E89 terminal outcomes plus controller occupancy and final-ratio checks | Current. Retain the explicit mechanism-gate failure; outcomes are descriptive. |
| `e72_decoding_frontier` | Frozen pass-12 policies, five domains, seeds 43--47, six decoding temperatures, and the 240 reproduction checks | Keep frozen and explicitly historical; do not mix with current x-Mode results. |
| `adaptive_semantic_replay_cross_scale_strip` | All terminal E89 and E91 five-seed trajectories plus exact E92 seed-70 trajectories | Current, with exact `n=1` at 3B and no scale pooling. |
| `replay_dose_qwen05b_progress_static_strip` | All 25 E90 cells; seeds 43--47 in every Qwen domain | Refreshed. All five complete blocks license their registered five-seed analyses. |
| `open_bank_bundle_endpoint_effects` | All 25 E102 endpoints versus matched ReplayDr.GRPO and Dr.GRPO plus admissions and priority telemetry | Current and terminal. This remains a bundle effect, not component attribution. |
| `cross_scale_terminal_endpoint_effects` | Every paired pass-8 E78/E79/E80r1 row, plus every available Falcon E95 GRPO endpoint; intervals only at paired `n=5` | Refreshed from the immutable ledgers. All 15 core model--domain blocks are complete at paired `n=5`, including all 25 Qwen2.5-3B pairs. |
| `qwen3b_exact_endpoint_progress` | Every terminal 3B endpoint for Dr.GRPO, GRPO, ReplayDr.GRPO, fixed semantic-on-replay, and adaptive semantic-on-replay | Refreshed. Dr.GRPO, GRPO, and ReplayDr.GRPO are `n=5/5` on all five domains; both legacy semantic-on-replay methods remain exact `n=1`. |
| `direct_comparator_endpoint_effects` | GRPO, UCPO, and RLEP-Dr minus the matched Dr.GRPO endpoint, one model/domain cell at a time, using every terminal seed intersection | Refreshed in the matched-baseline forest format: paired seed circles, a zero line, and a mean diamond plus paired Student-t interval only at `n=5`. GRPO is 75/75 with all 15 model--domain blocks complete. UCPO has all ten smaller-model blocks complete. RLEP-Dr is 47/75, with nine complete blocks plus Falcon Python `n=2`. |
| `core_retention_falcon1b_static_strip` | Every available distinct@8 checkpoint for GRPO, Dr.GRPO, and ReplayDr.GRPO at all three scales | Refreshed with current ledgers and exact method-specific `n`. |
| `core_retention_falcon1b_pass8_static_strip` | The same current core and GRPO trajectories, reporting pass@8 | Refreshed. |
| `core_retention_falcon1b_mean8_static_strip` | The same current core and GRPO trajectories, reporting mean@8 | Refreshed. |

## 2. Plots still missing

| Priority | Missing plot | Data/status | What it adds |
|---|---|---|---|
| Done | Direct-comparator paired endpoint forest | E95/E97/E98-R1/E99/E100 terminal cells and exact prefixes | Gives the clean matched answer to “how does this compare with GRPO, UCPO, and generic success replay?” without crowding the treatment frontier. |
| Done | E103 fallback-isolation endpoint and mechanism panel | 25/25 terminal; mechanism audit passed | Added paired E103-minus-E102 pass@8/distinct@8 effects beside per-seed activations, admissions, and extra groups. |
| Done | E108 admission-to-retention funnel | 10/10 terminal; mechanism gate passed | Added per-domain passive/adaptive funnel counts and bounded refresh/priority actuation, explicitly as mechanism rather than efficacy evidence. |
| Done | Repaired-v6 estimator mechanism diagnostic | Effective E104/E106/E110 grid is 15/15 terminal and the combined mechanism gate passes | Compiled 3x5 eligibility, two-mode support, centered RMS/mean, replay actuation, and three explicit zero-pressure cells. This validates implementation behavior, not long-horizon efficacy. |
| Historical; must not render | Repaired-v6 three-scale outcome forest and trajectory-AUC panel | 24 artifacts are immutable/audit-only | Do not render or pool this as the corrected treatment. |
| Supporting, not standalone | Repaired Python baseline check | 15/15 complete | Preserve it as the registered repaired-Python ReplayDr.GRPO comparator. |
| Done | Verified-support mechanism audit | 15/15 complete; mechanism audit passes | Added the 3x5 mechanism diagnostic. Eleven cells complete the full discovery-to-pressure chain; four explicit boundary cells remain visible. This is mechanism evidence, not efficacy. |
| Done; exploratory | Verified-support Semantic-MaxEnt + ReplayDr.GRPO vs ReplayDr.GRPO | Integrity audit admits 49 smaller-model pairs and excludes Qwen2.5-3B | Compiled paired endpoint forest with exact `n`; one Falcon Countdown comparator is excluded for conflicting duplicate rows. The bundled estimand, exploratory evidence class, and unavailable trajectory AUC are explicit. |
| No standardized result plot | Complete-recipe DAPO comparator | Four Qwen Graph runs provide upstream `acc@1` diagnostics, but standardized pass@8/breadth is unavailable | Omit an efficacy marker, mean, interval, or control comparison. |

The superseded group-centered diagnostic exposes its singleton-support
boundary and contributes no efficacy estimate. The
repaired Python comparator and the 15-cell verified-support mechanism audit are
complete. The endpoint comparison contains 49 integrity-valid smaller-model
pairs and is exploratory bundle evidence, not a component-isolated
estimate.

## 3. Data with no plotted result yet

All completed data licensed for manuscript visualization now has a compiled
figure. The verified-support mechanism has a terminal diagnostic, and the
exploratory endpoint comparison has a 49-pair forest; Qwen2.5-3B and one
conflicting comparator pair are excluded. DAPO has four upstream `acc@1`
diagnostics but no standardized pass@8/breadth endpoint, so it has no result
marker or aggregate. No other paper-eligible result lacks a plotted or
tabulated representation at this cutoff.
