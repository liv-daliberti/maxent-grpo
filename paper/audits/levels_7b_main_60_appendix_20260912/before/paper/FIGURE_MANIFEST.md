# Paper figure inventory

The ICLR hosted overview is summarized by a per-model/per-level table of `pass@8` and `distinct@8`; its complete domain plot now appears in the appendix. ICLR and the workshop each have **eight main and nineteen supplementary figures**. Both compile all **27 figures**: the nineteen original displays, four additional quantitative views, two later sampling-budget plots, the restored Methods construction panel, and the Pantry adaptation follow-up. Existing frozen estimates are unchanged.

## Current figure sequence

| Figure | Original asset | Restored role | Scope retained |
|---|---|---|---|
| 1, both editions | [modecollapse_story](figures/modecollapse_story.pdf) | Opening Graph illustration, page one. | Selected example; Dr.GRPO versus ReplayMaxRL changes both objective and replay. Matched comparisons establish separate effects. |
| 2, both | [modebench_examples](figures/modebench_examples.pdf) | Five-domain examples beside the ModeBench definition. | Execution keys and alias rules define task-specific output modes. |
| 3, both | [modebench_level_construction](figures/modebench_level_construction.pdf) | Methods admission check: ICLR Section 2.2, page 3; workshop page 2. | Frozen Qwen2.5-0.5B held-out pass@8 versus mean distinct@8; eight measured domain/level cells. Remaining measurements and Level-4/5 admission are pending. |
| 4, both | [verified_support_story](figures/verified_support_story.pdf) | Verified-memory schematic beside the method. | Banked exemplars continue receiving a signal; this is not a neural retention guarantee. |
| 5, both | [experiment1_retention_comparator_matrix](figures/experiment1_retention_comparator_matrix.pdf) | Original Dr.GRPO replay and direct-comparator results. | 74 replay pairs; Falcon Countdown has four, no five-seed interval. |
| 6, both | [e118_all_scale_factorial_progress](figures/e118_all_scale_factorial_progress.pdf) | Original factorial endpoints and seed paths. | All three scales; 75 MaxRL pairs; aggregate tracks use their exact common cohorts. |
| 7, both | [modebench_level_admission](figures/modebench_level_admission.pdf) | Terminal training comparison, separate from Methods admission. | Within-level replay inference; different cross-level populations and protocols do not isolate difficulty. |
| ICLR appendix; 8, workshop | [hosted_verified_breadth](figures/hosted_verified_breadth.pdf) | ICLR main uses the per-level averages table; complete domain plot remains available. | Seven deployments; frozen normalizer and revised Opus Python wording explicit; descriptive observations. |
| 8, ICLR; workshop supplement | [gpt56_temperature_curve](figures/gpt56_temperature_curve.pdf) | Original temperature view beside hosted sensitivity. | Controlled no-reasoning sweep; historical reference retained in appendix tables only; observed optimum only. |

The workshop figure pages are 1, 2, 2, 2, 3, 3, 4, 4 after adding the construction panel and tightening captions. Figure order is preserved around the added panel. ICLR retains the original section roles and page-one opener within nine main pages; paragraph and mathematics changes can shift later page breaks.

## Additional quantitative views remain available

| Asset | Supporting placement and evidence |
|---|---|
| [concentration_story](figures/concentration_story.pdf) | Longitudinal and matched replay collision analysis; Graph/Pantry at three scales, explicit eligibility and both disjoint-stream orientations. All 135 contrasts remain in the complete report. |
| [replay_factorial_effects](figures/replay_factorial_effects.pdf) | Controlled-results supplement; domain-specific replay-minus-control effects under both objectives, on pass@8 and distinct@8. |
| [replay_key_weighting](figures/replay_key_weighting.pdf) | Uniform-versus-frequency subsection; frozen 0.5B primary analysis, five seeds, original paired-bootstrap intervals. |
| [replay_level2_effects](figures/replay_level2_effects.pdf) | Within-Level-2 results; common four-arm cohort, four complete domains n=5 and Pantry n=1 descriptive. |

These four figures supplement the original main displays. The main text retains their findings and explicit references, with measurement identities and conditional assumptions visible beside the original examples and schematic.

## Other supporting figures

| Assets | Evidence |
|---|---|
| `e118_scale_extensions_appendix` | Absolute domain/scale factorial endpoints. |
| `factorial_training_curves_pass8`, `factorial_training_curves_distinct8` | Fixed-cohort trajectories; missing checkpoints remain gaps; bands show seed ranges. |
| `level2_factorial_training_curves` | All Level-2 arms, with partial Pantry histories. |
| `direct_comparator_endpoint_effects` | GRPO, UCPO and sparse RLEP paired endpoint comparisons with exact cohorts. |
| `direct_baseline_learning_curves_static_strip`, `direct_baseline_learning_curves_pass8` | Matching modes and accuracy trajectories for the smaller-model comparator populations. |
| `baseline_collapse_precheck` | All 150 Dr.GRPO/GRPO runs and their raw extra-mode changes. |
| `e121_fixed_bank_survival` | Teacher-forced exemplar scores, including declining tails; not exact mode probabilities. |
| `pantry_adaptation_recovery_20260912` | Inference follow-up: saved-plan survival and bounded recovery after independently specified feasible Pantry outages. |
| `frontier_comparison_20260911_graph` | Original-protocol hosted Graph correctness and correct-pair collision. |
| `modebench_prompt_ablation_local` | Completed local prompt comparison; hosted panel omitted. |
| `modebench_discovery_curves_local`, `modebench_discovery_correct_budget_local` | Later local sampling-budget and fixed-correct-count analyses; retained eligibility, rare MathIR exception, and Pantry ordering changes. |

## Source and build checks

`make -C paper figures-main` renders nine overview assets, including the hosted domain plot now placed in the ICLR appendix. `python ops/build_paper_hosted_level_averages.py` reproduces the hosted summary table from the same complete cohorts. `figures-supporting` renders the additional quantitative views and retained evidence. Figure JSON sidecars bind exact displayed estimates, cohorts, source bytes, and outputs. The construction plot is reconstructed from eight complete frozen-0.5B receipt sets; its sidecar authenticates native responses, canonical-set metrics, sources, and PDF/PNG outputs.

All existing numerical, source, prompt, discovery-curve, and proof checks remain active. Editorial contracts protect the restored main sequence and the appendix placement of the added views. The workshop retains its official style, four-page main, source receipts, and independently checked source bundle.

See the [restoration record](audits/figure_restoration_20260911/README.md), [machine-readable placement](audits/figure_restoration_20260911/figure_placement.json), and [coverage audit](FIGURE_DATA_AUDIT.md). The earlier reorganization and its previous placements remain archived.

## Inference follow-ups (2026-09-12)

Both appendices include `fig:pantry-adaptation-recovery`, rendered as `figures/pantry_adaptation_recovery_20260912.pdf` by `ops/build_paper_inference_followups.py`. Its numerical record is `results/inference_followups_20260912.json`; source hashes bind the complete offline, local coarse-key and Pantry analyses. `python ops/build_paper_inference_followups.py --check` verifies retained figures/tables and reconstructs Pantry problem means and stopping identities. The same appendix includes all 21 model-pair comparisons, both constituent baselines, hosted/local coarse keys, all Pantry controls, dietary outcomes and collection costs.
