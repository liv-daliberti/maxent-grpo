# Paper figure inventory

The reorganized manuscripts compile the same **23 source-bound figures**: **five main and eighteen supplementary** in ICLR, and **three main and twenty supplementary** in the four-page workshop version. The title remains *Mode Collapse in RLVR & ModeBench*. Earlier placements and manifests are preserved in [the reorganization audit](audits/narrative_reorganization_20260911/).

## Main narrative

| ICLR figure | Workshop placement | Asset | Purpose and evidence boundary |
|---|---|---|---|
| 1: Training concentration and replay mitigation | Figure 1 | [concentration_story](figures/concentration_story.pdf) | Twelve Graph/Pantry before/after contrasts and twelve matched replay contrasts across three scales; separate eligibility, nominal intervals and both disjoint-stream orientations. Full five-domain data remain in the appendix. |
| 2: Hosted relevance of the measurement | Supplement | [hosted_verified_breadth](figures/hosted_verified_breadth.pdf) | Seven deployments, five domains, three levels; per-response accuracy and distinct@8. Explicit revised Opus Python wording and frozen formatting normalization; no controlled training-cause claim. |
| 3: Central controlled replay factorial | Figure 2 | [replay_factorial_effects](figures/replay_factorial_effects.pdf) | Replay minus matched control under Dr.GRPO and MaxRL, all three scales and all five Level-1 domains; pass@8 and distinct@8. Dr.GRPO has 74 pairs, MaxRL 75; Falcon Countdown Dr has four, no CI. |
| 4: Why balance verified keys | Figure 3 | [replay_key_weighting](figures/replay_key_weighting.pdf) | Uniform minus fresh-frequency weighting, frozen Qwen2.5-0.5B primary comparison. Pass@8 and extra modes retain the original paired-bootstrap intervals and five seeds. |
| 5: Within-Level-2 replication | Supplement | [replay_level2_effects](figures/replay_level2_effects.pdf) | Qwen2.5-0.5B, two replay effects, common four-arm seed intersection. Four complete domains n=5; Pantry n=1 descriptive. No cross-level difficulty-effect interpretation. |

## Supporting evidence

All original figures remain compiled. Their complete numerical and source checks are retained even when their placement changes.

| Asset | Role |
|---|---|
| [modebench_examples](figures/modebench_examples.pdf) | Five worked validator/key examples, with benchmark construction. |
| [modebench_level_admission](figures/modebench_level_admission.pdf) | Construction admission and cross-level absolute reference display, beside benchmark construction. |
| [verified_support_story](figures/verified_support_story.pdf) | Fresh objective and canonical replay schematic, beside the algorithm. |
| [experiment1_retention_comparator_matrix](figures/experiment1_retention_comparator_matrix.pdf) | Full Dr.GRPO and alternative-method matrix, beside controlled results. |
| [e118_all_scale_factorial_progress](figures/e118_all_scale_factorial_progress.pdf) | Original absolute endpoints and seed trajectories, beside controlled results. |
| [e118_scale_extensions_appendix](figures/e118_scale_extensions_appendix.pdf) | All three models and five domains, absolute factorial endpoints. |
| [factorial_training_curves_pass8](figures/factorial_training_curves_pass8.pdf) | Full fixed-cohort training accuracy trajectories, with gaps and seed-range bands. |
| [factorial_training_curves_distinct8](figures/factorial_training_curves_distinct8.pdf) | Identical registered cohorts and checkpoints, measured in distinct verified keys. |
| [level2_factorial_training_curves](figures/level2_factorial_training_curves.pdf) | All Level-2 methods and both metrics, with partial Pantry histories explicit. |
| [direct_comparator_endpoint_effects](figures/direct_comparator_endpoint_effects.pdf) | GRPO 75, UCPO 50, and sparse RLEP 47 pairs; exact incomplete cells. |
| [direct_baseline_learning_curves_static_strip](figures/direct_baseline_learning_curves_static_strip.pdf) | UCPO/RLEP mode trajectories at 0.5B and 1B only. |
| [direct_baseline_learning_curves_pass8](figures/direct_baseline_learning_curves_pass8.pdf) | Matching UCPO/RLEP accuracy trajectories. |
| [modecollapse_story](figures/modecollapse_story.pdf) | Original selected Graph illustration; both fresh objective and replay differ. |
| [baseline_collapse_precheck](figures/baseline_collapse_precheck.pdf) | All 150 Dr.GRPO/GRPO runs; raw extra-mode diagnosis with per-method initial values. |
| [e121_fixed_bank_survival](figures/e121_fixed_bank_survival.pdf) | Five Graph seeds; teacher-forced fixed-exemplar scores, including declining tails. |
| [gpt56_temperature_curve](figures/gpt56_temperature_curve.pdf) | No-reasoning temperature sweep and separate medium-reasoning reference, with hosted sensitivities. |
| [frontier_comparison_20260911_graph](figures/frontier_comparison_20260911_graph.pdf) | Original-protocol hosted Graph correctness and correct-pair collision. |
| [modebench_prompt_ablation_local](figures/modebench_prompt_ablation_local.pdf) | Authenticated original-versus-hints-removed local comparison; hosted panel remains omitted. |

## New figures and reproduction

The concentration figure is generated by `ops/plot_paper_concentration_story.py`. The factorial, weighting, and Level-2 effect figures are generated by `ops/plot_paper_reorganized_results.py`. Each new PDF/PNG has a JSON sidecar containing exact displayed estimates, intervals, cohorts, source hashes, builder hash, output hashes, and interpretive limits. All estimates reuse frozen results; the figure update launches no experiments and changes no scientific endpoint.

`make -C paper figures-main` renders the five main assets; `figures-supporting` retains the original diagrams and full displays. The current numerical contracts still validate every original source and calculation. The editorial checks now verify five ICLR or three workshop main figures, their order, page budgets, and source-bound supplement placement. The workshop keeps official style, anonymity, complete copied inputs, and compiled-artifact checks.

## Measurement and cohort rules

Conditional concentration uses eleven nominal streams under the audited runtime mapping, deterministic representatives, and separate five/six-stream sensitivities. Eligibility is explicit; all 135 contrasts and both documented source-census phases remain in the result artifact. These random-eligible-population means are descriptive. Main figure 1 foregrounds Graph/Pantry because both have substantial initial sampled breadth; the full-domain analysis is not omitted.

Full-test pass@8 and distinct@8 retain all four intact eight-output groups. Per-sample correctness and conditional concentration on selected prompts are separate outcomes. Hosted collision uses correct-pair weights rather than the equal-prompt training weighting.

The complete primary MaxRL cohort contains 75 pairs, including five in every 3B domain. Dr.GRPO retains 74 pairs and the registered Falcon Countdown exclusion. Level 2 comparisons use their declared common-four-arm cohort; broader arm availability does not change that intersection. Weighting uses the original frozen primary analysis, with expanded scales preserved separately.

[Machine-readable placement](audits/narrative_reorganization_20260911/figure_placement.json) records both editions. [Coverage and scientific boundaries](FIGURE_DATA_AUDIT.md) accompany the figure inventory.
