# Paper figure inventory

The ICLR manuscript in [main.tex](main.tex) contains **40 numbered figures: 12 in the main text and 28 in the appendix**. The inventory below follows its compiled figure numbers. Figure 26 combines two assets; Figures 4 and 14 use the same asset at different sizes. The workshop edition has its own [manuscript](mathai2026/main.tex) and [build file](mathai2026/Makefile); its numbering is separate.

## Figure assets and numerical records

The records identify plotted values, source populations, and input files where applicable. Figures 2 and 7 are explanatory diagrams. Figure 6 is an illustrative probability schematic; its JSON records that construction. Figure 1 uses a selected model-output example, with its selection and source records linked below.

| Figure | Asset | Numerical or construction record | Rendering script |
|---|---|---|---|
| 1 | [modecollapse_story](figures/modecollapse_story.pdf) | [paper_graph_collapse_toy](../var/artifacts/paper_graph_collapse_toy.json) | [plot_paper_collapse_toy.py](../ops/plot_paper_collapse_toy.py) |
| 2 | [modebench_examples](figures/modebench_examples.pdf) | Diagram defined in renderer | [plot_paper_modebench_examples.py](../ops/plot_paper_modebench_examples.py) |
| 3 | [mode_diversity_levels_appendix](figures/mode_diversity_levels_appendix.pdf) | [mode_diversity_levels_appendix](figures/mode_diversity_levels_appendix.json) | [plot_paper_mode_diversity_levels.py](../ops/plot_paper_mode_diversity_levels.py) |
| 4 | [qwen_level_trends](figures/qwen_level_trends.pdf) | [qwen_level_trends](figures/qwen_level_trends.json) | [plot_paper_qwen_level_trends.py](../ops/plot_paper_qwen_level_trends.py) |
| 5 | [gpt56_all_levels32_sampling_budget](figures/gpt56_all_levels32_sampling_budget.pdf) | [gpt56_all_levels32_sampling_budget](figures/gpt56_all_levels32_sampling_budget.json) | [plot_paper_gpt56_all_levels32_sampling.py](../ops/plot_paper_gpt56_all_levels32_sampling.py) |
| 6 | [replay_bank_balance](figures/replay_bank_balance.pdf) | [replay_bank_balance](figures/replay_bank_balance.json) | [plot_paper_replay_bank_balance.py](../ops/plot_paper_replay_bank_balance.py) |
| 7 | [verified_support_story](figures/verified_support_story.pdf) | Diagram defined in renderer | [plot_paper_support_story.py](../ops/plot_paper_support_story.py) |
| 8 | [concentration_story_resampled](figures/concentration_story_resampled.pdf) | [concentration_story_resampled](figures/concentration_story_resampled.json) | [plot_paper_concentration_levels.py](../ops/plot_paper_concentration_levels.py) |
| 9 | [e118_all_scale_factorial_progress](figures/e118_all_scale_factorial_progress.pdf) | [e118_all_scale_factorial_progress](figures/e118_all_scale_factorial_progress.json) | [plot_paper_e118_all_scale_progress.py](../ops/exp_scaling/plot_paper_e118_all_scale_progress.py) |
| 10 | [replay_key_weighting](figures/replay_key_weighting.pdf) | [replay_key_weighting](figures/replay_key_weighting.json) | [plot_paper_reorganized_results.py](../ops/plot_paper_reorganized_results.py) |
| 11 | [modebench_level_admission](figures/modebench_level_admission.pdf) | [modebench_level_admission](figures/modebench_level_admission.json) | [plot_paper_modebench_levels.py](../ops/exp_scaling/plot_paper_modebench_levels.py) |
| 12 | [reference_kl_plane](figures/reference_kl_plane.pdf) | [reference_kl_comparison](results/reference_kl_comparison.json) | [plot_paper_reference_kl_plane.py](../ops/plot_paper_reference_kl_plane.py) |
| 13 | [replay_bank_decomposition](figures/replay_bank_decomposition.pdf) | [replay_bank_decomposition](figures/replay_bank_decomposition.json) | [plot_paper_replay_bank_decomposition.py](../ops/plot_paper_replay_bank_decomposition.py) |
| 14 | [qwen_level_trends](figures/qwen_level_trends.pdf) | [qwen_level_trends](figures/qwen_level_trends.json) | [plot_paper_qwen_level_trends.py](../ops/plot_paper_qwen_level_trends.py) |
| 15 | [mode_diversity_level_construction](figures/mode_diversity_level_construction.pdf) | [mode_diversity_level_construction](figures/mode_diversity_level_construction.json) | [plot_paper_mode_diversity_levels.py](../ops/plot_paper_mode_diversity_levels.py) |
| 16 | [frontier_level_grid](figures/frontier_level_grid.pdf) | [frontier_level_grid](figures/frontier_level_grid.json) | [plot_paper_frontier_level_grid.py](../ops/plot_paper_frontier_level_grid.py) |
| 17 | [mode_diversity_families_appendix](figures/mode_diversity_families_appendix.pdf) | [mode_diversity_families_appendix](figures/mode_diversity_families_appendix.json) | [plot_paper_mode_diversity_levels.py](../ops/plot_paper_mode_diversity_levels.py) |
| 18 | [replay_factorial_effects](figures/replay_factorial_effects.pdf) | [replay_factorial_effects](figures/replay_factorial_effects.json) | [plot_paper_reorganized_results.py](../ops/plot_paper_reorganized_results.py) |
| 19 | [e118_scale_extensions_appendix](figures/e118_scale_extensions_appendix.pdf) | [e118_scale_extensions_appendix](figures/e118_scale_extensions_appendix.json) | [plot_paper_e118_all_scale_progress.py](../ops/exp_scaling/plot_paper_e118_all_scale_progress.py) |
| 20 | [replay_level2_effects](figures/replay_level2_effects.pdf) | [replay_level2_effects](figures/replay_level2_effects.json) | [plot_paper_reorganized_results.py](../ops/plot_paper_reorganized_results.py) |
| 21 | [factorial_training_curves_pass8](figures/factorial_training_curves_pass8.pdf) | [factorial_training_curves_pass8](figures/factorial_training_curves_pass8.json) | [plot_paper_training_curves.py](../ops/exp_scaling/plot_paper_training_curves.py) |
| 22 | [factorial_training_curves_pmd](figures/factorial_training_curves_pmd.pdf) | [factorial_training_curves_pmd](figures/factorial_training_curves_pmd.json) | [plot_paper_mode_diversity_curves.py](../ops/exp_scaling/plot_paper_mode_diversity_curves.py) |
| 23 | [level2_factorial_training_curves](figures/level2_factorial_training_curves.pdf) | [level2_factorial_training_curves](figures/level2_factorial_training_curves.json) | [plot_paper_training_curves.py](../ops/exp_scaling/plot_paper_training_curves.py) |
| 24 | [level2_training_curves_pmd](figures/level2_training_curves_pmd.pdf) | [level2_training_curves_pmd](figures/level2_training_curves_pmd.json) | [plot_paper_mode_diversity_curves.py](../ops/exp_scaling/plot_paper_mode_diversity_curves.py) |
| 25 | [direct_baseline_learning_curves_pass8](figures/direct_baseline_learning_curves_pass8.pdf) | [direct_baseline_learning_curves_pass8](figures/direct_baseline_learning_curves_pass8.json) | [plot_paper_aligned_domain_strips.py](../ops/exp_scaling/plot_paper_aligned_domain_strips.py) |
| 26 | [concentration_story_all_scales](figures/concentration_story_all_scales.pdf), [concentration_levels](figures/concentration_levels.pdf) | [concentration_story_all_scales](figures/concentration_story_all_scales.json), [concentration_levels](figures/concentration_levels.json) | [plot_paper_concentration_story.py](../ops/plot_paper_concentration_story.py), [plot_paper_concentration_levels.py](../ops/plot_paper_concentration_levels.py) |
| 27 | [baseline_collapse_precheck](figures/baseline_collapse_precheck.pdf) | [baseline_collapse_precheck](figures/baseline_collapse_precheck.json) | [plot_paper_baseline_collapse_precheck.py](../ops/exp_scaling/plot_paper_baseline_collapse_precheck.py) |
| 28 | [e121_fixed_bank_survival](figures/e121_fixed_bank_survival.pdf) | [e121_fixed_bank_survival](results/e121_fixed_bank_survival.json) | [build_paper_e121_survival.py](../ops/exp_scaling/build_paper_e121_survival.py) |
| 29 | [modebench_prompt_ablation_local](figures/modebench_prompt_ablation_local.pdf) | [modebench_prompt_ablation_local](figures/modebench_prompt_ablation_local.json) | [analyze_modebench_prompt_ablation.py](../ops/analyze_modebench_prompt_ablation.py) |
| 30 | [decoding_objection](figures/decoding_objection.pdf) | [decoding_objection](figures/decoding_objection.json) | [plot_paper_decoding_objection.py](../ops/plot_paper_decoding_objection.py) |
| 31 | [modebench_discovery_curves_frontier](figures/modebench_discovery_curves_frontier.pdf) | [modebench_discovery_curves_frontier](figures/modebench_discovery_curves_frontier.json) | [analyze_modebench_discovery_curves.py](../ops/analyze_modebench_discovery_curves.py) |
| 32 | [modebench_discovery_correct_budget_frontier](figures/modebench_discovery_correct_budget_frontier.pdf) | [modebench_discovery_correct_budget_frontier](figures/modebench_discovery_correct_budget_frontier.json) | [analyze_modebench_discovery_curves.py](../ops/analyze_modebench_discovery_curves.py) |
| 33 | [modebench_discovery_curves_local](figures/modebench_discovery_curves_local.pdf) | [modebench_discovery_curves_local](figures/modebench_discovery_curves_local.json) | [analyze_modebench_discovery_curves.py](../ops/analyze_modebench_discovery_curves.py) |
| 34 | [modebench_discovery_correct_budget_local](figures/modebench_discovery_correct_budget_local.pdf) | [modebench_discovery_correct_budget_local](figures/modebench_discovery_correct_budget_local.json) | [analyze_modebench_discovery_curves.py](../ops/analyze_modebench_discovery_curves.py) |
| 35 | [pantry_adaptation_recovery_20260912](figures/pantry_adaptation_recovery_20260912.pdf) | [inference_followups_20260912](results/inference_followups_20260912.json) | [build_paper_inference_followups.py](../ops/build_paper_inference_followups.py) |
| 36 | [withdrawal_pmd_agreement](figures/withdrawal_pmd_agreement.pdf) | [portfolio_withdrawals_20260917](results/portfolio_withdrawals_20260917.json) | [build_paper_portfolio_withdrawals.py](../ops/build_paper_portfolio_withdrawals.py) |
| 37 | [frontier_comparison_20260911_graph](figures/frontier_comparison_20260911_graph.pdf) | [frontier_comparison_20260911_graph](figures/frontier_comparison_20260911_graph.json) | [build_frontier_paper_comparison.py](../ops/build_frontier_paper_comparison.py) |
| 38 | [gpt56_temperature_curve](figures/gpt56_temperature_curve.pdf) | [gpt56_temperature_curve](figures/gpt56_temperature_curve.json) | [plot_paper_gpt56_temperature_curve.py](../ops/plot_paper_gpt56_temperature_curve.py) |
| 39 | [hosted_verified_breadth](figures/hosted_verified_breadth.pdf) | [hosted_verified_breadth](figures/hosted_verified_breadth.json) | [plot_paper_hosted_breadth.py](../ops/plot_paper_hosted_breadth.py) |
| 40 | [reference_kl_knee](figures/reference_kl_knee.pdf) | [reference_kl_comparison](results/reference_kl_comparison.json) | [plot_paper_reference_kl_knee.py](../ops/plot_paper_reference_kl_knee.py) |

## Source and build details

Figures 12 and 40 both use [reference_kl_comparison.json](results/reference_kl_comparison.json). The domain-level comparison includes six coefficients through beta=0.3. Figure 12 averages Graph, MathIR, and Pantry across all six coefficients through beta=0.3. The reference-KL [builder](../ops/build_reference_kl_comparison.py) provides the table, macros, and plot inputs.

The training-curve records specify fixed seed cohorts and per-checkpoint observed subsets. Rings mark incomplete observed cohorts; the observed seed ranges are descriptive. Missing checkpoints are not interpolated. Plot records and manuscript captions identify the relevant eligibility rules and uncertainty estimators for other figures.

The linked rendering scripts identify their analysis inputs and available command-line arguments. Scripts that also analyze raw records require those recorded inputs; the self-contained Overleaf package compiles the supplied assets directly. A LaTeX-only build from `paper/` is:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The [Overleaf packager](../ops/package_iclr_overleaf.py) discovers the files actually read by LaTeX, includes available figure JSON sidecars, records hashes, and checks an independent compile. [Main-length validation](../ops/check_paper_main_length.py) checks the nine-page main text and its twelve figure placements. The manuscript and generated numerical assets determine the figure sequence; this inventory does not change measurements.
