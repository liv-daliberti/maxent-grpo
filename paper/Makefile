PAPER = main
PYTHON ?= python
# A normal rebuild uses retained evidence. Live collection is a separate target.
ANALYSIS_DATE ?= 2026-09-12
ANALYSIS_STAMP = $(subst -,,$(ANALYSIS_DATE))
AUDIT_DIR ?= audits/results_completion_$(ANALYSIS_STAMP)
ENDPOINT_AUDIT ?= $(AUDIT_DIR)/latest_endpoints.json
LEVEL_SNAPSHOT ?= results/modebench_level_comparison_snapshot.json
TRAINING_SNAPSHOT ?= results/training_curve_snapshot_$(ANALYSIS_STAMP).json
SYNC_AUDIT_DIR ?= audits/streamlining_$(ANALYSIS_STAMP)/workshop_sync
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
# The prompt appendix is regenerated from the frozen Hugging Face splits, so it
# needs the datasets-capable interpreter rather than the system python.
# Use the same interpreter for numerical contracts: Python versions can differ
# in the last bit of statistics.stdev, while retained JSON is compared exactly.
DATA_PYTHON = ../var/seed_paper_eval/paper310/bin/python
FIGURES = \
	figures/concentration_story.pdf \
	figures/replay_factorial_effects.pdf \
	figures/replay_key_weighting.pdf \
	figures/replay_bank_balance.pdf \
	figures/replay_bank_decomposition.pdf \
	figures/frontier_level_grid.pdf \
	figures/replay_level2_effects.pdf \
	figures/modecollapse_story.pdf \
	figures/baseline_collapse_precheck.pdf \
	figures/modebench_examples.pdf \
	figures/modebench_level_construction.pdf \
	figures/modebench_base_levels_appendix.pdf \
	figures/verified_support_story.pdf \
	figures/experiment1_retention_comparator_matrix.pdf \
	figures/e118_all_scale_factorial_progress.pdf \
	figures/modebench_level_admission.pdf \
	figures/e118_scale_extensions_appendix.pdf \
	figures/direct_comparator_endpoint_effects.pdf \
	figures/direct_baseline_learning_curves_static_strip.pdf \
	figures/factorial_training_curves_pass8.pdf \
	figures/factorial_training_curves_distinct8.pdf \
	figures/level2_factorial_training_curves.pdf \
	figures/direct_baseline_learning_curves_pass8.pdf \
	figures/e121_fixed_bank_survival.pdf \
	figures/frontier_comparison_20260911_graph.pdf \
	figures/hosted_verified_breadth.pdf \
	figures/gpt56_all_levels32_sampling_budget.pdf \
	figures/gpt56_temperature_curve.pdf
COMPARISON_JSON = \
	figures/concentration_story.json \
	figures/replay_factorial_effects.json \
	figures/replay_key_weighting.json \
	figures/replay_level2_effects.json \
	figures/baseline_collapse_precheck.json \
	figures/experiment1_retention_comparator_matrix.json \
	figures/e118_all_scale_factorial_progress.json \
	figures/modebench_level_admission.json \
	figures/modebench_level_construction.json \
	figures/modebench_base_levels_appendix.json \
	figures/e118_scale_extensions_appendix.json \
	figures/direct_comparator_endpoint_effects.json \
	figures/direct_baseline_learning_curves_static_strip.json \
	figures/factorial_training_curves_pass8.json \
	figures/factorial_training_curves_distinct8.json \
	figures/level2_factorial_training_curves.json \
	figures/direct_baseline_learning_curves_pass8.json \
	figures/frontier_comparison_20260911_graph.json \
	figures/hosted_verified_breadth.json \
	figures/gpt56_all_levels32_sampling_budget.json \
	figures/gpt56_temperature_curve.json

.PHONY: all clean prompt-ablation discovery-curves figures figures-main figures-supporting prompts terminal-results primary-breadth latest-results collect-results sync-workshop workshop both

all: $(PAPER).pdf

.PHONY: inference-followups
inference-followups:
	$(DATA_PYTHON) ../ops/build_paper_inference_followups.py --check

HOSTED_LEVEL_SUMMARY = results/hosted_level_averages_20260911.tex results/hosted_level_averages_20260911.json
HOSTED_REASONING_SOURCE = ../artifacts/hosted_reasoning_off_32_20260911_v2/reasoning_comparison.json
HOSTED_REASONING_ASSETS = results/hosted_reasoning_off_20260912.json results/hosted_reasoning_off_20260912_main.tex results/hosted_reasoning_off_20260912_appendix.tex


INFERENCE_FOLLOWUPS = results/inference_followups_20260912.tex results/inference_followups_20260912.json figures/pantry_adaptation_recovery_20260912.pdf figures/pantry_adaptation_recovery_20260912.png

# Publication targets require a complete authenticated report for this fixed study.
# They consume retained evidence and never launch model requests.
GPT56_DISCOVERY_ROOT = ../artifacts/modebench_discovery_all_levels_sol32x512_20260913
GPT56_DISCOVERY_PRIOR_ROOT = ../artifacts/modebench_discovery_all_levels_sol512_20260912
GPT56_DISCOVERY_SOURCE = $(GPT56_DISCOVERY_ROOT)/analysis/analysis.json
GPT56_DISCOVERY_EVIDENCE = $(GPT56_DISCOVERY_SOURCE) \
	$(GPT56_DISCOVERY_ROOT)/support_reference.json $(GPT56_DISCOVERY_ROOT)/support_certificate_manifest.json \
	$(GPT56_DISCOVERY_ROOT)/manifest.json $(GPT56_DISCOVERY_ROOT)/protocol.json \
	$(GPT56_DISCOVERY_ROOT)/selection.json $(GPT56_DISCOVERY_ROOT)/rows.jsonl \
	$(GPT56_DISCOVERY_ROOT)/new_rows.jsonl $(GPT56_DISCOVERY_ROOT)/retained_rows.jsonl \
	$(GPT56_DISCOVERY_ROOT)/level1_rows.jsonl $(GPT56_DISCOVERY_ROOT)/retained_prior_runs.json \
	$(GPT56_DISCOVERY_ROOT)/collection_completion_audit.json \
	$(GPT56_DISCOVERY_PRIOR_ROOT)/analysis/analysis.json $(GPT56_DISCOVERY_PRIOR_ROOT)/support_reference.json \
	preregistration/gpt56sol_generalization32x512_20260913.md
GPT56_DISCOVERY_SCRIPTS = ../ops/build_paper_gpt56_all_levels32_discovery.py \
	../ops/plot_paper_gpt56_all_levels32_sampling.py ../ops/analyze_gpt56_all_levels32_discovery.py \
	../ops/gpt56_all_levels32_discovery_support.py ../ops/prepare_gpt56_all_levels32_discovery.py \
	../ops/audit_gpt56_all_levels32_collection.py \
	$(GPT56_DISCOVERY_ROOT)/prepare_gpt56_all_levels32_discovery.py \
	$(GPT56_DISCOVERY_ROOT)/support_code/gpt56_all_levels32_discovery_support.py \
	$(GPT56_DISCOVERY_ROOT)/support_code/gpt56_all_levels_discovery_support.py \
	$(GPT56_DISCOVERY_ROOT)/support_code/gpt56_discovery_support.py \
	../ops/analyze_gpt56_all_levels_discovery.py ../ops/prepare_gpt56_all_levels512_discovery.py \
	../ops/prepare_gpt56_five_domain_discovery.py \
	$(GPT56_DISCOVERY_PRIOR_ROOT)/support_code/gpt56_all_levels_discovery_support.py \
	../artifacts/modebench_discovery_five_domains_sol_20260912/final_analysis_code/ops/analyze_gpt56_all_domain_discovery.py
GPT56_DISCOVERY_RESULTS = results/gpt56_all_levels32_sampling_20260913.tex results/gpt56_all_levels32_sampling_20260913.json

RESULTS = $(GPT56_DISCOVERY_RESULTS) results/discovery_hosted_interpretation_20260912.tex results/discovery_hosted_interpretation_20260912.json $(HOSTED_REASONING_ASSETS) results/semantic_current_summary_20260912.tex results/semantic_current_summary_20260912.json $(INFERENCE_FOLLOWUPS) $(HOSTED_LEVEL_SUMMARY) $(wildcard results/conditional_concentration_20260912*.tex results/conditional_concentration_20260912.json) \
	results/core_terminal_endpoints.json results/core_terminal_endpoints_table_body.tex \
	results/mode_diversity_retention_matrix.json results/pmd_retention_macros.tex \
	results/domain_support.json results/domain_support_macros.tex \
	results/retention_matrix_panel_a_table_body.tex results/retention_matrix_panel_b_table_body.tex \
	results/absolute_support_references.json results/absolute_support_references_table_body.tex \
	results/baseline_collapse_precheck.json results/baseline_collapse_precheck_table_body.tex \
	results/e120_frequency_progress.json results/e120_primary_breadth.json \
	results/e120_primary_breadth_table_body.tex results/e120_primary_breadth_seeds_table_body.tex \
	results/e121_fixed_bank_survival.json results/e78_python_collapse_diagnostics.json \
	results/latest_results_$(ANALYSIS_STAMP).json $(LEVEL_SNAPSHOT) $(TRAINING_SNAPSHOT) \
	$(wildcard results/current_campaign_results_$(ANALYSIS_STAMP)*.json results/current_campaign_results_$(ANALYSIS_STAMP)*.tex) \
	$(wildcard results/level2_factorial_contrasts_$(ANALYSIS_STAMP)*.json results/level2_factorial_contrasts_$(ANALYSIS_STAMP)*.tex) \
	$(wildcard results/frontier_hosted_*.tex results/frontier_hosted_*.json results/frontier_comparison_*.tex results/frontier_comparison_*.json results/frontier_temperature_*.tex results/frontier_temperature_*.json results/frontier_python_retry_*.tex results/frontier_python_retry_*.json results/gpt56_temperature_curve_*.tex)
PROMPT_ABLATION_ASSETS = $(wildcard results/modebench_prompt_ablation_python_failure_diagnostic_20260911.json results/modebench_prompt_ablation_20260911.json results/modebench_prompt_ablation_20260911.tex figures/modebench_prompt_ablation_*.pdf figures/modebench_prompt_ablation_*.png figures/modebench_prompt_ablation_*.json)
DISCOVERY_CURVE_ASSETS = $(wildcard results/modebench_discovery_curves_20260911.json results/modebench_discovery_curves_20260911.tex figures/modebench_discovery_*.pdf figures/modebench_discovery_*.png figures/modebench_discovery_*.json)
ANALYSIS_SCRIPTS = $(GPT56_DISCOVERY_SCRIPTS) $(GPT56_DISCOVERY_EVIDENCE) ../artifacts/modebench_base_level_grid_20260911/qwen7b_levels123_20260912/figure_source_v1.json ../artifacts/modebench_base_level_grid_20260911/status_updates/20260912T152423Z/figure_source.json ../ops/plot_paper_modebench_base_levels_appendix.py ../ops/plot_paper_modebench_base_grid.py ../artifacts/frontier_temperature_20260911/GPT56_TEMPERATURE_CURVE_EXPANDED480.json ../artifacts/frontier_temperature_20260911/GPT56_PASS8_FRONTIER_EXPANDED480.json ../artifacts/frontier_temperature_20260911/gpt56_expanded480_analysis_code_v2/manifest.json ../artifacts/frontier_temperature_20260911/gpt56_expanded480_analysis_code_v2/ops/analyze_gpt56_temperature_curve_expanded480.py ../artifacts/frontier_temperature_20260911/gpt56_expanded480_analysis_code_v2/ops/analyze_gpt56_pass8_frontier_expanded480.py ../ops/plot_paper_concentration_story.py ../ops/plot_paper_modebench_level_construction.py ../ops/plot_paper_reorganized_results.py ../ops/check_paper_discovery_curves.py ../ops/check_paper_prompt_ablation.py ../ops/analyze_gpt56_pass8_frontier.py ../ops/analyze_gpt56_pass8_frontier_with_zero.py ../ops/plot_paper_gpt56_temperature_curve.py ../ops/build_frontier_temperature_paper.py ../ops/build_frontier_retry_paper.py ../ops/plot_paper_hosted_breadth.py ../ops/check_paper_main_length.py ../ops/check_paper_current_contract.py ../ops/check_paper_line_fill.py \
	../ops/exp_scaling/build_paper_training_curve_snapshot.py ../ops/exp_scaling/plot_paper_training_curves.py \
	../ops/exp_scaling/build_paper_modebench_level_comparison.py ../ops/exp_scaling/build_paper_e120_primary_breadth.py \
	../ops/exp_scaling/build_paper_e121_survival.py ../ops/exp_scaling/build_paper_current_campaign_results.py \
	../ops/exp_scaling/build_paper_level2_factorial_contrasts.py \
	../ops/build_pmd_retention_matrix.py ../ops/build_paper_retention_matrix_tables.py

$(HOSTED_LEVEL_SUMMARY) &: ../ops/build_paper_hosted_level_averages.py ../ops/plot_paper_hosted_breadth.py results/frontier_comparison_20260911.json $(wildcard icons/*.png)
	$(PYTHON) ../ops/build_paper_hosted_level_averages.py

$(HOSTED_REASONING_ASSETS) &: ../ops/build_paper_hosted_reasoning_off.py $(HOSTED_REASONING_SOURCE) results/frontier_comparison_20260911.json $(wildcard ../artifacts/hosted_reasoning_off_32_20260911_v2/*/reasoning_comparison.json) ../ops/summarize_hosted_reasoning_off.py $(wildcard icons/*.png)
	$(DATA_PYTHON) ../ops/build_paper_hosted_reasoning_off.py

$(GPT56_DISCOVERY_RESULTS) &: ../ops/build_paper_gpt56_all_levels32_discovery.py ../ops/plot_paper_gpt56_all_levels32_sampling.py $(GPT56_DISCOVERY_SOURCE)
	$(DATA_PYTHON) ../ops/build_paper_gpt56_all_levels32_discovery.py

figures/gpt56_all_levels32_sampling_budget.pdf figures/gpt56_all_levels32_sampling_budget.png figures/gpt56_all_levels32_sampling_budget.json &: ../ops/plot_paper_gpt56_all_levels32_sampling.py $(GPT56_DISCOVERY_SOURCE)
	$(DATA_PYTHON) ../ops/plot_paper_gpt56_all_levels32_sampling.py

figures/replay_bank_balance.pdf figures/replay_bank_balance.png figures/replay_bank_balance.json &: ../ops/plot_paper_replay_bank_balance.py ../ops/paper_style.py
	$(DATA_PYTHON) ../ops/plot_paper_replay_bank_balance.py

results/domain_support.json results/domain_support_macros.tex &: ../ops/build_paper_domain_support_macros.py
	$(DATA_PYTHON) ../ops/build_paper_domain_support_macros.py

figures/frontier_level_grid.pdf figures/frontier_level_grid.png figures/frontier_level_grid.json &: ../ops/plot_paper_frontier_level_grid.py ../ops/plot_paper_mode_diversity_levels.py ../ops/paper_style.py results/mode_diversity_frontier_points.json
	$(DATA_PYTHON) ../ops/plot_paper_frontier_level_grid.py

figures/replay_bank_decomposition.pdf figures/replay_bank_decomposition.png figures/replay_bank_decomposition.json results/replay_bank_decomposition_macros.tex &: ../ops/plot_paper_replay_bank_decomposition.py ../ops/paper_style.py results/replay_bank_decomposition_measured.json
	$(DATA_PYTHON) ../ops/plot_paper_replay_bank_decomposition.py

$(PAPER).pdf: $(PAPER).tex example_paper.bib iclr2027_conference.sty iclr2027_conference.bst fancyhdr.sty natbib.sty $(FIGURES) $(COMPARISON_JSON) $(RESULTS) $(PROMPT_ABLATION_ASSETS) $(DISCOVERY_CURVE_ASSETS) $(ANALYSIS_SCRIPTS) | prompt-ablation discovery-curves inference-followups
	$(DATA_PYTHON) ../ops/check_paper_current_contract.py
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py
	$(LATEX) $(PAPER)
	bibtex $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(DATA_PYTHON) ../ops/check_paper_current_contract.py
	python ../ops/check_paper_line_fill.py --pdf $(PAPER).pdf

# Authenticate completed ablation evidence whenever any publication asset is present.
# This repository-only check precedes synchronization; the workshop stays standalone.
prompt-ablation:
	@if [ -n "$(PROMPT_ABLATION_ASSETS)" ]; then $(DATA_PYTHON) ../ops/check_paper_prompt_ablation.py --paper-dir .; fi

# This separate follow-up retains its own analysis and publication bindings.
discovery-curves:
	@if [ -n "$(DISCOVERY_CURVE_ASSETS)" ]; then $(DATA_PYTHON) ../ops/check_paper_discovery_curves.py --paper-dir .; fi

# Reproduce dated tables/report without consulting changing campaign directories.
latest-results:
	$(PYTHON) ../ops/exp_scaling/build_paper_latest_results.py --date $(ANALYSIS_DATE) --from-audit $(ENDPOINT_AUDIT)
	$(PYTHON) ../ops/exp_scaling/build_paper_current_campaign_results.py --snapshot results/latest_results_$(ANALYSIS_STAMP).json --figures
	$(PYTHON) ../ops/exp_scaling/build_paper_level2_factorial_contrasts.py --date $(ANALYSIS_DATE) --audit $(ENDPOINT_AUDIT)

# Explicit live collection: choose a new date/directory; existing audits survive.
collect-results:
	test ! -e $(ENDPOINT_AUDIT)
	$(PYTHON) ../ops/exp_scaling/build_paper_latest_results.py --date $(ANALYSIS_DATE) --audit-dir $(AUDIT_DIR)
	$(PYTHON) ../ops/exp_scaling/freeze_paper_modebench_levels.py --audit-directory $(AUDIT_DIR)/figure6 --endpoint-audit $(ENDPOINT_AUDIT) --output $(LEVEL_SNAPSHOT)

# The retention matrix reaches the manuscript as two tables. The PCMD deltas
# are differenced first, the composite record binds them, and the table bodies
# are rendered from that record so the printed cells and the machine-readable
# artifact cannot drift apart.
results/mode_diversity_retention_matrix.json results/pmd_retention_macros.tex &: ../ops/build_pmd_retention_matrix.py results/mode_diversity_curves.json results/mode_diversity_comparators_05b.json results/mode_diversity_comparators_rlep.json
	$(PYTHON) ../ops/build_pmd_retention_matrix.py

results/retention_matrix_panel_a_table_body.tex results/retention_matrix_panel_b_table_body.tex &: ../ops/build_paper_retention_matrix_tables.py figures/experiment1_retention_comparator_matrix.json
	$(PYTHON) ../ops/build_paper_retention_matrix_tables.py

# This registered primary analysis continues to use its original frozen input.
primary-breadth:
	$(PYTHON) ../ops/exp_scaling/build_paper_e120_primary_breadth.py

# Re-render scientific figures from retained records; no collection occurs here.
figures: figures-supporting
	$(MAKE) figures-main

figures-main:
	$(PYTHON) ../ops/plot_paper_collapse_toy.py
	$(PYTHON) ../ops/plot_paper_modebench_examples.py
	$(PYTHON) ../ops/plot_paper_modebench_level_construction.py
	$(PYTHON) ../ops/plot_paper_support_story.py
	$(PYTHON) ../ops/exp_scaling/plot_paper_experiment1_composite.py
	$(PYTHON) ../ops/exp_scaling/plot_paper_e118_all_scale_progress.py --endpoint-audit $(ENDPOINT_AUDIT)
	$(PYTHON) ../ops/exp_scaling/plot_paper_modebench_levels.py --snapshot $(LEVEL_SNAPSHOT)
	$(PYTHON) ../ops/plot_paper_hosted_breadth.py
	$(PYTHON) ../ops/plot_paper_gpt56_all_levels32_sampling.py
	$(PYTHON) ../ops/plot_paper_gpt56_temperature_curve.py

# The original main artwork stays in place; additional quantitative views
# accompany the complete source-bound analyses in the supplement.
figures-supporting:
	$(PYTHON) ../ops/plot_paper_modebench_base_levels_appendix.py
	$(PYTHON) ../ops/plot_paper_concentration_story.py
	$(PYTHON) ../ops/plot_paper_reorganized_results.py
	$(PYTHON) ../ops/exp_scaling/render_paper_retained_figures.py
	$(PYTHON) ../ops/exp_scaling/plot_paper_training_curves.py --snapshot $(TRAINING_SNAPSHOT)

prompts:
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py --emit

terminal-results: latest-results
	$(MAKE) figures

# Edit both manuscripts' prose before binding their assets and source hashes.
# The updater preserves every replaced asset and the prior snapshot in this audit.
sync-workshop: prompt-ablation discovery-curves
	$(PYTHON) ../ops/sync_paper_workshop_assets.py --date $(ANALYSIS_DATE) --audit-directory $(abspath $(SYNC_AUDIT_DIR)) --reason "Refresh the compiled figure set and dated result evidence in both manuscripts" --apply

workshop: sync-workshop
	$(MAKE) -C mathai2026 bundle

both: all
	$(MAKE) workshop

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
