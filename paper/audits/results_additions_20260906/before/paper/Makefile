PAPER = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
# The prompt appendix is regenerated from the frozen Hugging Face splits, so it
# needs the datasets-capable interpreter rather than the system python.
DATA_PYTHON = ../var/seed_paper_eval/paper310/bin/python
FIGURES = \
	figures/modecollapse_story.pdf \
	figures/baseline_collapse_precheck.pdf \
	figures/modebench_examples.pdf \
	figures/verified_support_story.pdf \
	figures/experiment1_retention_comparator_matrix.pdf \
	figures/e118_all_scale_factorial_progress.pdf \
	figures/modebench_level_admission.pdf \
	figures/e118_scale_extensions_appendix.pdf \
	figures/sustained_auc_effects_qwen05b.pdf \
	figures/direct_comparator_endpoint_effects.pdf \
	figures/direct_baseline_learning_curves_static_strip.pdf \
	figures/verified_support_discovery_two_scale_effects.pdf \
	figures/replay_mechanism_telemetry_qwen05b.pdf
COMPARISON_JSON = \
	figures/baseline_collapse_precheck.json \
	figures/experiment1_retention_comparator_matrix.json \
	figures/e118_all_scale_factorial_progress.json \
	figures/modebench_level_admission.json \
	figures/sustained_auc_effects_qwen05b.json \
	figures/e118_scale_extensions_appendix.json \
	figures/direct_comparator_endpoint_effects.json \
	figures/direct_baseline_learning_curves_static_strip.json \
	figures/verified_support_discovery_two_scale_effects.json \
	figures/replay_mechanism_telemetry_qwen05b.json

.PHONY: all clean figures prompts terminal-results

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex example_paper.bib iclr2027_conference.sty iclr2027_conference.bst fancyhdr.sty natbib.sty $(FIGURES) $(COMPARISON_JSON) figures/xmode_adaptive_cross_scale_distinct_at_k.json results/e118_qwen05b_factorial.json results/e118_qwen3b_python_terminal_table_body.tex results/xmode_adaptive_cross_scale_endpoints_table_body.tex results/core_terminal_endpoints.json results/core_terminal_endpoints_table_body.tex results/absolute_support_references.json results/absolute_support_references_table_body.tex results/baseline_collapse_precheck.json results/baseline_collapse_precheck_table_body.tex results/e95_falcon_plain_grpo_reportable.json results/e78_terminal_05b_table_body.tex results/maxent_factorial_05b_table_body.tex results/maxent_factorial_05b_uncertainty_table_body.tex results/e87_qwen3b_fixed_semantic_seed70.json results/e87_qwen3b_fixed_semantic_seed70_table_body.tex results/paper_program_status.json results/paper_program_status_table_body.tex results/e113r4_dapo_progress.json results/e78_python_collapse_diagnostics.json results/e120_frequency_progress.json results/e120_frequency_progress_table_body.tex ../var/artifacts/e102_full_open_bank_maxent_replay_05b_audit_latest.json ../ops/check_paper_line_fill.py
	python ../ops/check_paper_current_contract.py
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py
	$(LATEX) $(PAPER)
	bibtex $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
	python ../ops/check_paper_current_contract.py
	python ../ops/check_paper_line_fill.py --pdf $(PAPER).pdf

figures:
	python ../ops/plot_paper_modebench_examples.py
	python ../ops/plot_paper_verified_replay_mechanism.py
	python ../ops/plot_paper_modecollapse.py
	python ../ops/plot_paper_collapse_toy.py
	python ../ops/exp_scaling/plot_e70_clean_05b_wide_live.py

prompts:
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py --emit

terminal-results:
	python ../ops/exp_scaling/build_paper_e118_qwen05b_factorial.py
	python ../ops/exp_scaling/plot_paper_e118_qwen05b_factorial.py
	python ../ops/exp_scaling/plot_paper_e118_all_scale_progress.py
	python ../ops/exp_scaling/plot_paper_modebench_levels.py
	python ../ops/exp_scaling/build_paper_e120_frequency_progress.py
	python ../ops/exp_scaling/build_paper_e78_terminal_results.py
	python ../ops/exp_scaling/build_paper_e78_python_collapse_diagnostics.py
	python ../ops/exp_scaling/build_paper_core_terminal_endpoints.py
	python ../ops/exp_scaling/build_paper_absolute_support_references.py
	python ../ops/exp_scaling/build_paper_baseline_collapse_precheck.py
	python ../ops/exp_scaling/plot_paper_baseline_collapse_precheck.py
	python ../ops/exp_scaling/build_paper_e95_plain_grpo_results.py
	python ../ops/exp_scaling/build_paper_maxent_factorial_results.py
	python ../ops/exp_scaling/build_paper_e87_qwen3b_fixed_semantic_results.py
	python ../ops/exp_scaling/build_paper_program_status.py
	python ../ops/exp_scaling/build_paper_e113r4_dapo_progress.py
	python ../ops/exp_scaling/plot_paper_inference_frontier.py
	python ../ops/exp_scaling/plot_paper_inference_pass_at_k.py
	python ../ops/exp_scaling/plot_paper_fixed_semantic_factorial_effects.py
	python ../ops/exp_scaling/plot_paper_qwen3b_endpoint_progress.py
	python ../ops/exp_scaling/plot_paper_sustained_auc_effects.py
	python ../ops/exp_scaling/plot_e72_decoding_frontier.py
	python ../ops/exp_scaling/plot_paper_comparison_families.py --evidence terminal --output-dir figures/comparisons
	python ../ops/exp_scaling/plot_paper_replay_mechanism_telemetry.py
	python ../ops/exp_scaling/plot_paper_bank_occupancy_outcomes.py
	python ../ops/exp_scaling/plot_paper_adaptive_dose_gate.py
	python ../ops/exp_scaling/plot_paper_adaptive_mechanism_outcomes.py
	python ../ops/exp_scaling/plot_paper_replay_dose_progress.py
	python ../ops/exp_scaling/plot_paper_aligned_domain_strips.py
	python ../ops/exp_scaling/plot_paper_direct_comparator_endpoint_effects.py
	python ../ops/exp_scaling/plot_paper_cross_scale_endpoint_effects.py
	python ../ops/exp_scaling/plot_paper_experiment1_composite.py
	python ../ops/exp_scaling/audit_e102_full_open_bank_maxent_replay.py
	python ../ops/exp_scaling/plot_paper_open_bank_bundle_effects.py
	python ../ops/exp_scaling/plot_paper_starvation_fallback_isolation.py
	python ../ops/exp_scaling/plot_paper_admission_retention_funnel.py
	python ../ops/exp_scaling/plot_e106_group_centered_mechanism_diagnostic.py
	python ../ops/exp_scaling/plot_e111_verified_support_mechanism_diagnostic.py
	python ../ops/exp_scaling/build_e112r1_two_scale_exploratory_results.py
	python ../ops/exp_scaling/plot_e112r1_two_scale_exploratory_effects.py

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
