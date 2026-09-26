PAPER = main
PYTHON ?= python
# A normal rebuild uses retained evidence. Live collection is a separate target.
ANALYSIS_DATE ?= 2026-09-11
ANALYSIS_STAMP = $(subst -,,$(ANALYSIS_DATE))
AUDIT_DIR ?= audits/results_refresh_$(ANALYSIS_STAMP)
ENDPOINT_AUDIT ?= $(AUDIT_DIR)/latest_endpoints.json
LEVEL_SNAPSHOT ?= results/modebench_level_comparison_snapshot.json
SYNC_AUDIT_DIR ?= audits/figure_refresh_$(ANALYSIS_STAMP)/workshop_sync
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
	figures/replay_mechanism_telemetry_qwen05b.pdf \
	figures/e121_fixed_bank_survival.pdf
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

.PHONY: all clean figures figures-main figures-supporting prompts terminal-results primary-breadth latest-results collect-results sync-workshop workshop both

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex example_paper.bib iclr2027_conference.sty iclr2027_conference.bst fancyhdr.sty natbib.sty $(FIGURES) $(COMPARISON_JSON) figures/xmode_adaptive_cross_scale_distinct_at_k.json results/e118_qwen05b_factorial.json results/modebench_level_comparison_snapshot.json ../ops/exp_scaling/build_paper_modebench_level_comparison.py results/e118_qwen3b_python_terminal_table_body.tex results/xmode_adaptive_cross_scale_endpoints_table_body.tex results/core_terminal_endpoints.json results/core_terminal_endpoints_table_body.tex results/absolute_support_references.json results/absolute_support_references_table_body.tex results/baseline_collapse_precheck.json results/baseline_collapse_precheck_table_body.tex results/e95_falcon_plain_grpo_reportable.json results/e78_terminal_05b_table_body.tex results/maxent_factorial_05b_table_body.tex results/maxent_factorial_05b_uncertainty_table_body.tex results/e87_qwen3b_fixed_semantic_seed70.json results/e87_qwen3b_fixed_semantic_seed70_table_body.tex results/paper_program_status.json results/paper_program_status_table_body.tex results/e113r4_dapo_progress.json results/e78_python_collapse_diagnostics.json results/e120_frequency_progress.json results/e120_frequency_progress_table_body.tex results/e120_primary_breadth.json results/e120_primary_breadth_table_body.tex results/e120_primary_breadth_seeds_table_body.tex ../ops/exp_scaling/build_paper_e120_primary_breadth.py ../ops/check_paper_current_contract.py ../var/artifacts/e102_full_open_bank_maxent_replay_05b_audit_latest.json ../ops/check_paper_line_fill.py results/e121_fixed_bank_survival.json results/e121_fixed_bank_survival_table_body.tex results/e121_fixed_bank_survival_sequence_table_body.tex results/e121_fixed_bank_survival_bootstrap_table_body.tex ../ops/exp_scaling/build_paper_e121_survival.py $(wildcard results/current_campaign_results_*.json results/current_campaign_results_*.tex results/level2_factorial_contrasts_*.json results/level2_factorial_contrasts_*.tex results/latest_results_*.json results/latest_results_*.tex results/e120_falcon_graph_*.json results/e120_falcon_graph_*.tex)
	python ../ops/check_paper_current_contract.py
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py
	$(LATEX) $(PAPER)
	bibtex $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
	python ../ops/check_paper_current_contract.py
	python ../ops/check_paper_line_fill.py --pdf $(PAPER).pdf

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

# This registered primary analysis continues to use its original frozen input.
primary-breadth:
	$(PYTHON) ../ops/exp_scaling/build_paper_e120_primary_breadth.py

# All 14 compiled figure PDFs: 6 main figures and 8 supplementary figures.
# The E118 renderer produces Figure 5 and its per-domain appendix together.
figures: figures-supporting
	$(MAKE) figures-main

figures-main:
	$(PYTHON) ../ops/plot_paper_collapse_toy.py
	$(PYTHON) ../ops/plot_paper_modebench_examples.py
	$(PYTHON) ../ops/plot_paper_support_story.py
	$(PYTHON) ../ops/exp_scaling/plot_paper_e118_all_scale_progress.py --endpoint-audit $(ENDPOINT_AUDIT)
	$(PYTHON) ../ops/exp_scaling/plot_paper_experiment1_composite.py
	$(PYTHON) ../ops/exp_scaling/plot_paper_modebench_levels.py --snapshot $(LEVEL_SNAPSHOT)

# Rerender retained numerical records; do not recollect legacy campaigns.
figures-supporting:
	$(PYTHON) ../ops/exp_scaling/render_paper_retained_figures.py

prompts:
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py --emit

terminal-results: latest-results
	$(MAKE) figures

# Edit both manuscripts' prose before binding their assets and source hashes.
# The updater preserves every replaced asset and the prior snapshot in this audit.
sync-workshop:
	$(PYTHON) ../ops/sync_paper_workshop_assets.py --date $(ANALYSIS_DATE) --audit-directory $(abspath $(SYNC_AUDIT_DIR)) --reason "Refresh the compiled figure set and dated result evidence in both manuscripts" --apply

workshop: sync-workshop
	$(MAKE) -C mathai2026 bundle

both: all
	$(MAKE) workshop

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
