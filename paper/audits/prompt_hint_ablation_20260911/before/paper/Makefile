PAPER = main
PYTHON ?= python
# A normal rebuild uses retained evidence. Live collection is a separate target.
ANALYSIS_DATE ?= 2026-09-11
ANALYSIS_STAMP = $(subst -,,$(ANALYSIS_DATE))
AUDIT_DIR ?= audits/results_refresh_$(ANALYSIS_STAMP)
ENDPOINT_AUDIT ?= $(AUDIT_DIR)/latest_endpoints.json
LEVEL_SNAPSHOT ?= results/modebench_level_comparison_snapshot.json
TRAINING_SNAPSHOT ?= results/training_curve_snapshot_$(ANALYSIS_STAMP).json
SYNC_AUDIT_DIR ?= audits/streamlining_$(ANALYSIS_STAMP)/workshop_sync
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
	figures/direct_comparator_endpoint_effects.pdf \
	figures/direct_baseline_learning_curves_static_strip.pdf \
	figures/factorial_training_curves_pass8.pdf \
	figures/factorial_training_curves_distinct8.pdf \
	figures/level2_factorial_training_curves.pdf \
	figures/direct_baseline_learning_curves_pass8.pdf \
	figures/e121_fixed_bank_survival.pdf \
	figures/frontier_comparison_20260911_graph.pdf \
	figures/hosted_verified_breadth.pdf \
	figures/gpt56_temperature_curve.pdf
COMPARISON_JSON = \
	figures/baseline_collapse_precheck.json \
	figures/experiment1_retention_comparator_matrix.json \
	figures/e118_all_scale_factorial_progress.json \
	figures/modebench_level_admission.json \
	figures/e118_scale_extensions_appendix.json \
	figures/direct_comparator_endpoint_effects.json \
	figures/direct_baseline_learning_curves_static_strip.json \
	figures/factorial_training_curves_pass8.json \
	figures/factorial_training_curves_distinct8.json \
	figures/level2_factorial_training_curves.json \
	figures/direct_baseline_learning_curves_pass8.json \
	figures/frontier_comparison_20260911_graph.json \
	figures/hosted_verified_breadth.json \
	figures/gpt56_temperature_curve.json

.PHONY: all clean figures figures-main figures-supporting prompts terminal-results primary-breadth latest-results collect-results sync-workshop workshop both

all: $(PAPER).pdf

RESULTS = results/core_terminal_endpoints.json results/core_terminal_endpoints_table_body.tex \
	results/absolute_support_references.json results/absolute_support_references_table_body.tex \
	results/baseline_collapse_precheck.json results/baseline_collapse_precheck_table_body.tex \
	results/e120_frequency_progress.json results/e120_primary_breadth.json \
	results/e120_primary_breadth_table_body.tex results/e120_primary_breadth_seeds_table_body.tex \
	results/e121_fixed_bank_survival.json results/e78_python_collapse_diagnostics.json \
	results/latest_results_$(ANALYSIS_STAMP).json $(LEVEL_SNAPSHOT) $(TRAINING_SNAPSHOT) \
	$(wildcard results/current_campaign_results_$(ANALYSIS_STAMP)*.json results/current_campaign_results_$(ANALYSIS_STAMP)*.tex) \
	$(wildcard results/level2_factorial_contrasts_$(ANALYSIS_STAMP)*.json results/level2_factorial_contrasts_$(ANALYSIS_STAMP)*.tex) \
	$(wildcard results/frontier_hosted_*.tex results/frontier_hosted_*.json results/frontier_comparison_*.tex results/frontier_comparison_*.json results/frontier_temperature_*.tex results/frontier_temperature_*.json results/frontier_python_retry_*.tex results/frontier_python_retry_*.json results/gpt56_temperature_curve_*.tex)
ANALYSIS_SCRIPTS = ../ops/analyze_gpt56_pass8_frontier.py ../ops/plot_paper_gpt56_temperature_curve.py ../ops/build_frontier_temperature_paper.py ../ops/build_frontier_retry_paper.py ../ops/plot_paper_hosted_breadth.py ../ops/check_paper_main_length.py ../ops/check_paper_current_contract.py ../ops/check_paper_line_fill.py \
	../ops/exp_scaling/build_paper_training_curve_snapshot.py ../ops/exp_scaling/plot_paper_training_curves.py \
	../ops/exp_scaling/build_paper_modebench_level_comparison.py ../ops/exp_scaling/build_paper_e120_primary_breadth.py \
	../ops/exp_scaling/build_paper_e121_survival.py ../ops/exp_scaling/build_paper_current_campaign_results.py \
	../ops/exp_scaling/build_paper_level2_factorial_contrasts.py

$(PAPER).pdf: $(PAPER).tex example_paper.bib iclr2027_conference.sty iclr2027_conference.bst fancyhdr.sty natbib.sty $(FIGURES) $(COMPARISON_JSON) $(RESULTS) $(ANALYSIS_SCRIPTS)
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

# Retained training figures; the hosted comparison uses its dedicated exporter.
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
	$(PYTHON) ../ops/plot_paper_hosted_breadth.py
	$(PYTHON) ../ops/plot_paper_gpt56_temperature_curve.py

# Rerender retained numerical records; do not recollect legacy campaigns.
figures-supporting:
	$(PYTHON) ../ops/exp_scaling/render_paper_retained_figures.py
	$(PYTHON) ../ops/exp_scaling/plot_paper_training_curves.py --snapshot $(TRAINING_SNAPSHOT)

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
