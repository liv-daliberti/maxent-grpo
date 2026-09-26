LATEX = pdflatex -interaction=nonstopmode -halt-on-error -recorder
REQUIRED_TABLES = results/core_terminal_endpoints_table_body.tex \
	results/absolute_support_references_table_body.tex \
	results/baseline_collapse_precheck_table_body.tex \
	results/e118_qwen3b_python_terminal_table_body.tex \
	results/e120_frequency_progress_table_body.tex \
	results/e120_primary_breadth_table_body.tex \
	results/e120_primary_breadth_seeds_table_body.tex \
	results/e121_fixed_bank_survival_table_body.tex \
	results/e121_fixed_bank_survival_sequence_table_body.tex \
	results/e121_fixed_bank_survival_bootstrap_table_body.tex \
	results/latest_results_20260911_table_body.tex \
	results/latest_results_20260911_effects_table_body.tex \
	results/current_campaign_results_20260911_e118_table_body.tex \
	results/current_campaign_results_20260911_e119_table_body.tex \
	results/current_campaign_results_20260911_e120_table_body.tex \
	results/level2_factorial_contrasts_20260911_table_body.tex
SOURCES = main.tex preamble.tex appendix.tex example_paper.bib $(REQUIRED_TABLES) \
	$(wildcard *.sty figures/*.pdf figures/*.json results/*.tex results/*.json) \
	snapshot.json check_submission.py build_submission.py package_source.py Makefile

.PHONY: all check bundle camera-ready clean
all: main.pdf check

main.pdf: $(SOURCES)
	python3 build_submission.py --latex "$(LATEX)"

# A removed receipt requires a fresh compilation, not a hash of an old PDF.
build_receipt.json: main.pdf
	python3 build_submission.py --latex "$(LATEX)"

check: main.pdf build_receipt.json
	python3 check_submission.py

bundle: all
	python3 package_source.py

# Populate the author information in main.tex before using this target.
camera-ready:
	$(LATEX) -jobname=camera-ready '\def\mathaicameraready{1}\input{main.tex}'
	bibtex camera-ready
	$(LATEX) -jobname=camera-ready '\def\mathaicameraready{1}\input{main.tex}'
	$(LATEX) -jobname=camera-ready '\def\mathaicameraready{1}\input{main.tex}'

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.fls main.failed.log build_receipt.json
	rm -f camera-ready.aux camera-ready.bbl camera-ready.blg camera-ready.log camera-ready.out camera-ready.pdf camera-ready.fls
