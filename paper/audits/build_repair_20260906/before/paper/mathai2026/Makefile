LATEX = pdflatex -interaction=nonstopmode -halt-on-error -recorder
REQUIRED_TABLES = results/core_terminal_endpoints_table_body.tex \
	results/absolute_support_references_table_body.tex \
	results/baseline_collapse_precheck_table_body.tex \
	results/e118_qwen3b_python_terminal_table_body.tex \
	results/e120_frequency_progress_table_body.tex \
	results/e120_primary_breadth_table_body.tex \
	results/e120_primary_breadth_seeds_table_body.tex
SOURCES = main.tex preamble.tex appendix.tex example_paper.bib $(REQUIRED_TABLES) \
	$(wildcard *.sty figures/*.pdf figures/*.json results/*.json) \
	snapshot.json check_submission.py Makefile

.DELETE_ON_ERROR:
.PHONY: all check bundle camera-ready clean
all: main.pdf check

main.pdf: $(SOURCES)
	$(LATEX) main.tex
	bibtex main
	$(LATEX) main.tex
	$(LATEX) main.tex
	python3 check_submission.py --record-build

check: main.pdf
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
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.fls build_receipt.json
	rm -f camera-ready.aux camera-ready.bbl camera-ready.blg camera-ready.log camera-ready.out camera-ready.pdf camera-ready.fls
