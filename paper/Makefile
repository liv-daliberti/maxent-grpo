PAPER = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
# The prompt appendix is regenerated from the frozen Hugging Face splits, so it
# needs the datasets-capable interpreter rather than the system python.
DATA_PYTHON = ../var/seed_paper_eval/paper310/bin/python
FIGURES = \
	figures/modebench_examples.pdf \
	figures/modebench_pantry_example.pdf \
	figures/xdr_mechanism.pdf \
	figures/modecollapse_story.pdf \
	figures/modes_per_epoch.pdf \
	figures/modecollapse_training.pdf \
	figures/e68_e58_vs_grpo_05b_12ep_terminal_provenance.pdf

.PHONY: all clean figures prompts

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex example_paper.bib iclr2026_conference.sty iclr2026_conference.bst $(FIGURES)
	python ../ops/check_paper_figure1_contract.py
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py
	$(LATEX) $(PAPER)
	bibtex $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
	python ../ops/check_paper_figure1_contract.py

figures:
	python ../ops/plot_paper_modebench_examples.py
	python ../ops/plot_paper_xdr_mechanism.py
	python ../ops/plot_paper_modecollapse.py
	python ../ops/plot_paper_collapse_toy.py
	python ../ops/exp_scaling/plot_e70_clean_05b_wide_live.py

prompts:
	$(DATA_PYTHON) ../ops/check_paper_domain_prompts.py --emit

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
