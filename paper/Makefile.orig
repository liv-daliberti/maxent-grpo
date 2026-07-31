PAPER = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex example_paper.bib neurips_2025.sty
	$(LATEX) $(PAPER)
	bibtex $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
