LATEX = pdflatex -interaction=nonstopmode -halt-on-error
SOURCES = main.tex preamble.tex appendix.tex example_paper.bib $(wildcard *.sty figures/*.pdf results/*.tex)

.PHONY: all check bundle camera-ready clean
all: main.pdf check

main.pdf: $(SOURCES)
	$(LATEX) main.tex
	bibtex main
	$(LATEX) main.tex
	$(LATEX) main.tex

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
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
	rm -f camera-ready.aux camera-ready.bbl camera-ready.blg camera-ready.log camera-ready.out camera-ready.pdf
