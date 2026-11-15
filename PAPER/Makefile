# Build the manuscript and the supplement (pdflatex + bibtex).  `make figures` regenerates every figure and table
# from the repository data; `make bib` regenerates refs.bib from DOIs.
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
FIGDIR = figures
FIGSCRIPTS = make_fig_validation.py make_fig_pareto.py make_fig_champions.py make_fig_binaries.py \
             make_table_td.py make_fig_si_robustness.py make_table_literature.py

all: main.pdf supplement.pdf

main.pdf: main.tex refs.bib $(FIGDIR)/table_td.tex
	$(LATEX) main && bibtex main && $(LATEX) main && $(LATEX) main

supplement.pdf: supplement.tex refs.bib $(FIGDIR)/table_literature.tex
	$(LATEX) supplement && bibtex supplement && $(LATEX) supplement && $(LATEX) supplement

figures:
	cd $(FIGDIR) && for s in $(FIGSCRIPTS); do python $$s || exit 1; done

bib:
	python make_bib.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *Notes.bib

.PHONY: all figures bib clean
