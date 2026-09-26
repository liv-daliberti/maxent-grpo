# Presentation repairs — 2026-09-21

The ICLR manuscript now uses the title “Measuring and Mitigating Reasoning Mode Collapse in RLVR” consistently in its source, PDF metadata, and export README.

## Requested repairs

- Corrected Dr.GRPO citations to Liu et al. and clarified that replay does not increase the fresh-rollout sampling budget.
- Placed Table 1’s untrained checkpoint below the trained methods as a distinct reference, preserving every numerical cell and updating the generator.
- Moved the former Theorem P.6 to Theorem 3.1, with assumptions and interpretation in Section 3.1; its proof remains in the appendix.
- Promoted the frequency-weighting ablation to Figure 10 / Section 5.2, with its numerical effect, uncertainty, endpoint definition, and inconclusive correctness result.
- Removed constraint-withdrawal material from Section 5.3 and tightened repeated prose to retain nine main pages.
- Restored the omitted appendix contents entries, including Q. The current source already contained Q and valid D.1/former P.9 reference destinations; the completed rebuild resolves them.

## Additional presentation fixes

Replaced undefined paper citation aliases with existing bibliography entries. Model names now refer to the appendix’s deployment/evaluation records rather than nonexistent citation keys. Corrected the KL figure’s four-domain population and limited its accuracy tradeoff to coefficients beyond beta=.01. Identified the mixed-group percentages as Dr.GRPO/Re:Dr.

Removed conflicting redundant list packages and repaired figure placement. Fitted the sampling table to the page and split the cross-model-mixture table into two readable panels. Their generators preserve the same data, captions, and surrounding prose. Made the Overleaf exporter tolerate the legacy encoding of TeX diagnostic output.

Concurrent mathematical appendix revisions arrived during integration and were preserved. Their four new subsection headings are included: all 87 numbered appendix headings appear in the contents list. These mathematical revisions are separate from the presentation repairs recorded here.

## Validation

- Stable-source pdfLaTeX/BibTeX build: 123 total pages, 9 main pages, all 12 main figures, disclosures on page 10, references starting page 11.
- No undefined references/citations, duplicate labels, overfull boxes, or wrapfigure collisions.
- Main-length check passed; existing frequency-weighting statistical tests: 14 passed.
- Table 1 and the repaired appendix fragments match their generators; the presentation changes preserve all numerical cells.
- Inspected changed main pages and the split appendix table visually.

The incoming paper already failed its full legacy editorial contract (stale section/proof requirements) and paragraph-final-line audit. These unrelated gates are not reported as passing; figure-placement inventories were updated for the promoted ablation.

Validated main.tex SHA-256: `f135dafeff9e887450a535de2a836a885b0806175fd3af11b68a26c275f763a0`.

Overleaf export: standalone compilation passed. Packaged source and reference PDF match the installed manuscript. Full and main-only review PDFs were refreshed. A concurrent PDF rebuild had identical extracted content; the installed copies use the validated bundle bytes.
