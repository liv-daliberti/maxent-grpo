# Domain-name typography throughout the paper

[Updated PDF](exports/main.pdf) · [Updated Overleaf ZIP](exports/iclr2027_overleaf.zip) · [Final checks](validation.json)

Applied typewriter formatting to benchmark-domain names throughout the main text and appendix: prose, headings, contents entries, captions, tables and embedded figure labels. This includes Graph, Countdown, Python, MathIR and PantryPlan, plus the existing Pantry/Countd. abbreviations and PythonFactors spelling. Domain names in TeX use `\texttt{}`; Matplotlib labels use the corresponding monospace style.

Ordinary references to the Python programming language remain in the normal prose font. The final PDF has seven such occurrences, covering executable code, interpreter settings, invalid expressions, LaTeX-escaped code, lambda conversion and modulo syntax. Literal prompts, code, identifiers, paths, references and bibliography keys are preserved.

## Durable rendering

- `ops/paper_domain_typography.py` formats TeX at emitter boundaries. It is idempotent and protects literal/code regions, existing texttt spans, identifiers, paths and citations. Maintained table/text emitters use it, so regeneration retains the formatting.
- `ops/paper_domain_figure_typography.py` and the plot renderers format only domain labels, preserving surrounding styles, data, scales and measurements. Mixed labels format the domain word separately.
- Two table layouts needed small adjustments for wider labels. The cross-scale comparison now splits Countdown and PantryPlan headers over two lines, matching the main table; the semantic table uses 2.75pt column padding instead of 3pt. No measured cell values changed.

## Verification

- [Source audit](combined_source_audit.md): all **54 active TeX inputs** have complete, idempotent domain formatting, with no nested texttt or missed name variants. **826 tabular rows** retain their measured fields.
- [Figure audit](figures/README.md): **29 figures** updated. All **166 domain occurrences across 40 figure PDFs** are monospace, including four rotated labels checked at character level. The modified figures retain their visible numeric labels; renderer/output bindings are checked.
- [Hosted figure checks](hosted_figures/README.md) also compare plotted arrays and axes, and [hosted text checks](hosted/README.md) verify renderers against stored inputs. The shortened protocol paragraph is synchronized with its emitter. Its downstream [digest updates](hosted/metadata_cascade_validation.json) change metadata only.
- [Whole-PDF font audit](pdf_font_audit.json): every extracted benchmark name uses a monospace font. The only proportional-font matches are the seven ordinary Python-language references described above. Rotated figure labels are covered by the separate figure audit.
- Representative main-text pages, dense tables and figures were inspected visually. The final build has **122 pages**, **nine scientific main pages**, twelve main figures and all **89 numbered appendix headings** represented in the contents. There are no unresolved references/citations, duplicate labels/destinations, overfull boxes or figure collisions.
- The Overleaf archive compiles independently. Its 141 source/asset files and reference PDF match the release snapshot, and 108 compiler-recorded input hashes match. A final update refreshed two non-rendered JSON sidecars and their manifest hashes; all compilation inputs remain byte-identical to those independently compiled.
- No new training, model responses, metric estimation or bootstrap runs were performed for typography.

## Concurrent work

Other appendix prose/contents cleanup and eight replicate-label changes occurred concurrently and were preserved, as documented by the source audit. They are separate from this typography change. Later figure writes remain in live sources; the exports retain the font-verified figures captured at release staging. The final live-file audit found no typography regression: the reference-KL coefficient figure retains its five monospace domain labels; the other two figures have no domain labels. The exact release/live differences and all deliverable hashes are recorded in `validation.json`.
