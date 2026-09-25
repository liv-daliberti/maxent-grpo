# Latest reference-KL measurements: scientific and presentation audit

The replacement in `replacement.tex` covers the two measured-reference-KL subsections, retains their existing labels, and leaves all numbered theorem/equation statements outside that block unchanged. It replaces approximately 2,141 words with 1,502 words. `main_figure_suggestions.json` supplies exact old/new main-body caption and paragraph text for root integration.

The latest comparison JSON, generated table body, and macros are preserved byte for byte. The six coefficients are 0.001, 0.01, 0.04, 0.1, 0.2, and 0.3. There are 27 measured domain–coefficient cells and 132 contributing seed runs: Python 0.1 has three seeds, MathIR 0.3 has four, and the other 25 cells have five. Missing cells are Python 0.2/0.3 and Pantry 0.3. No model calls, training, bootstrap resampling, or rebuilding of measured-data summaries was performed.

## Corrections

- The table wrapper previously had 14 columns and five coefficient headings while its latest rows contained 15 columns and six coefficients. It now has all 15 columns, the 0.3 heading, six-column grouping, and correctly shifted reference-summary columns 12–14.
- Removed withdrawn-claim narration, references to results still arriving, predicted wins framed as confirmed mechanisms, unsupported advice to select one method, and manuscript-history language. Captions begin with a supported result and describe population, estimand, encodings, and missing uncertainty.
- Domainwise PCMD is not monotone in beta: Graph and MathIR decline slightly from 0.2 to 0.3, and Python declines from 0.04 to 0.1. No seed-level uncertainty estimates were inferred from the means.
- Initial-reference PCMD is the conditional stationary value in the categorical result, not a transient or neural upper bound. The measured Graph, MathIR, and Pantry 0.2 cells all exceed their corresponding initial PCMD estimates. The previous “91–100 percent of the reference” interpretation was therefore false.
- Beta-star is a threshold for stationary single-draw correctness in the specified promptwise categorical model. Substituting domain-average initial correctness creates a descriptive nonlinear plug-in, not the mean of per-prompt stationary predictions and not a prediction for observed pass@8. MathIR and Countdown lose pass@8 well below their displayed thresholds. The old figure claim that damage follows this boundary in all four domains was unsupported.
- The occupancy source reads `train/canonical_replay_actuator_modes` across logged updates and averages within runs and then seeds. The transform `1-1/mean(k)` is not the mean fixed-bank limit, not a terminal bank-capacity statistic, and not a held-out neural diversity bound. The Jensen statement is retained only for its actual population of nonempty fixed banks. Logged counts can include zero-key updates.
- The previous 0.85 capacity claim was false: capacity 16 gives categorical uniform diversity 0.9375. Pantry's roughly 6.7-key average gives the descriptive transform near 0.85, which cannot establish a capacity explanation for its measured KL advantage.
- Occupancy does not order all replay outcomes: Countdown Re:Max has lower mean replayed-key count than Re:Dr (2.10 versus 2.28) and higher PCMD (0.531 versus 0.497). Python's larger Re:Max count and PCMD remain as measured associations, without an identified discovery cause.
- Python/Pantry have measured KL cells above both replay arms in PCMD and no lower in pass@8; Countdown/MathIR's larger-PCMD KL cells have lower pass@8 than both replay arms; all Graph KL cells are below both replay arms on both metrics. These are point-estimate comparisons, not statistical or equal-compute dominance claims.
- The main reference-KL plane uses Graph, MathIR, and Pantry, not four domains excluding only Python. Its intersection of available coefficients ends at 0.2; all three existing 0.3 domain cells remain in the table and domain-level coefficient figure. The main text's direct 0.04 comparison is PCMD 0.432 versus Re:Dr 0.427 and pass@8 0.614 versus 0.846.
- The shaded plane is described as a piecewise-linear visual guide rather than a measured attainable frontier or confidence region. Its coordinates and colors are unchanged.
- Initial-policy missingness is explicit: reference PCMD requires at least 30 defined prompts. Countdown has no reportable initial PCMD; Python's observed initial successes are zero, which does not establish zero population success probability. Different policies condition PCMD on their own verified-response populations.
- Hosted diversity descriptions remain, with explicit evaluation-population and eligibility differences. They do not establish downstream training performance or a universal preference between reference KL and replay.
- Budget matching is limited to optimizer updates and fresh rollouts. The text explicitly preserves the absence of equal-total-compute training and of baseline reallocation to useful extra updates or fresh samples.

## Files changed outside the replacement block

- `ops/build_reference_kl_comparison.py`: documentation/comments only. Calculation AST is identical after stripping the module docstring. Legacy JSON keys containing `ceiling` remain for compatibility, but their correct descriptive interpretation is documented.
- `ops/plot_paper_reference_kl_plane.py`: corrected comments/docstring, removed duplicate import, changed the initial-policy vertical-rule label from “accuracy” to “pass@8”.
- `ops/plot_paper_reference_kl_knee.py`: corrected comments/docstring and legend labels; removed the claim that initial PCMD is a bound or that beta-star predicts the measured knee. Compact legend spacing and omission of the redundant middle-right beta label remove crowding above the legend; the bottom coefficient axis remains labeled.
- `paper/figures/reference_kl_plane.{pdf,png}` and `paper/figures/reference_kl_knee.{pdf,png}`: regenerated only from the existing latest comparison JSON.

## Validation

`validate.py` records 43 grouped assertions in `validation.json`. These include byte identity of all three numerical inputs, all 70 numeric table entries checked against the latest JSON, all missing/partial cell counts, 132 contributing runs, occupancy transforms, beta-star calculations, population and coefficient intersections, the three-domain main-text means, all domainwise comparison claims, and table-wrapper dimensions.

Both plot scripts were executed with their save functions replaced by figure captures before and after the edits. All data coordinates, line/marker styles, axis limits/positions, annotation positions, and shaded-polygon vertices are identical. Legend text/spacing and the redundant axis label are the only rendered differences beyond the corrected vertical-rule label. No old figure or numerical input was substituted. The two generated PNGs were visually inspected; the main plane has clear labels and the knee figure has a readable compact key.

The root agent is responsible for integrating the replacement and exact main-body suggestions, mirroring TOC headings, and full PDF/Overleaf/layout validation.
