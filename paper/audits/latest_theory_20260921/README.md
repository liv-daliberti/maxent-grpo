# Latest theory and reference-KL presentation cleanup — 2026-09-21

The current live manuscript, full PDF, main-body PDF, and Overleaf ZIP use the latest theory and measured inputs. No prior KL result set was substituted. This pass covers Appendix P (all fourteen theory subsections), Appendix Q (compute accounting), and the main Figure 12 caption and accompanying comparison. In the final PDF, P starts on page 91, Q on page 116, and the next appendix starts on page 119. The main scientific text remains nine pages with twelve figures; the full paper has 122 pages.

## Preserved latest work

- The latest replay retention, weighted-target and partial-coverage convergence, finite-time restoration, discrete/stochastic energy, shared-exemplar, neural-interference, and reference-KL results remain in place.
- All 102 displayed equations in P.1–P.11 are byte-identical to the latest source at the start of this pass. All 253 manuscript labels and all formal-environment counts are preserved. Statement and proof prose was clarified where needed; no theorem was removed.
- All packaged numerical input files are unchanged. The KL comparison retains six coefficients through beta=0.3, 27 measured cells, the partial seed counts, and all 132 contributing seed runs. All 16 existing categorical recovery times are unchanged.
- The two regenerated KL figures preserve all data coordinates, axis positions/limits, marker styles, annotation positions, and shaded polygon coordinates. Changes are labels, legend spacing, and removal of one redundant axis label.
- No training runs, recovery integrations, model calls, or bootstrap computations were performed.

## Corrections and cleanup

The KL table now has fifteen columns and all six coefficient headings, matching the latest table body. Figure 12 correctly names its three-domain population; the common-coefficient aggregate stops at beta=0.2 while the available beta=0.3 measurements remain in the domain-level table and figure. The displayed point comparisons were checked against the latest data.

The KL discussion separates stationary categorical targets from measured neural trajectories, initial reference diversity from a transient upper bound, and the single-draw stationary beta-star threshold from measured pass@8 changes. It describes actual nonmonotone coefficient responses and the missing/partial cells. Replay occupancy is a descriptive logged-update quantity, not a held-out diversity bound or the capacity limit. The recovery table's initial-depth heading now correctly denotes odds rather than probability.

The proofs retain their assumptions and limits. Retention with bounded signed fresh coefficients remains distinct from convergence requiring nonnegative coefficients. KL estimator value-unbiasedness is distinguished from an exact distributional KL derivative; forward KL differs from cross-entropy by a target-entropy constant. A low fresh-success probability is not equated with an empty preexisting bank.

Rendered revision-history, novelty/insertion narration, withdrawn-claim language, and process commentary were removed from P and Q. Captions start with a supported result, then specify population, setup, encodings, and uncertainty where applicable. Compute accounting explicitly retains the missing equal-total-compute comparison: the exact-zero replay derivative isolates the signal but does not give the baseline additional useful learning. Cost subtotals remain distinguished from total training costs.

Figure 40 was converted from a bottom-overflowing wrapfigure to a regular float. Its panels, caption, and Table 59 fit on page 113. The adjacent table caption was shortened, and a short paragraph ending was repaired. The main Figure 12 caption now includes Level 1. Three manual TOC titles were synchronized with the revised headings.

## Verification

- Replay theory: 7,018 deterministic consistency checks; independent definition/assumption review of P.1–P.5 found no further defect.
- KL theory: 592 checks, including stationary-policy, gradient, boundary, recovery, and stored-value consistency. Numerical checks support the algebra review; they are not substitutes for proofs.
- Measured KL: 43 grouped checks covering all numeric table entries, missingness, seed counts, source comparisons, and unchanged plotted geometry.
- Compute accounting: 281 checks over 75 existing run records; generated table and macro values match exactly.
- Visual review covered the main comparison and all theory/compute pages 91–118. The corrected Figure 40 placement was inspected again after recompilation.
- Final compilation has no undefined/multiply-defined references, overfull boxes, duplicate destinations, or rerun-required label diagnostics. All 91 numbered appendix headings appear in the manual TOC. The typeset and PDF metadata titles agree.
- The standalone Overleaf ZIP compiles independently. Its 141 checked source/assets and 108 compiler-recorded input hashes agree with the release snapshot. The live included sources and released sources agree.
- The nine-page main-body export has matching text and identical rendered pixels before/after removal of links to excluded appendix pages.

The general paragraph-ending audit still reports 45 preexisting short prose endings outside P/Q and one mathematical superscript-star extraction false positive on page 99. This pass introduces no new failures. Two existing stationary-wrapfigure warnings and an empty artifact-footnote anchor warning remain elsewhere in the manuscript; the current theory/KL figure has no wrap overflow. These are recorded rather than represented as a completely warning-free global build.

## Records

`validation.json` records release checks and artifact hashes. `preservation-and-layout.json` records formal-content preservation and the comparison with the incoming PDF's paragraph endings. `main-changes.patch` compares the latest starting source with the final source. The subdirectories contain the independent mathematical, empirical, cost, and visual review records. Verification scripts retain the temporary paths used for this run.
