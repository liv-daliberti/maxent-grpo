# Final appendix and operational-detail cleanup — 2026-09-21

The manuscript and its source package now use the concise paper-wide presentation requested by the user. Run identities, checkpoint/conflict inventories, retry accounting, revision-hash tables, RNG identifiers, source-selection histories, and logging/recovery mechanics were removed from rendered prose. Scientific design, sample sizes, estimators, uncertainty, measured outcomes, and substantive limitations remain. Detailed implementation records remain outside the manuscript; no LOCAL_ONLY inventory is included in this audit or the Overleaf package.

## Changes

- Completed the remaining extended-comparison and resource sections. The comparison accurately describes DMPO's Boltzmann target/MSE implementation and UCPO's sequence-likelihood weighting; it distinguishes evaluated comparators from literature-only comparisons and preserves the actual uncertainty in reported effects.
- Reduced the final appendix to evaluation comparisons, public dataset/model links, and analysis resources. Consolidated its headings and synchronized the manual TOC; old reference labels remain valid aliases.
- Applied 58 further presentation edits across the main source and thirteen active appendix inputs. Removed the unused model-revision metadata table and its unreferenced label. Replaced a figure's literal seed identifiers with neutral replicate labels; its numerical geometry is unchanged.
- Updated ten TeX-generating Python scripts so future rendering retains the concise wording. Eight isolated renderers reproduce the updated prose from existing records; the two mixed analysis/rendering builders were checked structurally without execution. Numerical logic and measurement records were not changed by those template edits.
- Removed the remaining reasoning-modes terminology, corrected the domain-table population comparison, and qualified unsupported inferences from explicit-reasoning settings and unobserved MathIR routes.
- Preserved incoming work: domain-name font formatting and new KL measurements arrived concurrently. The manuscript and both KL figures now use the latest 28 measured coefficient cells, with all six coefficients in the three-domain plot. Removed obsolete plane shading and updated the nonmonotone coefficient description. No old result set was substituted.
- Updated the 40-figure inventory. Added a float barrier so the exact-prompt appendix no longer begins across intervening benchmark figures.

## Verification

- All 124 displayed mathematical expressions in main.tex and six in included result files are preserved. All 69 formal statement/proof blocks are preserved, allowing only the concurrent domain-font markup. All scientific labels remain; the sole removed label belonged to the unused metadata table.
- The main text remains nine pages with twelve figures. The full PDF has 122 pages and all 89 numbered appendix headings appear in the manual TOC.
- No undefined or multiply-defined references, overfull boxes, duplicate destinations, or pending label updates occur. The metadata and typeset titles agree.
- Visual review covered the main KL comparison, the corrected prompt boundary, changed training/concentration/ablation/hosted pages, both latest KL figures, the extended comparison, and the concise final appendix. No clipping or collisions were found.
- The Overleaf ZIP compiled independently. All 141 checked source/assets and 108 compiler-input hashes agree with the release snapshot and current live inputs. The standalone nine-page PDF was checked after repairing links to excluded appendix pages.
- No training, new model calls, measurement collection, or bootstrap reruns were performed. The KL data update was incoming work; this task reconciled presentation with it. Existing JSON records retain the code hashes of their original creation; current presentation-template hashes and equivalence checks are recorded in template_cleanup rather than relabeling original measurements.

The repository's stricter global paragraph-ending audit still reports 47 flags, compared with 46 before this pass. Some line endings changed with the concurrent domain-font formatting and caption cuts; the one-symbol mathematical-star flag is an extraction artifact. These flags do not indicate overfull boxes or clipped content. Two preexisting stationary-wrapfigure warnings and the empty artifact-footnote anchor warning remain. The build and visual checks above are not a claim that every global typography gate passes.

`validation.json` contains deliverable hashes and packaging checks. `final-preservation.json` records mathematical/reference preservation and layout checks. The subdirectories contain the independent comparison, template, figure, and visual reviews. No operational inventory is rendered in the paper or separately added to its source package.
