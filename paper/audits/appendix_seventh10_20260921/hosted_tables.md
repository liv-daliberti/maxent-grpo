# Appendix O cohort tables and provider outcomes

Incoming coverage: PDF pages 82–85, including the provider-outcome subsection, strict equal-domain overview, strict cell longtable, formatting-normalized cell longtable, and the connected final provenance paragraph. The parent also authorized a narrow consistency repair to the earlier O opening and protocol after Figure 39's two rows were found to use different grading populations.

## Edited sources

- `paper/results/frontier_comparison_20260911_provider_outcomes.tex`
- `paper/results/frontier_comparison_20260911_strict_cells.tex`
- `paper/results/frontier_comparison_20260911_normalized_cells.tex`
- `paper/results/frontier_comparison_20260911_appendix.tex`
- `paper/results/frontier_comparison_20260911_protocol.tex`
- Rendering portions of `ops/build_frontier_paper_comparison.py` (`LAYOUT_LEGEND`, `cell_tables`, and `export`). No collection, grading, bootstrap, metric, or plotting function changed.

## Presentation and statistical repairs

- Removed frozen-verifier, overwritten-grades, completion/admission, source-digest, audit, prospective-comparison, and figure-selection chronology from the rendered text. Removed the final paragraph recounting result-record and figure-selection history.
- Provider narrative now states what native classifications measure and their limits. Refusal signals, filtering, and absence of visible answer text are distinct attributes. Reasoning-only output can have an empty visible answer; counters overlap. A native refusal is not evidence of mathematical inability, and a zero native refusal count does not detect or exclude refusals expressed solely in ordinary answer text.
- Preserved all native stop-state counts, original-condition Python refusal counts (409, 1,021, 1,024), and the provider's 2,454 Python category labels. Recast the category statement as a description of service behavior, without treating it as evidence of task harmfulness or correct-output concentration.
- Provider table caption leads with the observed fact that only Opus 5 has native refusal signals in this cohort. It gives the 128-prompt, eight-response denominator; R/F/E definitions; overlapping-counter caveat; 22 displayed versus 83 zero cells; and the limit of zero native refusal counts. All numerical rows are unchanged.
- Strict level-summary caption leads with the supported observation that every deployment-level row averages fewer than three distinct correct solution modes per prompt. It specifies original instructions, strict grading, five equal domain weights, eight draws on 128 prompts per domain, zero-inclusive mode counts, within-domain correct-pair pooling, 2,000 whole-prompt bootstrap resamples, and undefined macro means.
- Both cell-table captions define C as pooled correct-pair collision, distinguishing it from one minus prompt-averaged PCMD. U uses the same pair weights and certified support counts. D includes zero-mode prompts. Dashes distinguish no eligible correct pairs from unavailable total support; Countdown has no finite-total-support uniform reference.
- Cell-table intervals remain pointwise 95% percentile intervals from 2,000 whole-prompt resamples, separately within each domain-level cell. No new intervals were computed.
- Normalized caption leads with the observed result that accuracy increases without a consistent decrease in collision. It describes the same executable verifier following typography corrections, with no repair of values or program logic. There are 14 increases and 16 decreases in collision across the changed cells.
- Corrected the normalized longtable's continuation heading, which previously read “Strict executable grades.” The row-selection code compares formatted columns, so the caption now says omitted cells match at the printed precision rather than claiming identical underlying grades from the rendering rule alone. The existing 30 selected rows remain identical.
- Corrected O's opening and protocol to distinguish Figure 39's accuracy row (formatting-normalized, with revised Opus 5 Python instructions and original-condition hollow companion marks) from its PCMD row (strict grades and original instructions for every deployment). This follows `ops/plot_paper_hosted_breadth.py`: normalized accuracy is read from the comparison record, while promptwise PCMD is joined from `mode_diversity_hosted_cohort_20260917.json`. The parent handles the matching figure caption and N.1 forward reference.

## Validation

`validate.py` passes 109,652 assertions, including identity checks on 107,520 saved native provider classifications. `validation.json` summarizes the results.

- All 105 strict cell rows, all 30 normalized cell rows, all 22 provider counter rows, and the strict overview numerical fragment are byte-identical before and after. Both existing tabular environments are unchanged.
- Reconstructed every original native R/F/E cell and stop-state total from the provider classification records; all seven models have 15,360 distinct response identities. Classification file hashes match their source records. The records confirm 22 nonzero cells, 83 zero cells, and zero native refusals outside Opus 5.
- Checked strict and normalized accuracy, distinct-mode counts, pooled collision, uniform references, and all equal-domain level means against existing numeric counts. All defined C estimates have all 2,000 defined bootstrap replicates. Original Opus 5 Python Levels 2/3 remain undefined for collision under both grading rules.
- Every reported known-support cell has observed collision above its conditional uniform reference. Every strict deployment-level distinct-mode mean is below three.
- Confirmed that normalization never removes a strict success, adds 4,266 correct responses across the seven original-instruction cohorts, and can increase or decrease collision. Grok has no changed correctness counts.
- Re-rendered the five assigned TeX files from the existing numeric JSON with graph generation disabled. No API requests, new generations, verifier execution, bootstrap draws, or plotting ran.
- The numeric result JSON is unchanged, SHA-256 `5571694f63d5a18fa1e52e19f014042996ccb5321b225b36e253495bc7522c59`.
- All labels and input paths are unchanged. The normalized continuation heading is corrected. The owned rendered fragments pass the process-language scan.
- `python -m py_compile ops/build_frontier_paper_comparison.py` passes. AST comparison confirms changes only in the two rendering functions (plus the legend string).

Full-document compilation, pagination, figure placement, and release packaging are handled by the parent on the combined source snapshot.
