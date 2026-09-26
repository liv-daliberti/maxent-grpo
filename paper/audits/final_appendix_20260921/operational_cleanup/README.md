# Operational prose cleanup: main text and appendices A–O

## Deliverable

`edits.json` contains 58 exact replacements, each with `path`, `old`, `new`, and `reason`. It changes `paper/main.tex` and 13 active input TeX files. All 58 original strings matched the latest live files exactly in a sequential dry application before root integration. No live paper source, figure, data, builder, or model files were edited by this agent. Root's independent Reproducibility Statement changes are excluded.

`changes.patch` and `proposed/` show the proposed edits against the read-only scope snapshot in `before/`. **Do not replace main.tex with the snapshot/proposed copy**: the snapshot intentionally ends before Appendix P and root/other agents own subsequent content. Apply only the exact JSON replacements to the latest full source.

## Scope and changes

Recursively read/scanned 51 active TeX sources through Appendix O. Checked references across all 54 active sources in the full paper.

- Main experimental/results prose now states the scientific result directly instead of referring to saved completions, records, and a complete audit.
- Removed the three-model revision-hash table, which had no inbound references. The only removed label is **`tab:pinned-revisions`**; it is metadata, not a scientific result.
- Replaced exact training seed IDs with replication counts throughout the text/captions, and replaced raw seed IDs in the inference-cost table with paired replicate labels (a)/(b). Removed evaluation seed constants, bootstrap RNG constants, and hash-selection implementation details while retaining randomization/selection independence and sample sizes.
- Condensed training-curve descriptions to scientific populations and visual encodings. Removed precise missing-checkpoint location and partial-cohort construction procedures. Rings/dotted lines, seed ranges, conditional eligibility and changing-population limitations remain where needed to interpret unchanged figures.
- Removed the per-seed initial-sampling provenance paragraph, logging-completeness inventories, the sampling-audit run/checkpoint totals, and individual-response availability/exclusion narratives.
- Condensed the overlapping-stream analysis while retaining the actual dependence limitation, eleven nominal stream identities under the stated sampler convention, one outcome-independent representative per identity, two disjoint 5/6 splits, and the distinction between conditional concentration and intact-group marginal metrics. All following sensitivity effects/sign conclusions remain.
- Removed API retry counts, HTTP status/unfinished-attempt counts, response-ID narration, request-slot/hash-rank details, and exact bootstrap-success inventories. Retained provider refusals, truncations, denominator definitions, cost-understatement limits, collection-time confounding and validated-control limitations.

## Preserved scientific content

No plot assets or input paths changed. No theorem, proposition, corollary, lemma, proof or displayed equation blocks changed. Scientific labels remain. No data files, measured table values, intervals, or figure coordinates changed. The only changed numerical table entries are the removed revision hashes and row-name seed identifiers, not measurements. `table_row_changes.json` documents those differences.

Scientific replication counts, correct-pair eligibility, population definitions, nominal uncertainty, substantive provider outcomes and important experimental limitations remain. Actual replay scheduling/hash use in the algorithm is a method definition, not publication-process narration, and remains. Prompt examples retain their executable task content. Source comments are code-only and were not edited merely because they mention past auditing.

Removed numeric prose is administrative/auxiliary inventory: RNG IDs; hash constants; evaluation/run/response-attempt counts; scheduled-score completeness; per-seed initial request differences; a single checkpoint's eligibility explanation; and bootstrap-defined replicate counts. The corresponding scientific sample-size rules, aggregate results and uncertainty caveats remain.

## Validation

`validation.json` records exact-match checks, label/figure/input preservation and balanced environments. Every changed file passed. `tab:pinned-revisions` had zero inbound references across the full active TeX dependency closure.

Residual scan of changed prose finds no literal seed-ID, SHA/hash, API-attempt or checkpoint-inventory wording. The regex's sole apparent seed-ID match is the scientific phrase “single-seed 3B effects,” a false positive.

A follow-up parent-authorized figure cleanup changed `ops/exp_scaling/build_paper_e121_survival.py` and the corresponding PDF/PNG legend from Seed 43–47 to Replicate 1–5. Only the existing JSON and the renderer's `plot()` function were used; no analysis or bootstrap ran. `fixed-bank-figure/before/` preserves the original renderer, images and JSON. `fixed-bank-figure/geometry_check.json` confirms identical plot data, line styles, axes geometry/limits, PDF page box and PNG dimensions; pixel changes are confined to the legend. The retained input JSON hash is unchanged. All 40 active figure PDFs were scanned; no other raw seed-ID labels were found. Visual inspection confirmed that the new legend fits.

The separate first-valid retry diagnostic (`frontier_python_retry_20260911.tex`) remains because it reports a deliberate selected-sampling sensitivity, including changed collision and the distinction between selected and unconditional accuracy. It is not an API retry/error ledger. Parent notified as a possible further editorial cut if that auxiliary sensitivity is no longer wanted.

## Template synchronization

The exact JSON was sent to `/root/latest_measured_kl`, which owns renderer/template synchronization. This reviewer made no builder edits. Header-identified builders include `build_paper_gpt56_all_levels32_discovery.py`, `build_paper_hosted_reasoning_off.py`, `build_paper_inference_followups.py`, and `build_paper_portfolio_withdrawals.py`; other changed input basenames are listed in the JSON.
