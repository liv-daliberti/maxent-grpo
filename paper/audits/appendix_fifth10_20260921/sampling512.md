# Appendix K.3: 512 responses per problem

Scope: `paper/results/gpt56_all_levels32_sampling_20260913.tex`, its renderer in `ops/build_paper_gpt56_all_levels32_discovery.py`, and the single generator digest in the adjacent publication JSON. No changes to main.tex or figure assets. Before copies are saved here.

## Editorial changes

- Replaced follow-up, frozen-verifier, native-receipt, staged-collection, expansion-history, source-record, and archival-JSON prose with direct sampling design and interpretation. The nonrendered source record still preserves provenance.
- Kept outcome-independent SHA256 problem selection, original wording, deployment snapshot, medium reasoning, 8,192-token output cap, unspecified temperature/top-p, inclusion of failures, and transport-retry counting.
- Defined finite-pool rarefaction as uniform subsets of each 512-response pool and kept equal weighting across all 32 problems, including unsuccessful problems.
- Preserved the difference between ordered prefixes and rarefaction, paired whole-problem bootstrap resampling, pointwise coverage, support bounds, service-stability qualification, and the inability of flat finite-budget curves to exclude rare modes.
- Rewrote the table caption with a supported bold takeaway, model/population/budget, outcome definitions, both grading conventions, support interpretation, contrast, and interval type.
- Added the actual main-figure estimand: mean distinct modes divided by mean enumerated support, with fixed denominator for interval scaling. This is not a mean of per-problem coverage fractions.
- Explained that Python's table reference (at least two certified modes per problem) differs from the figure's mean enumerated counts (222, 335.75, 354.75). This resolves an otherwise conspicuous apparent inconsistency without altering either record.

## Validation

`python ops/build_paper_gpt56_all_levels32_discovery.py --check` passes. Its validator reconstructs rarefaction point means and final-doubling contrasts from mode counts for all 30 graded cells (15 cells under two grading conventions), checks all 960 graded prompt pools, all 10 budgets, prefix endpoints, retained prompt identities, and support counts. It checks interval shape and containment against stored intervals; it does not rerun the bootstrap or authenticate raw provider responses.

All table contents (including headers), the displayed rarefaction equation, and both labels are byte-identical to the starting fragment. Apart from `render_tex`, the generator AST is unchanged. The publication JSON differs only in `builder.sha256`, updated to keep the existing consistency check valid. All measured values, stored intervals, source data, selection information, and interpretation/provenance records are otherwise identical. No API calls, new experiments, or bootstrap draws were performed.

The main-body figure caption calls the denominators available modes. Its underlying non-Graph references are still qualified as lower bounds in this appendix, and the denominator implementation is now explicit. Main-body prose and captions were outside this task's assigned scope and were not changed.

Combined-document compilation and page-level visual review remain with the parent agent.
