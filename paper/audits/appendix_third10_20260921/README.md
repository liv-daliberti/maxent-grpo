# Third appendix prose and caption pass — 2026-09-21

## Scope

Reviewed PDF pages 38–47, following the prior 28–37 pass. This span contains the Level-2 training panels, supporting-comparator curves, diversity comparators, replay weighting, semantic-entropy factorials, conditional concentration, and the beginning of the baseline/output-collapse discussion. Adjacent Figure 27 and the rest of H.3/H.4 were also corrected so that their connected scientific explanations remain consistent.

Removed editorial/process narration, retrospective chronology, source-admission and repair history, archival/artifact paths, and reader-directed defenses. Kept actual sampling methods, populations, missing observations, uncertainty, and limitations. Figure captions use a bold supported finding followed by setup, visual encodings, and relevant statistical qualifications. The H and H.1 headings now stay together, and changed subsection titles match the contents.

## Scientific corrections

- **GAPO and SetPO:** specified the implemented GAPO reward rescaling and valid-rollout frequency denominator, SetPO coefficient 1, frozen embedding model, similarity clipping, and extra embedding computation. Having more valid modes than rollout slots limits within-group coverage; it does not make the uniform policy impossible or the inverse-support reward term numerically zero. Replaced that unsupported explanation with observed effects. MathIR effects are small, not uniformly absent. Separate controls and their observed differences remain explicit; small differences do not establish equivalence.
- **Replay weighting:** defined `P_8` and `B_8` locally. Falcon Graph's mean per-response correctness change is negative with an interval below zero; its pass@8 effect is inconclusive. Removed the unsupported causal division assigning all accuracy gain to reuse and all diversity gain to weighting. Bootstrap design and every reported effect/interval were verified against existing records.
- **Semantic entropy:** clarified the ten four-arm model/domain cells, nine with five common seeds and Falcon Countdown with four. The 3B extension has one seed and no semantic-only arm, so it cannot identify a factorial interaction. Fixed coefficients do not identify optimal tuning. The numerical table is byte-identical; the generator reproduces the revised prose/caption.
- **Conditional concentration:** retained the collision identity and ideal-flow derivation, selected-population limitations, 475-run/902-evaluation population, all missingness, and five-seed interval rules. Four eight-output groups contain eleven nominal stream identifiers, not 32 independent streams. The split-stream sensitivity results and uncertain runtime mapping remain explicit. Lower collision concentration does not prove survival of particular stored modes. All eight numerical tables, containing 135 rows, are byte-identical, and the generator reproduces their revised captions.
- **Figure 23:** removed the footer's unsupported suggestion of independent partial histories. The footer now identifies cohort means, seed ranges, and partial-seed points. Data and geometry are unchanged.
- **Figure 25:** verified that Re:Dr has the highest terminal mean pass@8 in all ten displayed panels. Clarified terminal cohorts, method-specific comparator populations, partial checkpoints, and seed-range bands. Removed source-admission wording from the embedded subtitle. Data and geometry are unchanged.
- **Figure 26A:** described the actual paired, jointly eligible prompt contrast, full-five-seed intervals, partial means, and unidentifiable comparisons.
- **Figure 26B:** corrected the claim that this is a Qwen3B-only panel. It pools available model-scale means equally, while averaging available evaluations within each scale. Some checkpoint seeds recur under different context limits; these remain separate evaluation entries and are not additional training seeds. The caption now discloses this existing aggregation rather than silently changing the analysis. Intervals appear only for single-scale marks. Corrected the open-marker support rules and removed false claims that MaxRL stays near zero or that Dr.GRPO narrows in every domain at every level.
- **Baseline summaries:** 150 runs correspond to 300 initial/final checkpoint evaluations, each with four eight-response groups. Current common-domain PMD summaries give 98.5%/99.7% declines for 0.5B GRPO/Dr.GRPO, and Dr.GRPO's terminal mean rounds to .001. The larger-model range is 50.8%–65.1%. Explicitly listed the common measurable domains and removed stale unpaired macro comparisons. The differing 3B initial requests belong to Dr.GRPO controls, not replay arms. Uniform-action references are specified for Graph/Pantry; other domains are not all open-ended.
- **Output collapse and learning rates:** distinguished eight-response uniqueness statistics from the 32 recorded, overlapping-stream responses. Retained the finite-sampling limitation. Rewrote the adjacent control discussion to state that controls change under the tested learning rates, while no rate sweep establishes optimal tuning, the sign of an alternative-rate contrast, or whether tuning could remove an accuracy gap. Aggregate pass@8/mean@8 equality alone does not prove identical strings. The control table's independently averaged run populations and extra Falcon Countdown replay seed are explicit.

## Validation

- Independent prose and numerical reviews against local result files, implementation, and plotting code.
- All ten target pages visually inspected; affected pages rechecked after pagination changes.
- Numerical tables, experiment records, training code, and plotted measurements are unchanged. No experiments were run.
- The two changed embedded figure text elements preserve their numerical snapshots and geometry.
- Main-body extracted text is byte-identical to the incoming PDF; nine main pages, all twelve main figures, references beginning on page 11.
- All 90 numbered appendix headings are represented in the contents.
- No unresolved references/citations, duplicate labels/destinations, overfull boxes, or wrap collisions.
- Full PDF: 122 pages, down from 123. The existing compute appendix is unchanged.

Deliverables are rebuilt from a validated source snapshot. The Overleaf archive is independently compiled and checked against the installed source, figures, and reference PDF.

The nine-page main-body export also removes 146 dangling link annotations pointing to omitted references/appendix pages. Valid internal and external links are retained. All nine pages render pixel-identically before and after this export-only repair; text and metadata are preserved, and PDF inspection/rendering produces no diagnostics. The full PDF and Overleaf archive are unaffected.
