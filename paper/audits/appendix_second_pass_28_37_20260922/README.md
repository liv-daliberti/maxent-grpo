# Second presentation pass: PDF pages 28–37

Reviewed exact prompts, the replay algorithm, Fixed Semantic MaxEnt, terminal comparisons, and the training-curve boundary, including Figures 18–21 and Tables 7–9. The next sequential batch is PDF pages **38–47**; it begins with the remaining training curves and supporting comparisons. Incoming material now extends the full manuscript beyond the previous 122-page version.

## Editorial changes

- Clarified that later levels have separate problems and prompting differences while preserving every character of the five exact prompt transcripts.
- Tightened teacher-forcing and Semantic MaxEnt count descriptions without changing admission, scheduling, loss weights, count semantics, or displayed mathematics.
- Removed redundant completion/cohort inventories from surrounding results prose. Captions and tables still identify sample sizes, prompt eligibility, uncertainty, and the populations defining each mean.
- Replaced the inaccurate description of all four non-MathIR domains as alternative-answer modes: Countdown uses derivation routes.
- Made hypothetical quality filters explicitly hypothetical; no filter is implied to have been used in the reported comparison.
- Clarified caption encodings, retained direct takeaway leads, and removed repetitive availability narration from the training-accuracy caption.
- Prevented Table 7 from floating above the Training results part heading. Repaired short paragraph/caption endings throughout the target range.

## Preserved content and verification

All original main-source displayed equations, formal statements/proofs, Algorithm 1, exact prompt blocks, and labels are preserved. Source from the mathematical appendix onward is unchanged. Experimental measurements, training code, existing figure coordinates, and numerical table bodies were not changed by this cleanup. No training, collection, model calls, or statistical recomputation was performed by this task.

All ten target pages were inspected as rendered images, including the corrected table placement and final caption wrapping. The target pages have no short paragraph endings, overfull boxes, undefined references, or visual collisions. The stricter global typography checker still flags paragraphs outside this range; it was not weakened. Existing empty-anchor and stationary-wrapfigure warnings remain.

## Incoming work

Concurrent work added a 64-response concentration evaluation, two result tables, a figure, main-text summaries, and an eligibility clarification. These additions are preserved. Their contents entries were synchronized. This cleanup shortened the newly added evaluation and KL summaries to maintain the nine-page main-text limit without dropping numerical comparisons or the stated qualifications. Subsequent incoming refinements to those summaries and a float barrier were also preserved. The original latest theory was not replaced by an earlier version.

The finalizer compiles an isolated snapshot, independently compiles the Overleaf archive, verifies the package against compiler inputs, checks all numbered appendix contents entries, and verifies the nine-page main-only export. Final page counts and released file hashes are recorded in validation.json.
