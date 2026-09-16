# Final verification after storage recovery

The current long paper has **8 main pages**, including all six main
figures, with references starting on page 9. The complete PDF has
55 pages. This build preserves the newer hosted table across all five
domains and three levels, six deployments, and the separate Python prompt follow-up.

- [Main text only](main-text.pdf)
- [Validated full paper](final/main.pdf) and [frozen source](final/main.tex)
- [Validation](validation.json), [compiled input hashes](compiled_inputs.json),
  [build log](build.log), [evidence checks](evidence.log),
  [25 pagination tests](tests.log), and [frozen prompt check](prompts.log)

Generated hosted captions now explicitly distinguish per-response accuracy from
pass@8. All admitted original model observations are identical to the concurrent
update captured in concurrent_update/. Earlier inputs are preserved in before/.

All eight rendered main pages were visually reviewed. The main-only PDF matches the first main pages of the full paper exactly under
pdftotext. There are no overflowing boxes, undefined references, or duplicate
labels. The existing line-fill gate passes. The six main figures, three training
model scales, both training metrics, partial-cohort qualifications, and smaller
model UCPO/RLEP restriction remain intact. This task did not edit the short paper.

Compilation used a frozen source copy and an isolated output directory. All
literal manuscript inputs, local styles, bibliography, and figure bytes were
checked against their hashes before promotion. The source bundle is in final/.
