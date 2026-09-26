# Second presentation pass: appendix PDF pages 18–27

The sequential cleanup previously reached the end of the appendix (PDF page 122). This pass starts a second review at the first ten appendix content pages, covering Appendices A and B and Figures 13–17. The next range is PDF pages **28–37**, starting with Appendix C, Exact Domain Prompts.

## Changes

- Removed repetitive explanations of correctness, diversity, and within-prompt pooling.
- Clarified that promptwise unbiasedness assumes independent responses from a fixed policy. Dependence between prompts alone does not invalidate each prompt’s expectation; selection by additional observed outcomes can change it.
- Replaced “full-support threshold” with “reporting threshold” to avoid implying that the observed sample exhausts solution support.
- Removed an unnecessary tally of low-support cells, while preserving estimator eligibility and reported sample sizes.
- Reduced the Python external-execution paragraph to its scientific contract: syntax restrictions, isolation, execution/response limits, validation, and failure grading. Removed worker/client/restart mechanics.
- Labeled the dataset table’s mode counts as certified catalogue counts, consistent with the domain table, rather than claiming exhaustive support for every verifier.
- Made Figure 17’s marker explanation self-contained and gave Table 6 a direct takeaway. The unequal-total-compute limitation remains explicit.
- Stated interval eligibility directly, without completion-status language.

## Verification

All ten rendered pages were visually inspected. The revised captions have no clipping, collisions, or short final lines. All displayed equations, formal statements/proofs, labels, main-text source, and source from Appendix C onward are unchanged. No experimental records, plotted coordinates, or analysis code were modified; no experiments or statistical estimates were rerun.

The full PDF remains 122 pages and the main text nine pages. All 89 numbered appendix headings are in the contents. Compilation has no undefined/multiply-defined references, overfull boxes, duplicate destinations, or pending label updates. The existing empty-anchor and two stationary-wrapfigure warnings remain outside this edit.

The strict paragraph-ending check has no flags on pages 18–27; global flags decrease from 47 to 46 and remain outside this batch. This is not a claim that every global typography gate passes. The Overleaf archive is independently compiled and checked against the release inputs; see validation.json for export hashes and checks.

A concurrent edit appended nine unused KL comparison macros during export. These definitions are preserved in the final source package; they do not change the rendered paper. The export was rebuilt after the consistency check detected the addition.
