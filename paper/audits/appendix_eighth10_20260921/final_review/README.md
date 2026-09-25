# Final review and exports — eighth appendix block

Reviewed **PDF pages 89–98**, including the connected P.5 proofs through page **100**. The next scheduled block begins on **page 99**, whose P.5 material has already been reviewed.

[Final PDF](exports/main.pdf) · [Final Overleaf archive](exports/iclr2027_overleaf.zip) · [Release checks and hashes](validation.json)

These fixed export copies also populate `paper/main.pdf` and `paper/iclr2027_overleaf.zip`. The main-only and full-with-figures PDFs are updated alongside them.

## Editing and review

The theory introduction and P.1–P.5 state results, definitions, assumptions and limitations directly. Derivation-history, novelty and presentation narration are removed. The text uses **solution modes** and distinguishes categorical logits from neural parameters, infinitesimal mean flow from finite steps, and mean identities from stochastic trajectories.

The review checks the five categorical assumptions, fixed reward-dependent advantages, MaxRL endpoints and the group-size-two case, neural means and covariances, inverse-length weighting, shared-prompt counterexample, SetPO specialization, finite expected steps, sampled symmetry breaking and finite-budget visibility. Scientific limitations remain explicit.

There are **no figure captions in this block**. Tables 56–57 already have bold findings followed by sampling conditions, populations, metric definitions, weighting and uncertainty status; their captions and numbers were checked and preserved. Collection time, provider defaults, missing responses and control compliance remain because they affect the reasoning-control comparison.

Final corrections remove the duplicated proof heading for Theorem 3.1, specify positive response length in the inverse-length example, and state the weighted covariance bound directly. One connected main-body sentence now distinguishes lower overall correctness and raw mode counts from conditional diversity, whose change varies by deployment.

## Checks

- Independent [early-theory](../theory_early.md), [neural-theory](../theory_neural.md), and [editorial](../language_review.md) reviews found no remaining blocking mathematical or interpretation issue. The final duplicate-heading and main-body suggestions are integrated.
- [Additional algebra validation](../algebra_validation.json): **1,480 checks pass**, including 60 estimator/policy cases and 1,344 exactly enumerated groups. These corroborate the identities; they are not experimental runs.
- [Source validation](source-validation.json): numbered equations and labels are preserved. Empirical tables, measurements and intervals are unchanged. No training, model calls or bootstrap reruns.
- Pages 89–100 and the connected main-body sentence were inspected visually. No clipping, equation collisions or stranded headings were found. A line-ending parser flag is a displayed summation index, documented in [target-line-fill.json](target-line-fill.json). Preexisting global line-ending findings outside this pass are not claimed resolved.
- The final PDF has **122 pages**, with **nine main-text pages**, twelve main figures and **all 91 numbered appendix headings** in the contents. Build diagnostics show no undefined references/citations, duplicate labels/destinations, overfull boxes or figure collisions. Existing empty-anchor and stationary-wrapfigure warnings are nonblocking.
- The Overleaf archive compiles independently. **141 packaged source/asset files** and its reference PDF match the release snapshot; **108 compiler-recorded input hashes** match. The main-only export matches the full PDF’s first nine pages after removing links to omitted pages.

## Concurrent source work

Four unrelated reference-KL files have newer live edits; their table currently has fifteen columns while the manuscript heading declares fourteen. The exports use the coherent prior figure/macros/table set and preserve all live edits. [Paths and hashes](concurrent-source-handling.json) record the distinction. All sources edited in this appendix pass match these exports.
