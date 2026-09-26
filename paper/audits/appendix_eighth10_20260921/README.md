# Eighth appendix prose and presentation pass — 2026-09-21

## Scope

Reviewed incoming PDF pages **89–98**, including the reasoning-control boundary from the previous pass and P.1–P.5 of the mathematical appendix. The complete connected P.5 subsection was also reviewed through the finite-budget visibility bounds on pages 99–100. The next ten-page pass starts on PDF page **99**, with P.5 already cleaned and P.6 remaining to review.

The two boundary tables (56–57) already have supported bold takeaways and concise definitions of populations, grading, weights, and uncertainty. Their captions and numbers were checked and preserved. There are no figure captions in this page range.

## Changes

Removed rendered narration about subsection insertion, derivation history, novelty, proof length, and which theorem was stated first. Replaced it with the mathematical conclusions, definitions, and assumptions. Deleted the two stale source comments instructing where to insert the neural subsection. Retained attribution and scientific distinctions between isolated categorical logits, neural parameter sharing, mean flow, exact finite steps, stochastic updates, and the implemented optimizer.

- **Categorical model (P.1–P.2):** explicitly requires a fixed common length-normalization factor. A sample-dependent shared factor can alter the expected gradient's direction. The collapse remark now cites all five assumptions, including absence of competing regularization and a nondegenerate start. Dr.GRPO and GRPO retain their distinct Liu/Shao attributions; the MaxRL coefficient and all-failure convention retain their source attribution.
- **Outcome-binary advantages (P.3):** gives the conditioning and Bernstein-coefficient argument directly. Specifies a fixed reward-dependent advantage for a single coefficient polynomial. The centered MaxRL endpoint gap was already correct and is explained explicitly. The coefficient ratio is nonincreasing, strictly decreasing for group sizes above two, and identically two at group size two. Mixed-reward groups become rare near both correctness boundaries, not only near zero.
- **Neural identities (P.4):** preserves the mean/covariance identities, finite-step bound, shared-prompt counterexample, length residual, and SetPO specialization. Explicitly states a fixed prompt-sampling distribution and integrable aggregate updates. A common response length removes the residual for inverse-length weighting; it need not remove arbitrary response-dependent weighting. The normalized mathematical comparison is distinguished from the implemented training update.
- **Finite-step and sampling bounds (P.5):** states the retained categorical assumptions explicitly. Clarifies that matching the diversity-versus-correctness trajectory requires infinitesimal mean flow. Corrects the interpretation of the stochastic coefficient: its asymptotic decay is 1/G, but it need not decrease at every finite increment of group size. All numbered mathematical equations and existing labels are preserved.

## Verification

No training, new responses, model calls, or bootstrap runs. Empirical fragments, tables, estimates, and intervals are unchanged. All manuscript source after P.5 is unchanged from this pass's baseline. A concurrent Section 2 clarification is included: disabled reasoning lowers raw mode counts, while conditional diversity changes vary by deployment. This agrees with the unchanged Tables 56–57. Its exact diff is retained separately. The final integration also uses a single proof heading and explicitly states positive response length for inverse-length weighting.

- [Categorical review](categorical/README.md): 509 checks, including exact group enumeration, mean-gradient identities, coefficient limits, and an example showing why a random shared normalization factor needs a separate assumption.
- [Outcome-binary review](binary/audit.md): 5,864 rational-arithmetic and structure checks, including the coefficient polynomial, MaxRL boundary cases, and the mixed-group bound.
- [Neural review](neural/README.md): 102 checks, including exact means/covariances, the two-prompt example, length residuals, and the verified-key functional.
- [Finite-step review](finite/audit.md): 1,584 deterministic identity/inequality checks, 92 exact binomial identities, and direct enumeration of the small-group symmetry-breaking expansion. Includes primary-source verification of the cited replicator variance result.

The release uses an isolated, consistent source snapshot. A concurrent reference-KL table update produces fifteen columns while the existing table heading declares fourteen. Its associated figure, second figure, and macros also differ from the incoming release. The four incoming reference-KL files are therefore retained only in the isolated PDF/archive snapshot; live work is preserved. Exact paths and hashes are in `concurrent-source-handling.json`. These unrelated changes are outside this appendix pass.

All twelve reviewed pages (89–100) were inspected as rendered images and extracted text. The mathematical identities and proof examples pass their deterministic checks. The final target prose has no short paragraph endings under the existing line-fill rule; its one automatic flag is an extracted superscript star in a display on page 99, visually confirmed as a structural false positive. The full-document scan retains 45 preexisting flags outside this batch. The checker itself was not changed.

After the source snapshot, a concurrent edit rephrased the Cauchy–Schwarz proof in P.4. That live refinement is preserved; the PDF and ZIP use the already-reviewed, mathematically equivalent wording. `live-wording-after-snapshot.patch` records the distinction. Source and package checks are against the immutable reviewed manifest, so subsequent live work cannot silently enter the release.

Final release checks pass: **122 full PDF pages, nine main-text pages, twelve main figures, and all 91 numbered appendix headings in the contents**. References start on page 11. No undefined references/citations, duplicate labels/destinations, overfull boxes, or collision diagnostics. The preexisting empty-anchor and two stationary-wrapfigure warnings remain nonblocking. All 141 archive source/assets and the included reference PDF match the reviewed snapshot; all 108 compiler-input hashes match. The standalone main PDF has nine pages and renders identically before and after removing 146 links to omitted pages. Final PDF/archive hashes were verified after installation in `paper/`.

## Final independent review and release

The final review integrates the duplicate proof heading, positive inverse-length condition, weighted-bound explanation and connected main-body correction. The final exports and checks are recorded in [final_review/README.md](final_review/README.md). This is the final release record for this pass.
