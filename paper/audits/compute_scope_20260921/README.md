# Compute-scope revision — 2026-09-21

Section 4, Table 6, the algorithm-control description, and Appendix Q now distinguish matched optimizer-update/fresh-rollout schedules from equal realized training compute. The three new Appendix Q subsections cover the control, component accounting, and the unperformed equal-compute comparison. Their entries appear in the appendix contents.

## Evidence and scope

- The maintained learner performs a detached replay scoring forward, then a live score forward and backward. The compute-only switch zeros score derivatives but retains those operations. Fresh-task and replay gradients precede the same optimizer step (`src/oat_drgrpo/learner/grpo.py`, around lines 1001–1037, 1279–1331, and 1907).
- Bank populations, sequence lengths, and skipped empty banks can differ across policies. Sharing code paths does not establish identical realized FLOPs or runtime.
- No reported control reallocates that work to useful additional baseline gradient passes, fresh samples, or a longer horizon. The manuscript explicitly leaves that empirical comparison open. No new training or runtime profiling was performed for this revision.
- The existing 75-run cost table is a descriptive likelihood-pass subtotal relative to a hypothetical reference forward. Its numbers are unchanged. The manuscript now explains its missing detached forward, unequal-length approximation, diagnostic-row averaging, and other excluded costs. The accounting builder changed only in documentation/comments.
- References to a “compute-matched” control elsewhere in the manuscript now specify the matched update and fresh-rollout budgets.

## Validation

- Clean stable-source pdfLaTeX/BibTeX build: 124 total pages, 9 main pages, 12 main figures; disclosures on page 10 and references on page 11.
- No undefined references/citations, duplicate labels/destinations, overfull boxes, or wrapfigure collisions.
- All 90 numbered appendix headings have contents entries.
- Visually inspected Section 4 and Appendix Q pages; inspected the final wording against the learner and accounting code.
- Independently compiled the Overleaf export. Its source and reference PDF exactly match the installed files; full and main-only review PDFs were refreshed.
- Preserved the solution-mode terminology, prior presentation repairs, and concurrent mathematical revisions.

Validated main.tex SHA-256: `dc509ffc13082051e517886c7d8039c9743512d3433086b4df63fdf34077b190`.
