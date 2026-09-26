# Author-main restoration and reference audit — 2026-09-22

The manuscript uses the supplied main in `user_main_exact.tex`. Its title, prose, numbers, captions, and organization are preserved except for the documented link/citation repairs and the subsequent edits explicitly requested by the user: defining discovery/rehearsal frequency and expressing four comparator gains in percentage points.

## Changes

- Thirteen main-text reference/citation/path repairs are recorded in `main_link_repairs.json`. Ten verified bibliography entries were appended; every existing bibliography entry was preserved.
- Two appendix backreferences were repaired (`appendix_link_repairs.json`), and `app:registered-endpoints` now anchors the existing support-threshold paragraph.
- All twelve original main figure assets are retained, in source order, with unchanged file hashes. The supplied main cited but omitted the weighting figure, so its original display and caption were restored beside that discussion. This added figure uses `[H]`; a page-break guard before the following level figure prevents wrapfigure clipping. No supplied figure or caption was removed or rewritten.
- The theorem and setup omitted from the supplied main were moved next to their appendix proof. All 68 formal statements/proofs are preserved, allowing only the recorded correction to one internal section reference.
- The latest user-requested definitions and four percentage-point conversions are recorded in `user_requested_edits.json`. These are absolute probability differences rescaled by 100, not relative percentage increases.

## Verification

`verify_restore.py` reconstructs the main from the immutable paste, applying only recorded repairs, explicitly requested edits, and figure/reference scaffolding. It also checks the preamble, appendix, formal blocks, all twelve figure assets, and preservation of existing bibliography entries. Results are in `fidelity_check.json`.

The isolated PDF build has no undefined citations, undefined references, duplicate-label warnings, or overfull boxes. Visual checks cover the main figures and captions. The main has **10 pages**, followed by one statement page, with references starting on page 12; the full PDF has 125 pages. The figure-placement check passes with all twelve figures inside the main. The existing nine-page limit still fails; the checker and the author's prose were not changed to conceal that result.

The line-fill style checker reports 42 of 563 prose blocks below its 50% final-line threshold. This is recorded in `build/line_fill.txt`; no prose was rewritten to satisfy that stylistic rule. The broader historical story-contract checker assumes the previous main structure and was not used as a requirement for this author-directed restoration.

Local build inputs and final outputs are hash-recorded in the adjacent JSON audit files. Original manuscript, PDF, bibliography, and build snapshots are retained as `*.before` files.

## Remaining factual questions

See [source_findings.md](source_findings.md). It distinguishes evidence mismatches from link fixes. In particular, the PCMD `.991` statement, Re:Max/Re:Dr attribution, and KL figure population/interpretation remain unchanged, as instructed. The frequency-definition placeholder has been resolved under the user's subsequent authorization; the other factual flags remain for review.

The [main-only review PDF](../../main-body.pdf) and [full review PDF](../../main-with-figures.pdf) were refreshed to match the final manuscript. The main-only copy has exactly the same extracted text as pages 1–10 of the full PDF; hashes are in `review_copy_verification.json`.
