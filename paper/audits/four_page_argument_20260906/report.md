# Four-page argument revision — September 6, 2026

The workshop main text makes the five requested points self-contained.

| Point | Main-page location | Evidence or definition |
|---|---|---|
| Problem | 1 | Abstract, collapse example, three-scale loss estimates |
| Measurement | 2 | Exact task-defined canonical keys; domain examples |
| Intervention | 2–3 | Replay bank and equation; each retained key weighted independently of fresh frequency |
| Evidence | 3–4 | Held-out protocol, 74 pairs, both objectives across completed scales, paired intervals, full five-seed harder Graph example |
| Boundary | 1–2, 4 | Verified output support does not establish latent strategy diversity; causal and downstream usefulness remain untested |

The Qwen2.5-0.5B ReplayMaxRL-minus-MaxRL paired t intervals and completed Graph
endpoints were independently recomputed and confirmed by the editorial review.
Only Graph is complete across all four methods and five seeds at Level 2.
The frequency-weight ablation improves extra modes but its primary correctness
effect is inconclusive. No frozen results, figures, template settings, or
parent-paper sources were changed by this editorial revision.

The preliminary compile places all six main figures and the conclusion within
four content pages. Final build and archive validation are recorded below.

The independent scheduler/receipt breakdown requested during the revision is
in `job_status.md`, with the corresponding raw snapshot and JSON.

Final `make bundle` passed the source, snapshot, compilation-receipt,
page, figure, anonymity, style, reference and overflow checks. The current PDF
has 4 content pages and 60 total pages; references begin on page 5. All four
main pages were visually inspected with no clipping or overlaps. The source
ZIP contains 54 files. Hashes are recorded in `final-artifacts.json`; detailed
compiler/checker output is in `workshop-build.log`.

An independent fresh-directory extraction of the 54-file ZIP also passed
`make` and `check_submission.py`. Its archived `main.tex` matches the current
source exactly, and the first four PDF pages produce byte-identical layout
text to the current PDF. See `standalone-build.md`, `.json`, and `.log`.
