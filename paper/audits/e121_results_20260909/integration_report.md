# E121 manuscript integration and build verification — September 9, 2026

The completed five-seed fixed-bank Graph study is now in both papers. The main
texts report the pooled median mean-token score change, +.038 nat/token, and
the 2.61% tail declining by more than .5 nat/token. Both appendices report all
five seeds, full scheduled-observation coverage, both score scales, worst
intermediate changes, the pooled empirical distributions, and the registered
10,000-draw hierarchical bootstrap.

The interpretation distinguishes teacher-forced exemplar scores from exact
canonical-mode probabilities and from a causal replay effect. Historical
occupancy-only claims are scoped to their original cohorts; the obsolete
statement that no E121 result is reported has been removed. The analysis also
discloses variable first/final visit steps, length-dependent sequence scores,
pruned resume checkpoints and coverage reconciliation, and exclusion of the
carried-forward terminal summary.

## Result placement

| Artifact | Main finding | Complete E121 appendix |
|---|---|---|
| [ICLR PDF](../../main.pdf) | Page 10, existing Limits and next steps paragraph | Section H.2 starts page 56; Tables 11–13 and Figure 13 on pages 56–57; interpretation continues page 58 |
| [Workshop PDF](../../mathai2026/main.pdf) | Page 4, Conclusion and Limits | Section L.2 starts page 54; Tables 11–12 on page 55; Figure 13 and Table 13 on page 56 |

The source and score provenance is in [analysis.md](analysis.md). The strict
coverage audit and independently replicated statistics are recorded in
[the integrity audit](../e121_20260909/integrity.json) and
[the statistical review](../e121_20260909/independent_statistics.json).

## Validation and page budgets

- `make -C paper` passed the current-paper contract, domain-prompt checks,
  three-pass TeX/BibTeX build, and rendered line-fill gate: 307 natural-prose
  blocks with minimum final-line fill of 50%. No overfull boxes or unresolved
  references remain.
- `make -C paper/mathai2026 bundle` passed isolated compilation, all source and
  asset bindings, anonymous official-style checks, the four-content-page limit,
  six main and eight supplementary figures, references beginning on page 5,
  and source archive validation.
- Root visual review passed for ICLR pages 56–57 and workshop pages 55–56:
  every table column and value is visible, figure legends and axes are readable,
  and nothing is clipped. PDF text contains the audited counts, negative-tail
  percentage, and confidence endpoints in both versions.
- An isolated three-pass compile of the saved pre-E121 ICLR source confirms
  that it already used ten main-content pages: Conclusion heading on page 9,
  Figure 6 and Limits on page 10, AI Use Statement on page 11. The update
  preserves that budget. Total ICLR length rises from 61 to 63 pages through
  the added appendix evidence; the workshop has 63 total pages and four main
  pages. See [baseline_pagination.json](baseline_pagination.json),
  [baseline_main.pdf](baseline_main.pdf), and [baseline_page10.txt](baseline_page10.txt).

The new figure, result JSON and all three generated table bodies are copied
byte-for-byte into the workshop directory and bound by `snapshot.json`.
The validated [source ZIP](../../mathai2026/mathai2026-source.zip) contains
65 files. The refreshed [Overleaf ZIP](../../mathai2026/mathai2026-overleaf.zip)
contains the same files plus a package manifest binding every entry and the
source archive by SHA256; archive content and CRC checks passed.

Final artifact hashes, sizes, labels and build outcomes are in
[integration_build.json](integration_build.json). Final build logs are
[iclr_build.log](iclr_build.log) and [workshop_build.log](workshop_build.log).
The exact source changes from the preserved pre-integration files are in
[integration_source_changes.diff](integration_source_changes.diff).
