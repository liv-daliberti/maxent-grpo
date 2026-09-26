# MATH-AI NeurIPS 2026 workshop version

**More than One Way to Skin a Cat: Preserving Verified Modes in RLVR**

`main.pdf` is the anonymous submission: **four pages of main content**, then
references and supplementary material. `main.tex` is the root source;
`preamble.tex` selects the format; `appendix.tex` holds the expanded material.
The original `paper/main.tex` and its PDF retain their ICLR title and receive
the same scientific corrections, proof additions, and evidence boundaries.

The version uses the unmodified style from the
[official MATH-AI template](https://mathai-2026.github.io/files/mathai_neurips2026_template.zip).
The [workshop requirements](https://mathai-2026.github.io/cfp/) allow four content
pages and unlimited references/supplementary material. The supplied archive's
example retains generic conference instructions; the workshop CFP determines
the four-page limit. The example is retained as `template_example.tex`.

## Editorial choices

The main text follows the reader's dependencies: a concrete collapse example
and measured losses across scales; executable benchmark identities; the fresh
objective and replay bank with direct appendix proof links; plain-language
evaluation metrics immediately before the matched experiments;
and Section 5, Conclusion and Limits. A short main-text related-work paragraph
links to the full discussion in the appendix.
All **six existing main-text figures** are retained, in their original order.
The seven existing supplementary figures remain in the supplement. Figures are
copied from the audited source artifacts; corrected result figures were
regenerated after the source-integrity review described below.

Full definitions, theory, related work, domain descriptions, experimental
design, expanded results, implementation details, disclosures, and the original
supplement are in `appendix.tex`. Main-text captions and prose were rewritten
for the workshop. Template font sizes, margins, and style definitions were not
changed. All main and supplementary figure captions use a consistent 7-point
skip with redundant interline glue removed. Figure 3 spans the text width;
Figures 2 and 3 crop only verified blank PDF padding. Figure 4 omits the in-graphic
paired-seed footnote (the caption explains the daggers) and adds 5 points below
the panel B heading. Numerical expansions and the exact binary MaxRL
advantage are in the appendix, leaving more main-text space for why the design
and findings matter.

Three-scale evidence concerns ReplayDr.GRPO versus Dr.GRPO. ReplayMaxRL's
ten complete MaxRL--ReplayMaxRL pair blocks cover Qwen2.5-0.5B and
Falcon3-1B. The four-arm intersections contain nine complete blocks and one
four-seed Falcon Countdown block. Only Graph has
complete trained Level-2 evidence. Cross-domain factorial averages are post-hoc
descriptive summaries. The paper keeps these boundaries explicit.

## Build

Requires a standard TeX Live installation with pdfLaTeX and BibTeX; artifact
checks also use Python 3 and Poppler's `pdftotext`.

```sh
make                # compile, resolve references, check page/figure limits
make bundle         # also create mathai2026-source.zip
```

In Overleaf, upload the source ZIP and select `main.tex` as the main document.
`environ.sty`, `trimspaces.sty`, and `natbib.sty` are copied from the original
paper to support the local TeX installation. All figure PDFs, input tables,
and bibliography entries needed to compile are included. Research-code paths
inside the supplement refer to the original repository, not this source bundle.

Submission mode uses:

```tex
\usepackage[dblblindworkshop]{neurips_2026}
```

For camera-ready preparation, populate the authors in `main.tex`, update the PDF
author metadata in `preamble.tex`, and run `make camera-ready`. This selects:

```tex
\usepackage[dblblindworkshop, final]{neurips_2026}
```

The resulting `camera-ready.pdf` is separate from the anonymous submission.

## Snapshot and verification

`snapshot.json` records the September 4, 2026 source hash, official style
hash, copied asset hashes, and an explicit integrity-correction history that
preserves superseded hashes. The source review excluded Falcon Countdown
ReplayDr.GRPO seed 59, job 30269051: its terminal log contains conflicting
four-draw evaluations and the existing August 28 amendment excludes it from
efficacy reporting. No retry value was selected. Primary results therefore use
74 admissible pairs: 14 complete five-seed blocks and one descriptive four-seed
block. All ten complete MaxRL replay comparisons remain intact.

The Falcon Dr.GRPO cross-domain track uses the same four admissible seeds in
every domain and has no five-seed interval. Supplementary trajectories use
recovery-authorized sources and omit ambiguous nonterminal checkpoints. The
full source audit and proof-to-evidence assessment are recorded in the parent
repository under `paper/audits/`; the paper discloses the exclusion in
Appendix `app:source-integrity`. Live partial experiments do not replace the
frozen five-seed evidence snapshot.

`check_submission.py` checks compiled page references, all thirteen figures,
anonymous output, unchanged official style, the corrected asset snapshot,
references beginning on page 5, and compilation diagnostics. The main pages
are also rendered for visual review.

The September 5 mathematical review clarified the finite-group MaxRL objective,
completed the categorical collapse argument with a direct ratio bound, and
added the positive-weight replay extension and its inverse-length interpretation.
It also corrected relative replay-frequency accounting and made uncertainty
and factorial-interaction scope explicit. Both manuscripts received the same
scientific corrections. Frozen empirical figures and tables were preserved;
see `../audits/mathematical_review_20260905.md` for the assessment and reviewer
requests, and the recorded snapshot correction history for source hashes.

The September 5 review also adds a deterministic finite-step categorical replay lemma with an explicit step-size bound. It proves convergence to the fixed bank weights and makes the incomplete-bank limitation explicit: unbanked correct modes have zero limiting mass in this idealized model. See the [independent proof review](../audits/finite_step_categorical_replay_independent_review_20260905.md); this does not extend the guarantee to stochastic neural optimizers.

The subsequent September 5 appendix expansion integrates the controlled
stochastic retention theorem, conditional admission and changing-bank results,
exact-entropy objective comparison, categorical natural-gradient comparison,
and sharp per-mode survival certificates. Each includes relevant literature
attribution and explicit application limits. The four-page workshop main text
is preserved. Both versions include matching new bibliography entries; see the
[appendix integration record](../audits/theory_appendix_integration_20260905.md).

The theory appendix now uses worked examples throughout its twelve subsections,
including a recurring three-correct-mode policy and examples for each part of
the survival analysis. Exact expressions, rounded numerical illustrations,
and probability guarantees are distinguished explicitly; all example values
are hypothetical. Formal statements, proofs, and bibliography are preserved.
See the [examples and readability record](../audits/theory_examples_20260905.md) for calculations and checks.
