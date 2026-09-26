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

The main text follows the reader's dependencies: mathematical-agent motivation;
a concrete collapse example
and measured losses across scales; executable benchmark identities; the fresh
objective and replay bank, with idealized theory in the appendix; plain-language
evaluation metrics immediately before the matched experiments;
and Section 5, Conclusion and Limits. A short main-text related-work paragraph
links to the full discussion in the appendix.
All **six existing main-text figures** are retained, in their original order.
The eight supplementary figures remain in the supplement. Figures are
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
eleven complete MaxRL--ReplayMaxRL pair blocks cover all five domains at
Qwen2.5-0.5B and Falcon3-1B, plus Qwen2.5-3B Python. The four-arm
intersections contain ten complete blocks and one four-seed Falcon Countdown
block. Graph, Countdown, Python, and MathIR now have complete five-seed
Level-2 factorials. Countdown gains extra modes under ReplayMaxRL, while its
correctness interval includes zero; Python/MathIR correctness effects remain
inconclusive. Cross-domain factorial averages are post-hoc
descriptive summaries. The paper keeps these boundaries explicit.

## September 9 results refresh

Both manuscripts now report 126/150 admitted E118 endpoints, 83/100 E119
endpoints, and 35/45 E120-R1 endpoints. They provide every planned model/domain
comparison with exact paired seed counts: 11 complete E118 pair blocks, four
complete Level-2 factorials, and six complete E120 blocks. The completed
Level-2 domains are Graph, Countdown, Python, and MathIR; PantryPlan remains
incomplete. Figure 6 uses step-3072 matched checkpoints for the four completed
domains and its explicitly labeled step-0 PantryPlan row.

The new appendix tables report pass@8, distinct@8, and extra correct modes
(B8 = distinct@8 − pass@8). Complete five-seed blocks carry the registered or
documented paired uncertainty; partial blocks retain their observed seed
counts and have no five-seed intervals. E120 uses uniform-minus-frequency
contrasts and preserves its frozen Qwen-0.5B primary analysis. The new
Qwen-3B Graph comparison has two completed, telemetry-validated pairs;
Falcon PantryPlan remains a three-pair contrast favoring frequency weighting.
The standalone result digest, CSV, JSON, and forest plot accompany the source
bundle. Earlier dated sections below record historical revisions.

## Build

Requires a standard TeX Live installation with pdfLaTeX and BibTeX; artifact
checks also use Python 3 and Poppler's `pdftotext`.

```sh
make                # compile, resolve references, check page/figure limits
make bundle         # also create mathai2026-source.zip
```

Compilation runs in a temporary directory. The PDF and compilation receipt
are replaced only after all source, layout, and output checks pass. A failed
compile or page-limit check preserves the last validated PDF and saves
`main.failed.log`; it does not publish the rejected draft. Intentional source
or asset updates must also update the corresponding bindings and history in
`snapshot.json`. The standalone source ZIP is validated before replacement.


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

`snapshot.json` records the current parent and workshop source hashes, official
style hash, copied asset hashes, and an explicit correction history that
preserves superseded hashes. The source review excluded Falcon Countdown
ReplayDr.GRPO seed 59, job 30269051: its terminal log contains conflicting
four-draw evaluations and the existing August 28 amendment excludes it from
efficacy reporting. No retry value was selected. Primary results therefore use
74 admissible pairs: 14 complete five-seed blocks and one descriptive four-seed
block. All eleven complete MaxRL replay comparisons retain their registered seed sets.

The Falcon Dr.GRPO cross-domain track uses the same four admissible seeds in
every domain and has no five-seed interval. Supplementary trajectories use
recovery-authorized sources and omit ambiguous nonterminal checkpoints. The
full source audit and proof-to-evidence assessment are recorded in the parent
repository under `paper/audits/`; the paper discloses the exclusion in
Appendix `app:source-integrity`. Live partial experiments do not replace the
frozen five-seed evidence snapshot.

`check_submission.py` checks every referenced input, source and asset hashes,
and a compilation receipt binding the PDF and compiler outputs to those inputs.
It also checks compiled page references, all fourteen figures, anonymous output,
unchanged official style, references beginning on page 5, and compilation
diagnostics. The main pages are also rendered for visual review. `make` records
the receipt after successful compilation; `make clean` followed by `make`
recreates it when needed.

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

The September 6 results addition includes the complete Qwen2.5-3B Python
factorial table and synchronized E118 figure assets. It also adds the
registered E120 extra-mode endpoint analysis for all five completed
Qwen2.5-0.5B domains, with all paired seeds, bootstrap intervals, and an
explicitly limited mechanism interpretation. The existing E120 progress snapshot remains fixed at 25 of 45
terminal cells; partial Level-2 cells, E117 development runs, and DAPO
upstream-only diagnostics do not become efficacy evidence. Both paper versions
receive the same numerical additions and evidence boundaries. The source ZIP
includes the new input tables and their JSON provenance. See the
[addition audit](../audits/results_additions_20260906/) and the dated snapshot
correction entry for the exact assets and source hashes.


## September 6 build repair and evidence refresh

The expanded Figure 6 caption had pushed two conclusion lines onto a fifth
content page. LaTeX succeeded, but the page-limit check rejected the output
and the old Makefile deleted `main.pdf`. Tightened prose restores four content
pages with all six main figures, unchanged template dimensions, and the full
conclusion. The build now preserves the last validated PDF on failure.

Figure 6B uses 21 matched domain–seed cells from its September 6 frozen
snapshot. Graph, Countdown, Python, and MathIR contribute five seeds each;
Pantry contributes seed 45 at the observed initial checkpoint only. Both
levels and all methods share a checkpoint within each cell; seeds are averaged
within domain and the five domains receive equal weight. This is an interim
comparison. Graph remains the sole complete five-seed Level-2 factorial block;
the snapshot also contains two complete four-arm seeds each for MathIR and
Python. Figure 5's Qwen2.5-3B Dr.GRPO average is complete; its MaxRL average
remains unavailable.


The dated endpoint update is in `results/latest_results_20260906.json` and
Appendix `app:latest-results-update`. It adds current completion/endpoint
counts alongside the frozen figure snapshots. Newly complete Falcon3-1B
Graph weighting results use every registered paired seed, 55–59: uniform
replay adds .233 extra modes (95% paired bootstrap interval [.182, .320]).
The pass@8 interval [−.0004, .0910] includes zero, and mean sampled correctness
decreases by .0152. Both effects and all seed values are reported; the result
does not establish universal accuracy safety. Both new JSON artifacts and all three table
bodies ship in the standalone source bundle.

The fresh census (September 6, 16:56:42–17:09:35 UTC) validates E118 at
116/150 terminal cells (11/15 complete blocks), E119 at 52/100 (1/5), and
E120 at 33/45 (6/9). E118 contains 57 matched MaxRL/ReplayMaxRL pairs;
its new Qwen3B Graph and MathIR pairs each have only one seed and receive
no interval. No frozen primary endpoint or figure was replaced by a partial run.


The conclusion now explains what ModeBench contributes, why retaining several
approaches may matter when problems or constraints change, and which links to
richer mathematical problem solving remain untested. The revised ending still
fits on page 4; the parent paper develops the same interpretation and limits.


The abstract and first section now lead with mathematical-agent candidate
generation, tool use, verification, recovery, and self-correction. Alternative
proof or computation routes motivate the study; ModeBench supplies the controlled
executable-output measurement. The first section is “Preserving alternatives
for mathematical agents.” Figure 1 remains on page 1 and the main text remains
four pages. See the [workshop-fit revision audit](../audits/build_repair_20260906/workshop_fit/report.md).


The main four pages carry the complete argument: the loss of sampled output
support, exact executable identities, replay weights independent of fresh
frequency, held-out evidence across objectives and scales, a complete harder
Graph factorial, and the output-versus-latent-strategy boundary. The harder
example reports correctness and mode counts, while the five-domain difficulty
figure is explicitly interim. Figure assets and audited endpoints are unchanged;
see `../audits/four_page_argument_20260906/report.md`.

The abstract follows the author's preferred framing: a language model
can become more accurate on mathematical problems while losing alternative
correct solutions.
It introduces the benchmark, replay, and the scoped results in that order,
then separates discovering a successful behavior from keeping it available.
The introduction develops why that distinction matters for mathematical agents
and how correctness feedback can hide the loss of alternatives. Figure 1 follows the abstract on page 1, which the build now explicitly
requires. The fuller narrative opening, all six main figures, and the conclusion
fit within four content pages; repeated setup was shortened to make room.
The result heading is now “Replay improves both Dr.GRPO and MaxRL.” All displayed
mentions of the benchmark use the existing `\mb{}` small-caps macro, including
headings, captions, the conclusion and supplement in both paper versions.
Code identifiers and file paths retain their literal names. See
`../audits/narrative_opening_20260906/report.md`.


The main metric paragraph now connects correctness and support to Theorem
`thm:shared-correctness`: under the stated idealized categorical assumptions,
stronger updates shared across correct outputs can preserve more sampled modes
at matched correctness for a fixed prompt, with identical pass@8. The raw
counts and extra-mode difference still depend on correctness. The main
frequency-weighting comparison reports its extra-mode gain; its correctness
analysis remains in the supplement.

The official CFP was checked again on September 6: submissions allow four
content pages and unlimited references/supplementary pages. It does not state
that a separate supplement is required; the current combined PDF follows those
published rules. Accepted camera-ready papers may use five content pages.

The current official template ZIP contains the same unmodified style file as
this bundle. Its anonymous branch hardcodes the generic conference footer.
A narrow submission-only override in `preamble.tex` now names the MATH-AI
workshop while retaining anonymity, line numbers, fonts, margins and the
official style bytes. Camera-ready mode retains the official workshop notice.
The compiled submission check enforces the corrected footer. Evidence and
source comparison: `../audits/correctness_proof_bridge_20260906/`.


The September 9 update adds the completed five-seed fixed-bank Graph mechanism
study. The supplement reports every seed's frozen-bank coverage, first-to-final
mean-token and sequence-score changes, worst intermediate changes, pooled
empirical distributions, and registered hierarchical bootstrap intervals. The
main text gives a short, scoped reference. This measures exemplar likelihood
trajectories; it does not measure exact canonical-mode probabilities or identify
a causal replay effect. The bundle carries the result JSON, three table bodies,
and the new two-panel empirical-distribution figure. See the parent audit in
`../audits/e121_results_20260909/`.
