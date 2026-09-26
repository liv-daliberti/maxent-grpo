# Narrative opening and consistent benchmark styling — September 6, 2026

The workshop abstract follows the author's preferred wording nearly verbatim,
adding “held-out” to qualify the correctness result and using the benchmark
macro. Its narrative begins with accuracy improving while alternatives are
lost, introduces measurement and replay, reports the scoped results, and ends
by separating discovery from continued availability. The introduction develops
the same distinction and why it matters for mathematical agents. The precise
frequency condition and the distinction between verified output support and
latent reasoning strategies remain in the main text. The broader conclusion
and all audited evidence remain intact. The result heading is now
“Replay improves both Dr.GRPO and MaxRL.”

All displayed benchmark names in both manuscripts use the existing small-caps
macro. This changes 14 parent-paper occurrences, two workshop main-text
occurrences, and seven workshop appendix occurrences. Macro definitions, code
identifiers and paths retain their literal spelling. The figure and bibliography
audit is recorded separately in `branding-audit.md`.

Figure 1 follows the expanded abstract on page 1. Repeated introductory and
method setup was shortened to retain four main pages, all six main figures,
and the conclusion on page 4. The build now explicitly requires Figure 1 on
page 1. The parent
paper rebuild passed the scientific contract and all 293 prose line-fill checks.
Final workshop validation and artifact hashes follow below.

Final `make bundle` passed: four content pages, all six main and seven
supplementary figures, references on page 5, current source/asset bindings,
unchanged anonymous template, and no unresolved references or overflow. The
54-file source ZIP matches the current sources. All four main pages were
visually inspected; extracted text exactly matches the staged pages reviewed.
Final hashes are in `final-artifacts.json`. The parent PDF has 61 pages; the
workshop PDF has 60 pages including references and supplement.

The subsequent Figure 1 placement requirement is recorded in the snapshot
history and `before-figure1-placement/`. Final artifact records are refreshed
after the build with the new page-1 guard.

The final guarded build passed with Figure 1 on page 1, four main pages,
all six main figures, and references on page 5. The current source ZIP includes
the new guard and matches current sources exactly. All four final main pages
were visually checked; `final-artifacts.json` and page previews reflect this
placement. No evidence values or template dimensions changed.

The author-preferred abstract is recorded in the final snapshot correction and
`before-user-abstract/`. It retains Figure 1 on page 1 and the four-page limit.

The build with the author-preferred abstract passed all submission checks,
including Figure 1 on page 1 and four main pages. The refreshed 54-file ZIP
matches current sources; all four final pages were visually inspected. Artifact
hashes and page previews now reflect this final abstract and introduction.
