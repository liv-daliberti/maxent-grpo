# Correctness proof, weighting result and workshop footer — September 6, 2026

The main metric paragraph now cites Theorem `thm:shared-correctness`: for a
fixed prompt under the idealized categorical assumptions, stronger updates
shared across correct outputs can preserve more sampled modes at matched
correctness, with identical pass@8. The theorem compares the same initial
policy with a specified shared-update geometry at a finite target correctness
below one. This is not an empirical equal-accuracy or neural-model guarantee.
The warning that both support counts remain coupled to correctness is retained.

The frequency-weighting paragraph now reports only its extra-mode improvement
and confidence interval. All correctness results and their uncertainty remain
in the unchanged supplement. Protocol and caption prose was tightened and the
Countdown endpoints expressed with arrows; no evidence values changed.

Official submission rules were verified at https://mathai-2026.github.io/cfp/:
four content pages, unlimited reference/supplementary pages, and double-blind
workshop mode. No separate-supplement requirement is stated. Five content pages
are allowed for accepted camera-ready papers. The website's instructions govern
over generic conference boilerplate retained in the example source.

The current official template ZIP was downloaded and its style file exactly
matches the local copy (see `official-template-check.json`). The anonymous
style branch ignores the workshop title. A narrow preamble notice override
corrects this while preserving the official style, anonymity and line numbers.
The independent footer probe checked anonymous and camera-ready behavior; see
`footer-audit.md`. The final build check now enforces the workshop notice.

The staging build retains Figure 1 on page 1, all six main figures, and four
main pages. Final combined-build evidence and hashes follow below.

The combined build passed all submission checks: four main pages, all six main
figures, Figure 1 on page 1, references on page 5, the correct MATH-AI footer,
and no unresolved references or overflow. Pages 1 and 3 were visually checked
for the changed footer and proof paragraph. All four main pages were rendered.
The 54-file source ZIP was refreshed; hashes are in `final-artifacts.json`.
