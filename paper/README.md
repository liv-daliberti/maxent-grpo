# Paper: Exploration Should Act at the Group Level in RL LLM Fine-Tuning (xDr.GRPO)

NeurIPS-format draft of the xDr.GRPO paper: a candidate-level maximum-entropy
variant of Dr.GRPO that places exploration on the prompt-local distribution over
sampled completions rather than at the prompt or token level.

## Build

```bash
make            # pdflatex + bibtex + 2x pdflatex -> main.pdf
make clean
```

Requires a TeX distribution with `pdflatex` and `bibtex`. Style files that are
not part of a minimal TeX Live install are vendored in this directory:
`neurips_2025.sty` (official NeurIPS 2025 bundle), `cleveref.sty`,
`nicefrac.sty`, `wrapfig.sty`, `environ.sty`, `trimspaces.sty`. The `mathabx`
package is loaded only if present on the system (no mathabx symbols are used).

## Files

- `main.tex` — the paper source.
- `example_paper.bib` — bibliography.
- `Makefile` — build.

## Provenance notes

- The source `main.tex` was imported from an external draft in two pieces
  (the main body plus the appendix tail through `\end{document}`). Two small
  pieces in the gap between them are reconstructions, not original text: the
  proof of the "Coordinatewise stability" proposition and the boundedness /
  tau-range subsection labeled `app:tau-range` (that label is referenced from
  the original text but no section carrying it appeared in either piece). Both
  are marked by the `NOTE TO AUTHORS` comment in `main.tex`.
- All PAC-Bayes material was removed at the author's request: contribution
  (iv) now reads "robustness guarantees" only, and the appendix subsection
  "On generalization bounds" was dropped.
- `fig:exploration-sidecar` is currently a framed placeholder — replace it with
  the real exploration side-car diagnostics plot.
- The experimental sections were rewritten (2026-07-12, replacing the imported
  draft's `[NEW DATASET]` placeholders and its Qwen2.5-Math-1.5B / MATH12k /
  math-benchmark setup) to describe the repo's exact multi-answer ModeBench
  pair: Countdown arithmetic (`ops/make_exact_countdown_mode_data.py`) and
  graph-coloring completion (`ops/make_exact_answer_mode_data.py`), trained on
  Qwen2.5-0.5B-Instruct with binary exact-verifier rewards and mode-coverage
  metrics.
- Bibliography entries were web-verified where possible; entries whose
  provenance could not be confirmed are marked with `TODO` comments in
  `example_paper.bib`.
