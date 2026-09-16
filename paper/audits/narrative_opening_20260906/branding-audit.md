# ModeBench branding audit — September 6, 2026

The user requested consistent formatting of every displayed occurrence of
ModeBench. The existing canonical form is `\mb{}`, defined as
`\newcommand{\mb}{\textsc{ModeBench}}`. Keep this definition unchanged in
`paper/main.tex` and `paper/mathai2026/preamble.tex`. Normal title, heading, bold,
and caption context can retain its existing font size and weight while the name
consistently uses small caps.

## Source findings and exact replacements

Replace only the literal display word `ModeBench` with `\mb{}` at these pre-edit
locations. Line numbers identify the audited source before the concurrent
narrative revision; the surrounding phrases identify the replacements after
line movement.

| Source | Count | Pre-edit lines and identifying text |
| --- | ---: | --- |
| `paper/main.tex` | 14 | 85 title “ModeBench and ReplayMaxRL”; 107 “ModeBench thus provides”; 162 “ModeBench therefore asks”; 186 bold contribution “ModeBench measures verified output support”; 279 “ModeBench reports three averages”; 404 “ModeBench targets”; 420 “turns ModeBench keys”; 573 “ModeBench Level 2”; 793 “ModeBench makes”; 803 “comparisons. ModeBench”; 808 “ModeBench provides”; 3933 “ModeBench currently releases”; 4139 “five ModeBench domains”; 4434 “The ModeBench example renderer”. |
| `paper/mathai2026/main.tex` | 2 | 48 section “ModeBench makes output identity executable”; 191 conclusion “ModeBench makes a missing part of evaluation measurable”. The narrative draft placed the corresponding occurrences at lines 58 and 201. |
| `paper/mathai2026/appendix.tex` | 7 | 122 “ModeBench reports”; 272 “ModeBench targets”; 365 “ModeBench Level 2”; 3574 “ModeBench currently releases”; 3803 “five ModeBench domains”; 4098 “The ModeBench example renderer”; 4134 “ModeBench measures”. |

Existing `\mb`/`\mb{}` occurrences in prose, headings, and captions already use
the canonical small-caps styling. In particular, the parent benchmark section
and the benchmark example/table captions need no separate stylistic change.
The parent title inherits small caps from the ICLR style, but using `\mb{}` there
still makes the brand explicit and consistent at the source level.

Preserve the macro definitions and all code/data references, including
`sec:modebench`, `fig:modebench-examples`, `modebench_examples.pdf`,
`modebench_level_admission.pdf`, `graph_coloring_modebench_v2`,
`python_factor_modebench_v1`, `pantry_plan_modebench_v2`,
`modebench_harder_v2_matched_r5`, `modebench_level_comparison_snapshot.json`,
`plot_paper_modebench_levels.py`, `python_modebench.py`,
`python_modebench_process.py`, and `python_modebench_worker.py`. These are literal
identifiers and paths, not prose branding. Both `example_paper.bib` files contain
no ModeBench occurrence; bibliography metadata requires no change. Result-table
TeX fragments in both versions likewise contain no raw ModeBench occurrence.

## Figure audit

Checked extracted text from **26 displayed figure PDFs**: the complete set of
13 referenced PDFs in `paper/figures/` and their 13 workshop copies in
`paper/mathai2026/figures/`. All files contain extractable text. A case-insensitive
`mode\s*bench` search found **no display occurrence** in any of them, so no figure
regeneration or image editing is required.

The 13 filenames are:

- `baseline_collapse_precheck.pdf`
- `direct_baseline_learning_curves_static_strip.pdf`
- `direct_comparator_endpoint_effects.pdf`
- `e118_all_scale_factorial_progress.pdf`
- `e118_scale_extensions_appendix.pdf`
- `experiment1_retention_comparator_matrix.pdf`
- `modebench_examples.pdf`
- `modebench_level_admission.pdf`
- `modecollapse_story.pdf`
- `replay_mechanism_telemetry_qwen05b.pdf`
- `sustained_auc_effects_qwen05b.pdf`
- `verified_support_discovery_two_scale_effects.pdf`
- `verified_support_story.pdf`

The audit follows multiline `\includegraphics` references as well as references
whose option and filename share one line; thus it includes all seven appendix
figures, not only the six main figures.

## Macro and typography verification

Compiled isolated two-pass examples using each actual paper preamble and its
local style files. Each example exercised `\mb{}` in a title, section,
subsection, table of contents, normal prose, bold prose, and bold caption.
Both passed with no undefined command, LaTeX error, PDF-string warning, or
small-caps font-substitution diagnostic. Table-of-contents writes retain
`\textsc{ModeBench}`, and PDF bookmarks contain no literal `\mb` command.
The existing macro is safe for the tested moving arguments; no replacement
macro or hyperref workaround is needed.

Temporary test artifacts and logs are in
`/tmp/modebench-typography-audit-brpz_7xb/{parent,workshop}/`.

Source edits and full-paper builds belong to the root integration task. The
parent's required full build and line-fill audit were reported passing after
its replacements. The workshop narrative draft already used the canonical macro
in its heading and conclusion when the audit completed; its promotion/build
was in progress. This audit changed no manuscript, bibliography, figure, style,
macro definition, or generated submission artifact.

Audit completed at 2026-09-06T18:12:14.261396+00:00.
