# Hosted-domain typography

Applied the benchmark-domain typewriter style to all 17 included hosted fragments in the assignment inventory. Added 275 `\texttt{...}` spans to domain mentions in headings, prose, captions, and table labels, including the short domain label `Pantry`. The shared formatter also handles `Countd.`; none required an additional change in this hosted scope. Existing typewriter spans, verbatim content, paths, reference identifiers, and ordinary code names are preserved.

Every edited fragment differs from its saved original only in these domain-name wrappers. All table measurements, intervals, numeric values, prose wording, and references are unchanged.

Six maintained emitters now call `ops.paper_domain_typography.format_domain_names` only when returning or writing rendered TeX:

- `ops/build_frontier_paper_comparison.py`
- `ops/build_frontier_temperature_paper.py`
- `ops/build_paper_gpt56_all_levels32_discovery.py`
- `ops/build_paper_gpt56sol_python_parity.py`
- `ops/build_paper_hosted_level_averages.py`
- `ops/build_paper_hosted_reasoning_off.py`

Raw data labels, dictionary keys, and plot labels are not transformed by those calls. Thirteen included fragments emitted by these builders reproduce byte-for-byte from the existing stored JSON records. The comparison export's independent figure renderer is suppressed during this check; no figures are redrawn by it. The remaining four fragments—the historical `frontier_hosted` appendix and its two domain-row tables, plus `discovery_hosted_interpretation`—are stored TeX without a maintained emitter found in `ops`.

The six corresponding JSON sidecars have their existing builder/renderer SHA-256 fields refreshed. The subsequent figure-typography work also refreshes the existing `cohort_builder` binding in the hosted-level averages record and `validation_dependency` binding in the all-level sampling record. Measurements and all other JSON values remain unchanged. The builder computations have identical ASTs after removing formatter imports, the language-exception tuple, and render-output wrapper calls.

Three generic references to the Python programming language remain plain:

- “boxed Python lambda” in the prompt-sensitivity description;
- “Python modulo” in the same section's normalization description;
- “Python `lambda`” in the historical hosted appendix's normalization description.

`validation.json` records fragment counts and hashes, direct-render comparisons, generator AST checks, and the exceptions. `before/` preserves every edited source, fragment, and sidecar; `patches/` contains this task's generator-only diffs. No model, training, bootstrap, grading, or experimental computation was rerun.

After the completed typography comparisons, the live DeepSeek paragraph in `frontier_comparison_20260911_protocol.tex` was independently shortened. That later prose change was preserved; it is recorded separately in `validation.json` and is outside the typography-only comparison. At root’s request, its emitter was subsequently synchronized to the current paragraph. The deployment label remains dynamic, the protocol renders byte-identically, and the TeX was not changed. Only the comparison JSON’s builder hash was refreshed; `protocol_sync_validation.json` records this separately authorized follow-up.

The final metadata cascade refreshes eight dependent source digests across seven current JSON sidecars after the comparison builder hash changes. It includes the dated hosted cohort because the current hosted-level averages record still references it. `metadata_cascade_validation.json` lists every changed JSON path and digest field and verifies the resulting dependency order. Reversing only those digest edits recovers every original JSON value; all 161 existing figure PDFs retain their hashes. No TeX, measurements, intervals, plots, or analyses changed in this cascade.
