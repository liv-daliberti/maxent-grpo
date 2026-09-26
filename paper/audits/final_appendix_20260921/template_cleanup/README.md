# TeX-template presentation cleanup

Updated 10 Python scripts with 33 exact source substitutions to match the concise appendix prose applied by the parent agent. Changes remove display-only seed identifiers, retry and missing-response inventories, hash-ranking descriptions and RNG identifiers; aggregate sample sizes, statistical eligibility, uncertainty methods, compute limits and numerical table values remain.

## Validation

- Every source diff is reproduced exactly by `source_replacements.json`; the saved `before/` files are the actual pre-edit source snapshots.
- All 10 scripts parse. Every top-level AST node outside the explicitly changed presentation functions is identical.
- Eight isolated before/after TeX renderers used existing numerical records. Their outputs differ only by the requested manuscript replacements and match current active TeX after whitespace normalization. The supplied records are unchanged.
- The inference-followups and portfolio-withdrawals builders contain rendering and analysis in the same functions. They were not executed. Their ASTs are identical after masking only TeX-string values and the inference-cost table's visible checkpoint-label expression.
- No full builders, model calls, experiment loops, bootstraps, figure renderers or numerical-record writes were performed.
- Existing domain-name formatting wrappers are unchanged; the isolated outputs retain the manuscript's `\texttt` domain formatting.

`validation.json` records the checks, functions and hashes. `changes.patch`, individual `diffs/`, and the before/after isolated TeX renderings provide reviewable evidence. `validate_templates.py` repeats the checks without running builders.

## Scope notes

The conditional-concentration and frontier-hosted appendix narrative inputs have no corresponding prose-renderer references in `ops`; their separate numerical-table builders were untouched. An external concurrent change had already synchronized the frontier-comparison protocol paragraph before its saved baseline; that change was preserved. The two remaining sensitivity-paragraph edits were applied by this task.

Five existing JSON bindings identify these edited presentation-code files. `renderer_binding_audit.json` records their exact fields and updated hashes for the parent's packaging step. No JSON binding or numerical record was modified by this task.

Files changed:

- `ops/build_paper_semantic_current_summary.py`
- `ops/analyze_modebench_prompt_ablation.py`
- `ops/analyze_modebench_discovery_curves.py`
- `ops/build_paper_gpt56_all_levels32_discovery.py`
- `ops/build_paper_inference_followups.py`
- `ops/build_paper_portfolio_withdrawals.py`
- `ops/build_frontier_paper_comparison.py`
- `ops/build_frontier_temperature_paper.py`
- `ops/plot_paper_gpt56_temperature_curve.py`
- `ops/build_paper_hosted_reasoning_off.py`
