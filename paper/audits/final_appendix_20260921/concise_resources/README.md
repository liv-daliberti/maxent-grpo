# Concise Appendix S presentation

The replacement follows the user's latest preference: the paper keeps evaluation estimands, dataset/model links, and analysis resources; operational inventories and procedures belong in code and local records.

`replacement.tex` is 167 words. It removes the run-level conflict inventory, checkpoint-selection and partial-cohort procedures, sensitivity sample-count history, source/hash-validation mechanics, and restart-state inventories. It makes no claim that every possible run or checkpoint is complete. Numerical inputs, scientific sample sizes elsewhere in the paper, measured values, and code remain unchanged.

All eight existing labels and all seven public URLs are preserved. `app:initial-provenance` and `app:scheduled-provenance` are aliases under “Evaluation comparisons,” avoiding broken incoming references without retaining separate subsections. The exact contents changes are in `toc_changes.json`: rename the section and three retained subsections; remove the two obsolete initial/scheduled contents entries.

The exact unresolved identities requested for local inspection are in `LOCAL_ONLY_unresolved_evaluations.json` and its Markdown rendering. **Those two files are local audit materials, not manuscript inputs or Overleaf assets.** Both identity sets were verified twice: original conflicts minus the recorded selections, and direct comparison with the current unresolved checkpoint sets. Registered checkpoint filtering prevents off-grid evaluations from being counted as scheduled gaps. A separate whole-run exclusion is recorded in the local JSON. No model inference, measurement calculation, training, bootstrap, or source-record rewrite occurred.

The root agent handles source integration and the final build. `before.tex` records the current live section used as the input to this reduction, and `changes.patch` shows its exact replacement.
