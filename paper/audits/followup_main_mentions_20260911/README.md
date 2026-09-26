# Main-body follow-up mentions

Both paper main bodies now summarize the completed local prompt-guidance and fresh 64-response follow-ups and reference their respective appendices. The scope is selected Qwen2.5-0.5B Level-2/3 Python, MathIR and Pantry problems. Removing guidance and the executable Python example lowers Python correctness; larger sampling budgets retain strong concentration in most trained MathIR conditions while revealing further alternatives in Pantry and neutral-prompt Python replay. The parent text explicitly separates correctness changes from redistribution among correct solutions.

These edits do not claim completion of the separately collected hosted follow-ups. Existing detailed analysis, numerical tables and figures remain supplementary. The initial sources are preserved under `before/`, and `main_body_checks.json` records both main-text references and scope checks.

The workshop initially exceeded its four-page limit. Shortening repeated prose recovered four content pages with all seven main figures and eighteen supplementary figures. `workshop_build_tightened.log` records successful standalone validation and source packaging. The complete parent rebuild passed: `parent_build_full.log` authenticates the evidence, compiles all references, verifies nine main pages with seven figures and the hosted summary table, and passes the line-fill check. Both rendered main bodies contain the new findings (`rendered_mentions.json`).

This addresses the two missing main-body mentions identified in the earlier coverage audit. The separate recommendation about the full supporting Semantic-MaxEnt factorial is outside these two edits and is not marked complete.
