# Experiment coverage and main-body review

Reviewed both current manuscript sources, their recursive TeX includes, comparison registries, figure metadata and archived cut plan. All 27 long-paper and 29 workshop TeX inputs resolve. This is a reporting audit, not a revalidation of every historical experiment or a new endpoint census.

## Current coverage

| Evidence family | Main body | Detailed paper coverage | Finding |
|---|---|---|---|
| DrGRPO/ReplayDrGRPO, three scales | Both | Full endpoint and trajectory comparisons | Present, including 74/75 admissible pairs |
| MaxRL/ReplayMaxRL, three scales | Both | All five domains and trajectories | Present; all 75 pairs |
| Level-2 four-method factorial | Both | Four complete domains, partial Pantry explicitly shown | Present |
| GRPO, UCPO and sparse RLEP-Dr | Main comparator and prose | Supporting endpoint/trajectory figures and scope | Present |
| Fixed Semantic MaxEnt | 0.5B semantic-only contrast in Figure 4 | Estimator and registry present; full semantic/replay factorial not displayed | Reporting gap described below |
| Uniform/frequency replay | Both | Full weighting results | Present; quantitative 0.5B effect could be made more visible in long main |
| Before/after and paired concentration | Both | All 135 saved-output contrasts | Central result and its limitations already summarized |
| Fixed-bank exemplar scores | Limitation in both conclusions | Full study and declining tail | Appropriate placement |
| Seven hosted deployments | Both | Original protocols, normalization, revised Opus wording and retry diagnostics | Present |
| Temperature controls | GPT sweep in long main; brief workshop pointer | GPT, Grok and Kimi conditions | Main text omits the contrasting Grok/Kimi takeaways |
| Local prompt-hint ablation | Neither main body | Both appendices | Important main-body omission |
| Local 64-sample discovery | Neither main body | Both appendices; 110,592 complete local outputs | Important main-body omission |
| Hosted prompt/discovery follow-ups | No completed full-panel main claim | Pending panels disclosed; collection status separate | Await complete validated results |

## Recommended main-text additions

1. Summarize the local 64-sample result: trained MathIR remains highly concentrated, while Pantry and neutral Python replay show meaningful late discovery. This directly qualifies whether K=8 understates alternatives. Say local Qwen 0.5B, not frontier robustness.
2. Summarize Level-2/3 prompt dependence: deleting the Python strategy/example guidance severely changes correctness; MathIR concentration persists under hint removal. This limits a purely diversity-based reading of raw breadth changes. Do not generalize this wording intervention to Level-1 prompts.
3. Add a short temperature qualification beside the GPT example: the Grok comparison has no clear breadth gain and Kimi's higher-temperature condition has severe truncation. Avoid implying a general temperature remedy.

These can be prose with appendix references while preserving the user's existing main figure sequence. A new main figure is not required. The detailed training curves, per-seed tables, full prompt strings, retry logs and full proofs can stay supplementary.

## Actual reporting gap to resolve

`paper/figures/fixed_semantic_factorial_effects.json` retains complete historical 0.5B and Falcon factorial records: semantic-only minus DrGRPO, semantic-plus-replay minus replay, and their interaction. The current Figure 4 uses only the 0.5B semantic-only contrast. Neither manuscript recursively includes the full factorial figure/table; the separate 3B seed-70 table also remains an artifact. A registry entry and a method definition do not display these results.

Restore a compact supporting summary of the useful semantic comparisons only after checking them against CURRENT source admission. In particular, the historical Falcon Countdown factorial lists five seeds, whereas the current primary replay comparison excludes a conflicted endpoint; do not blindly republish old five-seed inference. This review has not rerun that source audit.

## Intentional exclusions

The approved `paper/STREAMLINING_REVIEW_20260911.md` moved the bundled E112-R1 discovery experiment, AUC sensitivity, scheduler telemetry and redundant seed tables into the archive. The bundled contrast changes multiple components, so it is not an isolated mechanism test. Those cuts are intentional, and this review does not recommend reversing them wholesale. DAPO lacks a standardized comparable endpoint; unfinished Level-2/3 registrations cannot be represented as completed evidence.

## New work requested during this review

The proposed Pantry adaptation, cross-model mixture, and coarser-key tests are not yet results in either paper. Protocols are in `artifacts/modebench_inference_followups_20260911/`. Pantry adaptation directly addresses the main conclusion's currently untested downstream-utility claim. Cross-model mixtures and coarser-key sensitivity can largely reuse saved outputs. Include outcomes only after completed analysis, including adverse or null findings.
