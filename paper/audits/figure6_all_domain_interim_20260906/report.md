# Figure 6 all-domain interim comparison — 2026-09-06

Panel B now compares Level 1 and Level 2 for Dr.GRPO, ReplayDr.GRPO, MaxRL, and ReplayMaxRL on pass@8 and distinct@8. It shows descriptive means across all five domains. Open slate and filled teal markers denote Level 1 and Level 2 consistently in both panels; method names label the rows. Panel A retains the same frozen admission values.

For every registered domain and seed, selection uses the latest observed checkpoint available for all eight level-by-method combinations, with four valid sampled K=8 draws, temperature 1, and 128 evaluation prompts per draw. Exact duplicate records are deduplicated; conflicting retries are refused. This availability rule was fixed before inspecting the selected effects. Because Pantry has no valid common trained checkpoint, the rule permits its actually measured step 0; there is no substitution of admission values or zero imputation. The figure explicitly labels Pantry as initial-checkpoint-only.

There are 21 selected domain-seed cells. Every series uses the same selected cells and steps. The mean is formed over draws within a cell, then seeds within each domain, then equally over all five domains. Counts differ across domains, so there are no pooled seed points or confidence intervals.

| Domain | Seeds | Steps in seed order | Pass range |
|---|---|---|---|
| countdown | 43, 44, 45, 46, 47 | 960, 192, 96, 192, 96 | 0.25–2.5 |
| graph_coloring | 43, 44, 45, 46, 47 | 3072, 3072, 3072, 3072, 3072 | 8–8 |
| mathir | 43, 44, 45, 46, 47 | 3072, 3072, 1920, 2304, 1632 | 4.25–8 |
| pantry_plan | 45 | 0 | 0–0 |
| python_factors | 43, 44, 45, 46, 47 | 1728, 1248, 3072, 960, 3072 | 2.5–8 |

| Method | Level 1 pass@8 / distinct@8 | Level 2 pass@8 / distinct@8 |
|---|---|---|
| drgrpo | 0.443672 / 0.774687 | 0.356406 / 0.373437 |
| replay_drgrpo | 0.726172 / 1.369219 | 0.613203 / 0.788750 |
| maxrl | 0.501250 / 0.870234 | 0.451016 / 0.485859 |
| replay_maxrl | 0.738359 / 1.399141 | 0.593984 / 0.736250 |

The terminal Graph example is preserved exactly as separate evidence. Terminal-progress metadata is refreshed from the frozen Level 2 snapshot: 50 exact terminal cells (Graph 20, MathIR 14, Python 14, Countdown 2, Pantry 0). These terminal counts do not define the displayed interim average.

Frozen selected data and all 200 availability records: `paper/results/modebench_level_comparison_snapshot.json`, mirrored under the workshop results directory. The compact snapshot contains 672 selected draw records plus 200 separate terminal draw records, complete per-file prefix hashes, exact line origins, and references to the full frozen inventories. Rendering never rereads live evaluation logs. The paper contract independently rebuilds the selection and averages from this snapshot.

Validation: 16 tests passed (11 comparison/aggregation tests and 5 coverage-reader integrity tests). The independent audit rechecked all selected metrics, metadata, origins, checkpoint intersections, and equal-domain means against both frozen inventories. Admission values and the terminal Graph record are unchanged. Figures 4, 5, and 12 are byte-identical to the prior completed work. The new figure measures 496.499 × 135.976 points, versus 497.366 × 136.241 points before, preserving the workshop height budget.

Visual evidence: `figure6_before.png` and `figure6_after.png`. Numerical/provenance evidence: `validation.json`, `independent_validation.json`, and `selection_summary.json`. Reproducers: `freeze_level1.py` (one-time inventory), `compose_snapshot.py` (immutable snapshot composition), and `ops/exp_scaling/plot_paper_modebench_levels.py` (frozen data only). Root owns final captions, Makefile/manifest dependencies, and manuscript builds.
