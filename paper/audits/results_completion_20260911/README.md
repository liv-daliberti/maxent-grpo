# Completed 3B results refresh — September 11, 2026

The new endpoint census was collected from 20:03:22 to 20:08:24 UTC using the
existing source-admission rules. The earlier paper snapshot ended at 06:45 UTC;
the final 3B completion receipt arrived at 18:54 UTC.

- E118: 150/150 admitted endpoints, 75 paired comparisons, 15 complete blocks.
  Qwen2.5-3B contributes 50/50 endpoints and five complete five-seed domains.
- E119: 90/100 endpoints; Graph, Countdown, Python, MathIR have 80/80.
  Pantry has 10/20, two Dr.GRPO/replay pairs and one MaxRL/replay pair.
- E120-R1: the shared census also admits the completed 3B Pantry frequency
  comparison, bringing the extension to 44/45 endpoints and eight complete
  blocks. Its original September 4 Qwen-0.5B primary analysis is unchanged.

`latest_endpoints.json` retains source hashes and frozen ledger bindings.
No ledger changed during collection. Every efficacy endpoint is exactly
step 3072 with all four registered sampled evaluation draws; completion
receipts are checked separately. No run or missing seed was imputed.

`before/` preserves the pre-refresh publication files; `pre_install/` preserves
exact sources immediately before local edits. `figure6/` and `training_curves/`
retain recollected checkpoint evidence. `workshop_sync/` preserves replaced
workshop assets and source bindings.

The manuscripts, current tables, Figure 5 and its domain appendix now present
the full 3B grid. Its cross-domain display averages all five domains within
each of seeds 70–74 and uses the same complete-cohort visual encoding as the
smaller MaxRL models. Partial historical snapshots remain supported by the
renderer. Level-2 Pantry remains outside the four-domain aggregate.

## Reproduction

Use the repository's `var/seed_paper_eval/paper310/bin/python` environment
(including matplotlib). Activate its bin directory on PATH for the Makefiles.
All paths below are relative to the repository root.

```sh
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/build_paper_latest_results.py --date 2026-09-11 --from-audit paper/audits/results_completion_20260911/latest_endpoints.json --audit-dir paper/audits/results_completion_20260911
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/build_paper_current_campaign_results.py --snapshot paper/results/latest_results_20260911.json --figures
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/build_paper_level2_factorial_contrasts.py --date 2026-09-11 --audit paper/audits/results_completion_20260911/latest_endpoints.json
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/plot_paper_e118_all_scale_progress.py --endpoint-audit paper/audits/results_completion_20260911/latest_endpoints.json
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/plot_paper_modebench_levels.py --snapshot paper/results/modebench_level_comparison_snapshot.json
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/plot_paper_training_curves.py --snapshot paper/results/training_curve_snapshot_20260911.json
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/plot_paper_experiment1_composite.py
```

Do not use a new live census to reproduce this result. The historical primary
weighting result remains frozen. Intervals are unadjusted and descriptive;
Graph/Python 3B correctness intervals include zero and MathIR's extra-mode
increase is small despite a positive nominal interval.

## Validation

- 66 focused regression tests passed.
- The full current-paper evidence contract passed, including exact cohort
  reconstruction and figure/source hashes.
- Long manuscript: eight main-text pages, references on page nine; all eight
  main figures and the line-fill check passed.
- Short manuscript: four content pages, references on page five; all seven
  main and twelve supplementary figures, source hashes, references, and
  overflow checks passed. Its source ZIP was rebuilt and validated.
- `paper/main-body.pdf` contains exactly the first eight pages of the validated
  long paper; extracted text matches those pages byte for byte.
