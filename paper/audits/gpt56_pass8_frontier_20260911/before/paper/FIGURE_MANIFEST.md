# Paper figure inventory

The long [ICLR paper](main.tex) and the short
[MATH-AI paper](mathai2026/main.tex) use the same **eighteen scientific figures**:
eight main and ten supplementary figures in ICLR, seven main and eleven
supplementary figures in MATH-AI. Training coverage is frozen to the September
11, 2026 endpoint audit; hosted figures bind their separate retained response
cohorts. Paths below are relative to `paper/`; the workshop copies the same
paths under `paper/mathai2026/` and binds them in its `snapshot.json`.

## Main figures

| Figure | Asset | Model set and evidence |
|---|---|---|
| 1: collapse example | `figures/modecollapse_story.pdf` | Mechanically selected Qwen2.5-3B Graph illustration, Dr.GRPO versus ReplayMaxRL, from its existing fixed source set. Four fixed K=8 draws per checkpoint; illustrative rather than an aggregate. |
| 2: benchmark | `figures/modebench_examples.pdf` | One executed, verified response and canonical key for each of the five domains. |
| 3: method | `figures/verified_support_story.pdf` | MaxRL fresh-sample objective plus uniform verified-key replay; conceptual mechanism. |
| 4: retention and alternatives | `figures/experiment1_retention_comparator_matrix.pdf` | A: all three models × five domains, 74 admissible ReplayDr.GRPO/Dr.GRPO pairs. Falcon Countdown and its common-seed Average use four seeds. B: complete Qwen2.5-0.5B before-training and alternative-method comparisons. The Average is descriptive. |
| 5: MaxRL and replay | `figures/e118_all_scale_factorial_progress.pdf` | Two shared main panels show correctness and distinct modes for all three models, including Qwen2.5-3B MaxRL/ReplayMaxRL. Its dashed descriptive track averages each domain's own paired seeds before equally weighting the five domains (n=5/3/5/3/1 in Graph/Countdown/Python/MathIR/Pantry order). No across-domain common seed set, confidence interval or individual-seed macro paths are implied for that track. Completed tracks retain their common-seed summaries; Falcon Dr.GRPO uses four common seeds. |
| 6: benchmark levels | `figures/modebench_level_admission.pdf` | Frozen Qwen2.5-0.5B admission plus the terminal Level-1/Level-2 comparison for Graph, Countdown, Python and MathIR, all four methods and five seeds. Pantry's endpoint and pair counts are disclosed separately; partial checkpoints do not enter the terminal average. |
| 7: hosted correctness and breadth | `figures/hosted_verified_breadth.pdf` | All seven deployments × five domains × three levels. Per-response accuracy and distinct@8 use frozen normalized grading. Opus 5 Python uses its complete revised-wording cohort; all other cells retain original prompts. Every cell retains 128 prompts and 1,024 draws. |
| 8 in ICLR; 17 in MATH-AI appendix: GPT temperature tradeoff | `figures/gpt56_temperature_curve.pdf` | Four temperatures × three levels on 120 fixed prompts with eight draws each, all at reasoning none. Normalized accuracy versus distinct@8. The three historical medium-reasoning reference markers remain unconnected. |

## Supplementary figures

| Role | Asset | Evidence boundary |
|---|---|---|
| Verifier-only collapse precheck | `figures/baseline_collapse_precheck.pdf` | All 150 registered Dr.GRPO/GRPO runs, three models × five domains × two objectives × five seeds, with four fixed K=8 draws at pass 0 and pass 8. Pass 0 is measured per arm; retention percentages require initial extra modes above .05. |
| Primary accuracy trajectories | `figures/factorial_training_curves_pass8.pdf` | Three models × five domains, Dr.GRPO/ReplayDr.GRPO/MaxRL/ReplayMaxRL. Fixed domain-specific terminal paired cohorts match Figure 5; all four fixed K=8 draws are required at each displayed checkpoint. |
| Primary verified-mode trajectories | `figures/factorial_training_curves_distinct8.pdf` | Identical seeds and checkpoints to primary accuracy, now distinct@8. Per-domain axes shared across scales, exact n, seed-range bands and gaps for unavailable checkpoints. |
| Level-2 accuracy and modes | `figures/level2_factorial_training_curves.pdf` | Qwen2.5-0.5B, two metric rows × five domains, all four methods. Four complete domain factorials; Pantry Dr/replay n=2 paired, available MaxRL/replay histories explicitly individual and unpaired. |
| Direct alternatives | `figures/direct_comparator_endpoint_effects.pdf` | GRPO 75 pairs; UCPO 50 pairs; sparse RLEP-Dr 47 pairs, including Falcon Python n=2. Missing Qwen-3B alternatives stay blank. Means/intervals require n=5. |
| Direct-alternative modes | `figures/direct_baseline_learning_curves_static_strip.pdf` | UCPO/RLEP terminal trajectories with Dr/replay references at 0.5B and 1B only; exact method-specific counts and seed ranges. |
| Direct-alternative accuracy | `figures/direct_baseline_learning_curves_pass8.pdf` | Same snapshot, populations and checkpoints as the companion distinct@8 plot; pass@8 axes span [0,1]. |
| Fixed-bank exemplar scores | `figures/e121_fixed_bank_survival.pdf` | Five Qwen2.5-0.5B Graph seeds, every frozen identity, registered post-freeze visits. Teacher-forced score changes; no causal replay comparison or exact mode-probability claim. |
| MaxRL per-domain detail | `figures/e118_scale_extensions_appendix.pdf` | All three models × correctness/distinct modes × five domains. Qwen-3B MaxRL partial pairs remain visible with exact n; four-arm comparisons use common admissible seeds. |
| Hosted Graph comparison | `figures/frontier_comparison_20260911_graph.pdf` | Completed, integrity-audited hosted deployments on identical 128-prompt cells and eight draws. Strict correctness and correct-pair collision with pointwise prompt-bootstrap intervals; uniform references condition on correct draws. Graph was selected after GPT and before the prospective models. Native refusal/filter counts and all five domains remain in the supplement; this is an inference-only comparison. |

## Hosted measurement and prompt conditions

Both main texts put Figure 7 after the training results. Its sidecar identifies
all 105 cells, both metrics, and each original or revised prompt cohort. The
complete 3,072-response Opus 5 Python follow-up is used with the same frozen
formatting normalizer as every other displayed cell. Failed responses remain
in the denominator. Original-protocol tables, refusals and repair comparisons
remain in the appendix. The obsolete introduction tables are archived under
`audits/hosted_reframe_20260911/retired_includes/`.

## Additional current assets

The dated [`current_campaign_results_20260911`](results/current_campaign_results_20260911.md)
report has CSV, JSON, manuscript tables, and a standalone overview PDF/PNG.
It reports all E118/E119/E120 model/domain comparisons, including incomplete
blocks. The domain-resolved Figure 6 companion,
`figures/modebench_level_terminal_by_domain.pdf`, is a standalone diagnostic;
it does not add another compiled figure. The level snapshot and every empirical
figure's adjacent JSON retain numerical provenance; E121 uses
`results/e121_fixed_bank_survival.json` and Figure 1 uses
`var/artifacts/paper_graph_collapse_toy.json` in the repository root.

| Campaign | Admitted endpoints | Complete blocks | Partial coverage |
|---|---:|---:|---|
| E118 Level-1 MaxRL replay | 138/150 | 12/15 | 67 pairs; Qwen-3B Countdown 3, MathIR 3, PantryPlan 1. |
| E119 Level-2 factorial | 88/100 | 4/5 | Pantry terminal arm counts D/RD/M/RM=3/3/1/1; Dr replay pairs n=2 (43,46), MaxRL replay pairs n=0. |
| E120-R1 weighting | 43/45 | 7/9 | Qwen-3B Graph n=4 (70–73), Pantry n=4 (71–74); frozen September 4 primary preserved. |

## Builders and reproduction

`make -C paper figures` renders the fifteen training-study PDFs and the new
hosted breadth figure (`ops/plot_paper_hosted_breadth.py`) and GPT temperature
curve (`ops/plot_paper_gpt56_temperature_curve.py`). The original hosted
Graph comparison uses `ops/build_frontier_paper_comparison.py` and its admitted
run directories. The active paths
are explicit in [Makefile](Makefile); the full reproduction and workshop-copy
procedure is in [README.md](README.md). The dated audit is
`audits/results_refresh_20260911/latest_endpoints.json`, and Figure 6 reads
`results/modebench_level_comparison_snapshot.json`.

| Figure(s) | Builder |
|---|---|
| 1 | `ops/plot_paper_collapse_toy.py` |
| 2 | `ops/plot_paper_modebench_examples.py` |
| 3 | `ops/plot_paper_support_story.py` |
| 4 | `ops/exp_scaling/plot_paper_experiment1_composite.py` |
| 5 and its appendix | `ops/exp_scaling/plot_paper_e118_all_scale_progress.py --endpoint-audit ...` |
| Primary/Level-2 training curves | `ops/exp_scaling/plot_paper_training_curves.py --snapshot paper/results/training_curve_snapshot_20260911.json` |
| 6 and domain companion | `ops/exp_scaling/plot_paper_modebench_levels.py --snapshot ...` |
| Other five supplementary figures | `ops/exp_scaling/render_paper_retained_figures.py` dispatches to the original plotters using retained JSON only. |

The separate `collect-results` target creates a new endpoint/level audit;
normal figure rendering does not recollect changing or historical campaigns.
`ops/sync_paper_workshop_assets.py` copies current referenced assets and sidecars,
preserves replaced files, and records source hashes and correction history.

## Training-curve source contract

`results/training_curve_snapshot_20260911.json` retains both accuracy and mode
metrics, fixed paired populations, source-prefix hashes, draw-level line
provenance, and explicit unavailable checkpoints. It covers all three Level-1
models and the registered Qwen2.5-0.5B Level-2 model. Each objective pair keeps
the same terminal seeds across its curve. Means require all cohort members;
gaps are not joined, carried forward or filled from another seed. Partial
unpaired Pantry histories are supplementary observations only.

The direct-alternative modes JSON remains the shared numerical source for both
of its metrics at 0.5B and 1B only. Its raw records preserve historical before-training references
used by Figures 4 and 5; the plotted core subset applies the same admissible
Dr/replay pairing. Numerical snapshot collection and PDF rendering remain
separate so future builds do not silently acquire new observations.

## Historical assets and evidence rules

Retired omnibus frontiers, Qwen-only MaxRL plots, adaptive/open-bank/dose
monitors, DAPO progress charts, and old Semantic-MaxEnt figure families stay on
disk as provenance. Their existence does not make them compiled evidence.
The August 26 figure audit and earlier inventory are preserved under
`audits/figure_refresh_20260911/organization_before/`.

Missing observations are not imputed. A complete efficacy block requires all
five registered paired terminal seeds; partial blocks retain exact n and no
five-seed interval. The Falcon Countdown ReplayDr.GRPO seed-59 exclusion
applies everywhere its endpoint is reused. Factorial contrasts use the same
admissible four-arm seed intersection. Cross-domain means are descriptive;
no pooling across models or levels licenses a confirmatory claim. Figure 5's
Qwen-3B MaxRL track is explicitly descriptive: it is the equal-domain average
of five separately paired domain means, with varying n and no interval or
cross-domain seed paths. Intervals elsewhere are nominal and do not adjust for
repeated looks or multiple comparisons.

## September 11 streamlining

The active set contains 18 figures. The Qwen-0.5B AUC sensitivity figure,
bundled semantic/proposal/replay comparison, and scheduler telemetry figure
are archived, alongside their numerical records. Fixed Semantic MaxEnt remains
a defined comparator in the main Figure 4; it is distinct from the removed
bundled experiment. All five accuracy/mode trajectory figures remain active.
The [archive](audits/streamlining_20260911/README.md) preserves the full
pre-edit PDFs, manuscripts, extended proofs, and figure inventory.

## Temperature and retry evidence

The GPT temperature curve is Figure 8 on ICLR main page 9 and Figure 17 on
workshop appendix page 30.
Its adjacent JSON binds all twelve none-reasoning points, three separate medium
reference markers, native-control audits, and paired analysis. All 3,840 new
responses are retained. These conditions do not establish temperature robustness
of the original medium-reasoning setting. Tables in
`results/gpt56_temperature_curve_20260911_appendix.tex`
report strict and normalized intervals. The two other source-bound appendix
exports are `results/frontier_temperature_20260911.{json,tex}` for the paired
Grok/Kimi conditions and `results/frontier_python_retry_20260911.{json,tex}`
for the three selected Opus 5 retries. Neither changes the seven-model display.
