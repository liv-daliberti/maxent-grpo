# Paper figure inventory

The long [ICLR paper](main.tex) and the short
[MATH-AI paper](mathai2026/main.tex) use the same **nineteen scientific figures**:
eight main and eleven supplementary figures in ICLR, seven main and twelve
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
| 5: MaxRL and replay | `figures/e118_all_scale_factorial_progress.pdf` | Both main panels show all three models. All MaxRL/ReplayMaxRL domains are complete at five paired seeds, including 3B; cross-domain tracks average domains within each common seed and show seed paths. Falcon Dr.GRPO retains four common seeds. |
| 6: benchmark levels | `figures/modebench_level_admission.pdf` | Frozen Qwen2.5-0.5B admission plus the terminal Level-1/Level-2 comparison for Graph, Countdown, Python and MathIR, all four methods and five seeds. Pantry's endpoint and pair counts are disclosed separately; partial checkpoints do not enter the terminal average. |
| 7: hosted correctness and breadth | `figures/hosted_verified_breadth.pdf` | All seven deployments × five domains × three levels. Per-response accuracy and distinct@8 use frozen normalized grading. Opus 5 Python uses its complete revised-wording cohort; all other cells retain original prompts. Every cell retains 128 prompts and 1,024 draws. |
| 8 in ICLR; 17 in MATH-AI appendix: GPT temperature frontier | `figures/gpt56_temperature_curve.pdf` | Four temperatures on 120 fixed prompts with eight draws each, all at reasoning none. Two panels show normalized empirical pass@8 versus distinct@8, overall and by level. The historical medium-reasoning reference markers remain unconnected. |

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
| MaxRL per-domain detail | `figures/e118_scale_extensions_appendix.pdf` | All three models × correctness/distinct modes × five domains. Qwen-3B MaxRL uses all five paired seeds in every domain; four-arm comparisons use common admissible seeds. |
| Hosted Graph comparison | `figures/frontier_comparison_20260911_graph.pdf` | Completed, integrity-audited hosted deployments on identical 128-prompt cells and eight draws. Strict correctness and correct-pair collision with pointwise prompt-bootstrap intervals; uniform references condition on correct draws. Graph was selected after GPT and before the prospective models. Native refusal/filter counts and all five domains remain in the supplement; this is an inference-only comparison. |

| Local prompt-hint control | `figures/modebench_prompt_ablation_local.pdf` | Original versus specified hints removed, identical problems and within-checkpoint sampling. Initial Qwen2.5-0.5B-Instruct plus 24 Dr.GRPO/ReplayDr.GRPO checkpoints; three domains, Levels 2 and 3, 27,648 fresh draws. Paired pass@8 and distinct@8 effects, all cells. Five seeds for Python/MathIR, two for Pantry. Level 3 is transfer. Frontier panel remains pending and is not plotted. |

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
| E118 Level-1 MaxRL replay | 150/150 | 15/15 | Complete: 75 pairs; all five 3B domains n=5. |
| E119 Level-2 factorial | 90/100 | 4/5 | Pantry arm counts D/RD/M/RM=3/4/1/2; Dr replay n=2 (43,46), MaxRL replay n=1 (43). |
| E120-R1 weighting | 44/45 | 8/9 | Qwen-3B Graph n=4 (70–73), Pantry n=5 (70–74); frozen September 4 primary preserved. |

## Builders and reproduction

`make -C paper figures` renders the fifteen training-study PDFs and the new
hosted breadth figure (`ops/plot_paper_hosted_breadth.py`) and GPT temperature
curve (`ops/plot_paper_gpt56_temperature_curve.py`). The original hosted
Graph comparison uses `ops/build_frontier_paper_comparison.py` and its admitted
run directories. The active paths
are explicit in [Makefile](Makefile); the full reproduction and workshop-copy
procedure is in [README.md](README.md). The dated audit is
`audits/results_completion_20260911/latest_endpoints.json`, and Figure 6 reads
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

The active set contains 19 figures, including the local prompt-hint ablation. The Qwen-0.5B AUC sensitivity figure,
bundled semantic/proposal/replay comparison, and scheduler telemetry figure
are archived, alongside their numerical records. Fixed Semantic MaxEnt remains
a defined comparator in the main Figure 4; it is distinct from the removed
bundled experiment. All five accuracy/mode trajectory figures remain active.
The [archive](audits/streamlining_20260911/README.md) preserves the full
pre-edit PDFs, manuscripts, extended proofs, and figure inventory.

## Temperature and retry evidence

The GPT temperature curve is Figure 8 in the ICLR main text and Figure 17 in
the workshop appendix.
Its adjacent JSON binds four aggregate and twelve level-specific none-reasoning
points, four separate medium-reference markers, native-control audits, and
paired analysis. The source is
`artifacts/frontier_temperature_20260911/GPT56_PASS8_FRONTIER.{json,md}`;
the original per-response report is preserved. Empirical pass@8 is the fraction
of prompts with at least one correct answer in eight saved draws; distinct@8
is the mean count of unique correct modes in those draws. All 3,840 responses
and all failure-only groups are retained. Temperature 1.5 is the best observed
aggregate point on both axes, not an established optimum. These conditions
do not establish temperature robustness of the original medium-reasoning
setting. Tables in
`results/gpt56_temperature_curve_20260911_appendix.tex`
report strict and normalized intervals. The two other source-bound appendix
exports are `results/frontier_temperature_20260911.{json,tex}` for the paired
Grok/Kimi conditions and `results/frontier_python_retry_20260911.{json,tex}`
for the three selected Opus 5 retries. Neither changes the seven-model display.

## Prompt-hint control provenance

The local ablation figure and its PNG/JSON companions are exact copies from
`artifacts/modebench_prompt_ablation_20260911/analysis_local_complete_editorial_v2/`. The
result JSON and TeX use the stem `results/modebench_prompt_ablation_20260911`.
The sealed analyzer is `ops/analyze_modebench_prompt_ablation.py`; the paper
checker reconstructs all included statistics and verifies exact copies. This
local-only publication explicitly discloses the omitted frontier panel.

The editorial v2 corrects quotation marks and LaTeX layout only. The original
report is preserved; the editorial audit verifies identical numerical records
and plotted values.
