# Paper figure inventory

The long [ICLR paper](main.tex) and the short
[MATH-AI paper](mathai2026/main.tex) use the same **six main figures and eight
supplementary figures**. Coverage is frozen to the September 11, 2026 endpoint
audit. Paths below are relative to `paper/`; the workshop copies the same paths
under `paper/mathai2026/` and binds them in its `snapshot.json`.

## Main figures

| Figure | Asset | Model set and evidence |
|---|---|---|
| 1: collapse example | `figures/modecollapse_story.pdf` | Mechanically selected Qwen2.5-3B Graph illustration, Dr.GRPO versus ReplayMaxRL, from its existing fixed source set. Four fixed K=8 draws per checkpoint; illustrative rather than an aggregate. |
| 2: benchmark | `figures/modebench_examples.pdf` | One executed, verified response and canonical key for each of the five domains. |
| 3: method | `figures/verified_support_story.pdf` | MaxRL fresh-sample objective plus uniform verified-key replay; conceptual mechanism. |
| 4: retention and alternatives | `figures/experiment1_retention_comparator_matrix.pdf` | A: all three models × five domains, 74 admissible ReplayDr.GRPO/Dr.GRPO pairs. Falcon Countdown and its common-seed Average use four seeds. B: complete Qwen2.5-0.5B before-training and alternative-method comparisons. The Average is descriptive. |
| 5: MaxRL and replay | `figures/e118_all_scale_factorial_progress.pdf` | Two shared main panels show correctness and distinct modes for all three models, including Qwen2.5-3B MaxRL/ReplayMaxRL. Its dashed descriptive track averages each domain's own paired seeds before equally weighting the five domains (n=5/3/5/3/1 in Graph/Countdown/Python/MathIR/Pantry order). No across-domain common seed set, confidence interval or individual-seed macro paths are implied for that track. Completed tracks retain their common-seed summaries; Falcon Dr.GRPO uses four common seeds. |
| 6: benchmark levels | `figures/modebench_level_admission.pdf` | Frozen Qwen2.5-0.5B admission plus the terminal Level-1/Level-2 comparison for Graph, Countdown, Python and MathIR, all four methods and five seeds. Pantry's endpoint and pair counts are disclosed separately; partial checkpoints do not enter the terminal average. |

## Supplementary figures

| Role | Asset | Evidence boundary |
|---|---|---|
| Verifier-only collapse precheck | `figures/baseline_collapse_precheck.pdf` | All 150 registered Dr.GRPO/GRPO runs, three models × five domains × two objectives × five seeds, with four fixed K=8 draws at pass 0 and pass 8. Pass 0 is measured per arm; retention percentages require initial extra modes above .05. |
| Training-wide retention | `figures/sustained_auc_effects_qwen05b.pdf` | Qwen2.5-0.5B, five domains, five paired seeds, all 17 checkpoints. AUC sensitivity analysis, without inferential domain pooling. |
| Direct alternatives | `figures/direct_comparator_endpoint_effects.pdf` | GRPO 75 pairs; UCPO 50 pairs; sparse RLEP-Dr 47 pairs, including Falcon Python n=2. Missing Qwen-3B alternatives stay blank. Means/intervals require n=5. |
| Direct-alternative trajectories | `figures/direct_baseline_learning_curves_static_strip.pdf` | The same available terminal model/domain blocks and exact prefixes, with supported historical checkpoints and explicit method-specific counts. |
| Semantic-MaxEnt supporting comparison | `figures/verified_support_discovery_two_scale_effects.pdf` | 49 integrity-valid Qwen/Falcon pairs; Falcon Countdown n=4. Exploratory bundled comparison, no Qwen-3B or component-isolated efficacy claim. |
| Replay actuation | `figures/replay_mechanism_telemetry_qwen05b.pdf` | All 25 E78 replay and 25 exact-zero controls, all 3,072 updates. Bank occupancy and dose do not establish identity-level survival. |
| Fixed-bank exemplar scores | `figures/e121_fixed_bank_survival.pdf` | Five Qwen2.5-0.5B Graph seeds, every frozen identity, registered post-freeze visits. Teacher-forced score changes; no causal replay comparison or exact mode-probability claim. |
| MaxRL per-domain detail | `figures/e118_scale_extensions_appendix.pdf` | All three models × correctness/distinct modes × five domains. Qwen-3B MaxRL partial pairs remain visible with exact n; four-arm comparisons use common admissible seeds. |

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

`make -C paper figures` renders all fourteen compiled PDFs. The active paths
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
| 6 and domain companion | `ops/exp_scaling/plot_paper_modebench_levels.py --snapshot ...` |
| Other seven supplementary figures | `ops/exp_scaling/render_paper_retained_figures.py` dispatches to the original plotters using retained JSON only. |

The separate `collect-results` target creates a new endpoint/level audit;
normal figure rendering does not recollect changing or historical campaigns.
`ops/sync_paper_workshop_assets.py` copies current referenced assets and sidecars,
preserves replaced files, and records source hashes and correction history.

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
