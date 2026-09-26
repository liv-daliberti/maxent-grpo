# There’s More Than One Way: Mode Collapse in RLVR & ModeBench

The [ICLR manuscript](main.pdf) opens with diverse successful model outputs,
including explanations, plans, and programs. Its hosted comparison now uses a
compact [per-level averages table](results/hosted_level_averages_20260911.tex)
for all seven deployments: empirical `pass@8` and mean `distinct@8`, weighting
the five domains equally. The complete domain plot is retained in the hosted
appendix. ICLR has seven main and eighteen supplementary figures; all 25
figures remain available. The [main-only review copy](main-body.pdf) and
[full review copy](main-with-figures.pdf) mirror the compiled paper.

The table uses the existing evaluated reasoning settings and the same complete
cohorts as the former main plot, including revised Opus 5 Python wording.
Regenerate it with `python ops/build_paper_hosted_level_averages.py` from the
repository root. The separately authorized reasoning-off comparison uses 32
matched prompts per domain and level, eight draws each, and original prompts
for a paired comparison against the original reasoning-enabled responses.
Its [run record](../artifacts/hosted_reasoning_off_32_20260911_v2/README.md)
tracks collection and provider-control checks. Opus 5 is paused after a
response reported thinking tokens despite disabled reasoning; its incomplete
cohort receives no reasoning-off scores. The other six deployments continue.
These new runs do not yet replace the completed results in the main table.

The [workshop manuscript](mathai2026/main.pdf) restores its original seven main figures in their original positions. The corrected measurement definitions, conditional theory, concentration findings, and interpretation limits remain. Four newer quantitative plots support the appendix analyses. Both versions retain all 25 figures, including the two later sampling-budget plots: seven main and eighteen supplementary in each version.

The [restoration record](audits/figure_restoration_20260911/README.md) documents the user's visual preference, preserved sources, and validation. The [figure inventory](FIGURE_MANIFEST.md), [coverage audit](FIGURE_DATA_AUDIT.md), and [conditional-concentration report](results/conditional_concentration_20260911/report.md) identify each figure's evidence. All eighteen formal statement/proof blocks remain unchanged.

## Model and data archive

The [ModeBench dataset](https://huggingface.co/datasets/od2961/ModeBench) publishes
[Level 1](https://huggingface.co/datasets/od2961/ModeBench#level-1),
[Level 2](https://huggingface.co/datasets/od2961/ModeBench#level-2), and
[Level 3](https://huggingface.co/datasets/od2961/ModeBench#level-3) with per-domain
configurations, original splits, source identities, and a
[loading guide](https://huggingface.co/datasets/od2961/ModeBench#loading).
The local training comparisons cover Levels 1 and 2. The separate hosted-model
inference-only comparison evaluates all five domains at Levels 1, 2 and 3.

The [public research archive](https://huggingface.co/od2961/maxent-grpo-models)
organizes completed models and supporting evidence by experiment, model, domain,
method, and seed. Both manuscripts link the relevant model groups beside their
results and include a reproducibility map.

- [Core replay and MaxRL comparisons](https://huggingface.co/od2961/maxent-grpo-models/blob/main/EXPERIMENTS.md#core-replay)
- [GRPO, UCPO, and RLEP-Dr baselines](https://huggingface.co/od2961/maxent-grpo-models/blob/main/EXPERIMENTS.md#direct-comparators)
- [Weighting and semantic ablations](https://huggingface.co/od2961/maxent-grpo-models/blob/main/EXPERIMENTS.md#frequency-ablation)
- [Mechanism-study artifacts](https://huggingface.co/od2961/maxent-grpo-models/blob/main/EXPERIMENTS.md#mechanism-studies)
- [Data, results, and reproducibility](https://huggingface.co/od2961/maxent-grpo-models/blob/main/EXPERIMENTS.md#reproducibility)

Use each model's immutable revision and the [restore instructions](https://huggingface.co/od2961/maxent-grpo-models/blob/main/RESTORE.md).
The archive distinguishes available weights from preserved scientific records.
Each frozen figure or table determines its own admitted seed set and checkpoint;
newer model availability does not change those analysis choices.

## Current evidence: September 11, 2026, 20:08 UTC

**The 3B MaxRL / ReplayMaxRL grid is complete: 50/50 terminal endpoints,
five domains, both methods, and five matched seeds per domain.** See the
[completed 3B and Level-2 results](results/completed_3b_and_level2_20260911.md)
for the focused numerical presentation.

| Campaign | Admitted endpoints | Complete paired blocks | Coverage |
|---|---:|---:|---|
| E118 MaxRL / ReplayMaxRL, Level 1 | 150/150 | 15/15 | Every domain at Qwen2.5-0.5B, Falcon3-1B, and Qwen2.5-3B is complete. |
| E119 four-method factorial, Level 2 | 90/100 | 4/5 | Graph, Countdown, Python and MathIR complete (80/80); PantryPlan has 10/20 endpoints. |
| E120-R1 fresh-frequency replay weights | 44/45 | 8/9 | Qwen2.5-3B Pantry now has five pairs; Graph remains at four. |

E118 has all 75 MaxRL/ReplayMaxRL pairs. The original main factorial figure again shows absolute endpoint tracks and individual-seed paths at all three scales. The newer domain-specific effect plot remains supplementary. At 3B, mean pass@8 rises from .547 to
.701 and distinct@8 from .744 to 1.107. Every domain has positive point
estimates on both metrics; Graph and Python correctness intervals include
zero. The corrected Dr.GRPO/ReplayDr.GRPO core retains its 74 admissible
pairs, including the registered four-seed Falcon Countdown exclusion.

Level 2 supports four complete five-seed domain factorials at Qwen2.5-0.5B.
The main level figure presents only the replay terminal results, with population and protocol limits stated. Construction and admission are described in benchmark Methods. The requested five-domain frozen-base grid (four Qwen sizes at every Level 1–5) is being collected separately; its registry is in `../artifacts/modebench_base_level_grid_20260911/`. Levels 4–5 have no admitted test release yet, so a complete grid is not presented as measured evidence. The supplementary within-Level-2 effect plot uses the common four-arm seed intersection. PantryPlan has Dr.GRPO 3, ReplayDr.GRPO 4, MaxRL 1, and
ReplayMaxRL 2 endpoints. Its paired intersections are two for Dr.GRPO and
one for MaxRL; only seed 43 is common to all four methods. These partial
results remain outside the completed four-domain aggregate. No additional
Level-2 model is implied.

The earlier 06:45 UTC snapshot is preserved under
[audits/results_completion_20260911/before/](audits/results_completion_20260911/before/).
The fresh endpoint audit ran from 20:03 to 20:08 UTC and admits only the
registered step-3072 evaluation. The previous partial 3B counts describe
that earlier snapshot, not unfinished runs in the current paper.

Direct alternatives retain all 75 GRPO pairs, 50 smaller-model UCPO pairs,
and 47 sparse RLEP-Dr pairs. DAPO has no standardized pass@8/breadth endpoint
for an efficacy comparison. Current Level-3 campaigns provide no terminal
results for these figures.

Every effect uses the exact admissible paired seed intersection. Partial
blocks show their observed `n` and receive no five-seed interval. Cross-domain
summaries are descriptive. The original E120 Qwen2.5-0.5B primary analysis
remains bound to its September 4 input; its five-domain extra-mode effect is
+.318 [.272, .358]. Current additions do not replace that analysis.

## Appendix organization

Both supplements open with the claim/evidence map, measurement identities, and complete conditional theory. Benchmark construction and verbatim prompts precede the implemented algorithm and controlled results. Longitudinal concentration and exemplar-score diagnostics follow, then the separate hosted observations, prompt interventions, and reproducibility. The original examples, schematic, comparator matrix, absolute endpoints, admission display, and temperature figure are restored to the main; the new quantitative views accompany supporting analyses.

The shared formal statements and proofs are preserved exactly. Source and numerical checks validate every figure; editorial checks protect the restored main sequence and supporting placement. Hosted observational limits, collision eligibility, simultaneous correctness changes, incomplete cohorts, and neural-theory boundaries remain visible in the main text.

## Appendix training curves

Both papers include full training trajectories for **accuracy (`pass@8`) and
verified modes (`distinct@8`)**. The primary figures use the same three-model
rows and four methods as the central factorial: Qwen2.5-0.5B, Falcon3-1B and Qwen2.5-3B,
each with Dr.GRPO, ReplayDr.GRPO, MaxRL and ReplayMaxRL. Separate supporting
plots show accuracy and modes for the UCPO/RLEP comparisons at 0.5B and 1B. A two-metric Level-2 figure covers the
Qwen2.5-0.5B factorial and labels incomplete Pantry histories explicitly.

Every primary curve uses its objective pair's fixed terminal seed cohort;
accuracy and modes share checkpoints and seeds. The source snapshot records
missing evaluations and retry exclusions. Missing checkpoints leave gaps, and
bands show seed ranges rather than confidence intervals. The retained fixed-bank
score study keeps its original population; historical AUC and replay telemetry
displays are preserved in the streamlining archive.

Render the three primary/Level-2 plots from the frozen snapshot with:

```sh
python ops/exp_scaling/plot_paper_training_curves.py \
  --snapshot paper/results/training_curve_snapshot_20260911.json
```

To recollect primary histories for this completion audit, use a new output audit
directory (do not reuse a prior coverage file):

```sh
python ops/exp_scaling/build_paper_training_curve_snapshot.py \
  --endpoint-audit paper/audits/results_completion_20260911/latest_endpoints.json \
  --audit-directory paper/audits/results_completion_20260911/training_curves
```

The full `make -C paper figures` target also rebuilds both supporting trajectory
metrics. See the [figure inventory](FIGURE_MANIFEST.md) for all five curve plots.

## Reproduce the figures and results

From the repository root:

```sh
make -C paper latest-results      # rebuild dated tables/report from retained audit
make -C paper figures             # render current main and supporting figures
make -C paper                    # check and build long paper
```

The Makefile pins `ANALYSIS_DATE=2026-09-11`,
`ENDPOINT_AUDIT=audits/results_completion_20260911/latest_endpoints.json`, and
`LEVEL_SNAPSHOT=results/modebench_level_comparison_snapshot.json`.
The retained absolute endpoint and construction figures use those records. The new concentration, factorial, weighting, and within-Level-2 figures read the frozen sources listed in the figure inventory.
`figures-supporting` calls
[`render_paper_retained_figures.py`](../ops/exp_scaling/render_paper_retained_figures.py),
which uses retained JSON records and leaves their numerical contents unchanged.
The original Graph illustration, now supplementary, uses its fixed log set and selection rule;
the benchmark and method diagrams have no campaign endpoints to refresh.

Data collection is explicit. For a later analysis, select a new date and audit
directory, then run `make -C paper collect-results ANALYSIS_DATE=YYYY-MM-DD`.
The target refuses an existing endpoint audit. Review coverage and update both
manuscripts before regenerating their dated report and figures. Reproduction
never invokes the historical live E120 frequency builder or recollects retired
Semantic-MaxEnt experiments.

## Synchronize the short paper

Edit [mathai2026/main.tex](mathai2026/main.tex) and
[mathai2026/appendix.tex](mathai2026/appendix.tex) for their distinct narrative.
Then review the asset copy before applying it:

```sh
python ops/sync_paper_workshop_assets.py --date 2026-09-11 \
  --audit-directory paper/audits/narrative_reorganization_20260911/workshop_next \
  --reason 'Synchronize the reorganized papers while preserving frozen numerical evidence'
```

Add `--apply` to copy the reviewed assets, preserve replaced files and the old
snapshot, and record the new parent/workshop hashes. Choose a new audit directory
for each subsequent synchronization. `make -C paper sync-workshop` runs this
apply step with the Makefile defaults. Then:

```sh
make -C paper/mathai2026 bundle
```

The workshop build validates four main-text pages, all twenty-five figures,
references starting on page five, source hashes, and the unchanged anonymous
style. A failed build preserves the last validated PDF. The source ZIP is built
and checked independently. See the [workshop README](mathai2026/README.md).

## What the paper measures

ModeBench's five domains are Graph coloring, Countdown, executable Python
factors, MathIR and PantryPlan. Each verifier executes an answer and assigns a
canonical correct-outcome key. `pass@8` measures whether eight samples contain
a correct response; `distinct@8` counts distinct correct keys; their difference
measures correct modes beyond the first. Lexical variation alone does not count.

ReplayMaxRL combines MaxRL's fresh-sample objective with replay that stores
verified exemplars by key and revisits the bank uniformly. The idealized theory
protects recurrently replayed discovered modes under its stated update and
coverage assumptions. It does not certify current AdamW/PPO updates, establish
coverage of every correct mode, or convert teacher-forced scores into exact
canonical-mode sampling probabilities. E121's fixed-bank figure therefore
reports exemplar-score trajectories, not a causal replay comparison.

The ICLR build also checks the byte-exact official July 28, 2026 style archive,
domain prompts, manuscript evidence contract, and PDF line filling. The original
organization documents and superseded figures are preserved under
[audits/figure_refresh_20260911/organization_before](audits/figure_refresh_20260911/organization_before/).

## Hosted deployment comparison

The hosted overview is restored to Figure 7, after the training results, in
both manuscripts. It shows per-response accuracy and verified modes for all seven
deployments, five domains and three levels. The same frozen formatting normalizer applies throughout. Opus 5 Python
uses the complete separately evaluated revised-wording condition; its caption
identifies the change. Every point retains all eight responses to each of 128
prompts, including failed draws. This is a descriptive comparison across the
stated task formulations, rather than a common-prompt model ranking.

The supplement retains every original-protocol result, native refusal counts,
strict and normalized grades, conditional uniform references, intervals and
the separate before/after prompt study. Original evidence is not overwritten
by the main display. Refusal statistics and repair details appear only there.

`ops/build_frontier_paper_comparison.py` binds the original and prompt-study
evidence. `ops/plot_paper_hosted_breadth.py` reconstructs every displayed cell
and records its condition, denominator, grading and source hashes. The evidence
checker validates these selections and rendered bytes. There are eight main and seventeen supplementary figures in the ICLR version,
and seven main and eighteen supplementary figures in the workshop version. Superseded intro tables and the prior PDFs are
preserved in [the reframing audit](audits/hosted_reframe_20260911/README.md).

## Hosted temperature and selected-retry diagnostics

[The GPT temperature curve](figures/gpt56_temperature_curve.pdf) uses five
temperatures (0, 0.5, 1.0, 1.5, 2.0), 480 fixed prompts and eight draws per prompt,
with reasoning `none`. The left panel averages all fifteen level/domain cells;
the right panel shows three level curves, each averaging all five domains.
The horizontal axis is empirical `pass@8`: the fraction of prompts with at
least one correct response in their eight saved draws. The vertical axis is
mean `distinct@8`, the number of unique correct modes in those same draws.
All 19,200 responses and all failure-only prompt groups remain included.
The expansion retains the original 120 prompts and 4,800 responses unchanged,
adding 360 prompts and 14,400 responses through the same outcome-independent
row-hash ranking: 32 prompts per domain/level cell, up from eight.

| Temperature | Normalized pass@8 | Normalized distinct@8 |
|---|---:|---:|
| 0.0 | 65.62% | 0.673 |
| 0.5 | 72.29% | 0.840 |
| 1.0 | 75.21% | 0.963 |
| 1.5 | 79.79% | 1.125 |
| 2.0 | 79.38% | 1.146 |

Temperature 1.5 has the highest observed aggregate pass@8, while 2.0 has the
highest distinct@8. The paired T=2 minus T=1.5 difference is -0.42 percentage
points in pass@8 (95% interval [-2.50, 1.67]) and +0.021 distinct modes
([-0.029, 0.071]); neither interval resolves a difference. These exploratory
results do not establish a population optimum. The appendix reports all
pointwise intervals and separate original-120/additional-360 comparisons;
cohort differences may reflect prompt composition or collection time.
The figure and expanded tables show the controlled no-reasoning sweep.
Historical medium measurements cover only the original 120 prompts and remain
in source records, excluded from the expanded-cohort figure and tables.
The source artifacts are
`artifacts/frontier_temperature_20260911/GPT56_TEMPERATURE_CURVE_EXPANDED480.{json,md}` and
`artifacts/frontier_temperature_20260911/GPT56_PASS8_FRONTIER_EXPANDED480.{json,md,csv}`.
All earlier four-temperature and 120-prompt reports remain preserved.
Fresh [deployment checks](../artifacts/frontier_temperature_20260911/GPT56_REQUESTED_GRID_SUPPORT.md)
reject temperature 2.5 (maximum 2.0) and medium reasoning at every tested
non-default temperature. Those unsupported conditions have no benchmark points. The linked support report predates
the expansion; its appendix-table description refers to the retained 120-prompt
version. Current expanded tables exclude that historical medium reference.

Both appendices retain the complete Grok/Kimi paired-temperature results,
including Kimi's 696/960 token-limited outputs at 1.5, and the selected-validity
Opus 5 Python diagnostic. Its three additional requests all succeeded without
adding new modes; it does not replace the fixed 3,072-response main cohort.
Builders are `ops/plot_paper_gpt56_temperature_curve.py`,
`ops/build_frontier_temperature_paper.py`, and `ops/build_frontier_retry_paper.py`.
Each export authenticates its independent source audit and frozen condition
files. Numerical uncertainty and provider outcomes accompany the figures.

## Claim and theory alignment

Both papers share a claim-by-claim evidence map, the success–breadth lemma, and the complete conditional theory. The [correction and verification record](audits/claim_theory_alignment_20260911/README.md) explains the assumptions and the construction-reserve versus terminal-test distinction in the Level-2 comparison.

## Matched prompt-hint ablation

Both appendices include the original-versus-neutral control on identical Python
factors, MathIR, and Pantry problems at Levels 2 and 3. Each cell uses 32 fixed
problems and eight fresh draws per wording, with unchanged verifier and sampling
settings within each model. The local panel contains the initial
Qwen2.5-0.5B-Instruct checkpoint and 24 archived Level-2 Dr.GRPO/ReplayDr.GRPO
checkpoints: 27,648 responses. Python and MathIR have five matched training
seeds; Pantry has two. Level 3 is transfer evaluation.

The [ablation result record](results/modebench_prompt_ablation_20260911.json)
and its generated tables retain every included checkpoint and cell. The
figure shows paired changes in pass@8 and distinct@8; tables also separate
additional modes beyond the first correct answer. Strict verification is primary,
with the unchanged formatting normalizer as a sensitivity check. The initial
model is already instruction-tuned, with no fine-tuning-seed replication here.
Removing Python guidance also removes an executable example and shortens the
prompt, so that effect is a combined wording intervention.

This version reports the complete local panel only. The registered 9,216-response
frontier panel still awaits an API credential; the overall experiment is marked
`partial_panels`. Source evidence and the sealed runner are under
`artifacts/modebench_prompt_ablation_20260911/`. The repository build runs
`ops/check_paper_prompt_ablation.py` to reconstruct the analysis and authenticate
publication copies. Workshop builds independently validate the copied scope,
figure hashes, and supplement placement without access to the experiment archive.

The separately labeled [post-hoc Python failure audit](results/modebench_prompt_ablation_python_failure_diagnostic_20260911.json)
checks all 5,120 Dr.GRPO Python draws using the unchanged verifier. Original
outputs all share the example conditional body; 70.94% of neutral outputs are
invalid Python expressions. This describes failure types without identifying
an internal training mechanism.

## Sampling-budget ablation

Both appendices include the complete 110,592-response local follow-up: 64 fresh
draws per problem and wording on 16 selected problems per domain/level cell,
using all 25 registered checkpoints. The two new figures show pass@k and
distinct@k through 64, then breadth at matched numbers of correct responses.
The [complete local findings](../artifacts/modebench_discovery_curves_20260911/LOCAL_DISCOVERY_SUMMARY.md)
and [focused appendix PDF](../artifacts/modebench_discovery_curves_20260911/appendix_review/local_complete_v1/appendix_review.pdf)
retain all cells. MathIR largely remains concentrated; Python replay and Pantry
show additional discovery at larger budgets. The separately registered hosted
cohorts are collecting and do not yet contribute to this local-only report.

The analyzer, raw-response receipts, independent point-count review and exact
publication assets are authenticated before the paper builds. The Yue et al.
large-k motivation uses the existing `yue2025rlvrlimit` bibliography entry.
