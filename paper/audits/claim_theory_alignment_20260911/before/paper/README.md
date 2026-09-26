# ModeBench and ReplayMaxRL papers

The [long ICLR manuscript](main.pdf), **There's More Than One Way: Mode Collapse in RLVR & ModeBench**, is built from [main.tex](main.tex).
Its scientific main is nine pages, including all eight main figures, with
references starting on page ten. A [main-text-only PDF](main-body.pdf) is also
available. The build checks every main
figure and the end of the conclusion against the rendered PDF, rather than
only the conclusion's starting page.
The [short MATH-AI manuscript](mathai2026/main.pdf) has a distinct four-page
main text plus the expanded supplement. Both versions share the same eighteen scientific figures: the long paper has
eight main and ten supplementary figures; the workshop has seven main and
eleven supplementary figures. The GPT temperature curve is Figure 8 on page 8
of the long paper and appears in the workshop supplement.

Start with the [figure inventory](FIGURE_MANIFEST.md), the
[coverage and provenance audit](FIGURE_DATA_AUDIT.md), or the
[September 11 numerical report](results/current_campaign_results_20260911.md).
The inventory identifies compiled assets; older figures on disk remain available
for provenance and are not part of the current manuscript.

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

## Current evidence: September 11, 2026

| Campaign | Admitted endpoints | Complete paired blocks | Coverage |
|---|---:|---:|---|
| E118 MaxRL / ReplayMaxRL, Level 1 | 138/150 | 12/15 | All five Qwen2.5-0.5B and Falcon3-1B domains; Qwen2.5-3B Graph and Python complete. |
| E119 four-method factorial, Level 2 | 88/100 | 4/5 | Qwen2.5-0.5B Graph, Countdown, Python and MathIR complete; PantryPlan incomplete. |
| E120-R1 fresh-frequency replay weights | 43/45 | 7/9 | All five Qwen2.5-0.5B domains and Falcon Graph/Pantry complete; Qwen2.5-3B Graph/Pantry each have four pairs. |

The Level-1 model set is Qwen2.5-0.5B, Falcon3-1B, and Qwen2.5-3B.
The corrected Dr.GRPO/ReplayDr.GRPO core has 74 admissible pairs: fourteen
five-seed blocks and Falcon Countdown at four seeds after the registered
conflicting-endpoint exclusion. E118 has 67 MaxRL/ReplayMaxRL pairs;
Qwen2.5-3B counts are Graph 5, Countdown 3, Python 5, MathIR 3, PantryPlan 1.
Figure 5 shows all three models in both main panels, including Qwen2.5-3B
MaxRL and ReplayMaxRL. Its dashed Qwen-3B MaxRL track is a descriptive
five-domain mean: first average each domain's own paired seeds, then weight
the five domains equally. The domain counts are 5/3/5/3/1; there is no seed
common to all five domains, so this track has no confidence interval or
cross-domain individual-seed paths. Its appendix shows every domain at all
three models, with exact paired counts.

Level 2 currently supports a four-domain terminal comparison at Qwen2.5-0.5B.
Figure 6 uses the same completed domains and five seeds for all four methods
at both levels. PantryPlan's observed endpoint counts are disclosed separately:
Dr.GRPO 3, ReplayDr.GRPO 3, MaxRL 1, ReplayMaxRL 1; the replay/control
seed intersection is two for Dr.GRPO and zero for MaxRL. These are neither a
complete Pantry factorial nor evidence for additional Level-2 models.
The supplementary domain detail stays alongside Figure 6 as a standalone asset.

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

Both supplements now follow benchmark/protocol, algorithm, current results,
mechanism checks, core theory, and reproducibility. The short supplement uses
a compact metric primer instead of repeating the long main narrative.
Current endpoint tables appear with the results; completion history, redundant
seed tables, extended proofs, and three secondary figures are preserved in the
[streamlining archive](audits/streamlining_20260911/README.md). The compact
fixed-semantic definition remains a main-figure comparator, and the fixed-bank
study retains its declining-tail evidence. No numerical snapshot changed.

## Appendix training curves

Both papers include full training trajectories for **accuracy (`pass@8`) and
verified modes (`distinct@8`)**. The primary figures use the same three-model
rows and four methods as Figure 5: Qwen2.5-0.5B, Falcon3-1B and Qwen2.5-3B,
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

The full `make -C paper figures` target also rebuilds both supporting trajectory
metrics. See the [figure inventory](FIGURE_MANIFEST.md) for all five curve plots.

## Reproduce the figures and results

From the repository root:

```sh
make -C paper latest-results      # rebuild dated tables/report from retained audit
make -C paper figures             # render the 15 training-study PDFs
make -C paper                    # check and build long paper
```

The Makefile pins `ANALYSIS_DATE=2026-09-11`,
`ENDPOINT_AUDIT=audits/results_refresh_20260911/latest_endpoints.json`, and
`LEVEL_SNAPSHOT=results/modebench_level_comparison_snapshot.json`.
Figure 5 uses the endpoint audit; Figure 6 uses its frozen level snapshot.
`figures-supporting` calls
[`render_paper_retained_figures.py`](../ops/exp_scaling/render_paper_retained_figures.py),
which uses retained JSON records and leaves their numerical contents unchanged.
The illustrative Figure 1 uses its existing fixed log set and selection rule;
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
  --audit-directory paper/audits/streamlining_20260911/workshop_sync \
  --reason 'Streamline both supplements while preserving the frozen primary evidence'
```

Add `--apply` to copy the reviewed assets, preserve replaced files and the old
snapshot, and record the new parent/workshop hashes. Choose a new audit directory
for each subsequent synchronization. `make -C paper sync-workshop` runs this
apply step with the Makefile defaults. Then:

```sh
make -C paper/mathai2026 bundle
```

The workshop build validates four main-text pages, all eighteen figures,
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

Both main texts place the hosted observation after the three controlled training
comparisons. Figure 7 replaces the dense introductory table with per-response
accuracy and verified modes for all seven deployments, five domains and three
levels. The same frozen formatting normalizer applies throughout. Opus 5 Python
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
checker validates these selections and rendered bytes. There are eight main and ten supplementary figures in the ICLR version,
and seven main and eleven supplementary figures in the workshop version. Superseded intro tables and the prior PDFs are
preserved in [the reframing audit](audits/hosted_reframe_20260911/README.md).

## Hosted temperature and selected-retry diagnostics

[The GPT temperature curve](figures/gpt56_temperature_curve.pdf) uses four
temperatures (0.5, 1.0, 1.5, 2.0), 120 fixed prompts and eight draws per prompt,
with reasoning `none`. The left panel averages all fifteen level/domain cells;
the right panel shows three level curves, each averaging all five domains.
The horizontal axis is empirical `pass@8`: the fraction of prompts with at
least one correct response in their eight saved draws. The vertical axis is
mean `distinct@8`, the number of unique correct modes in those same draws.
All 3,840 responses and all failure-only prompt groups remain included;
`pass@8` is measured directly, not inferred from per-response accuracy.

| Temperature | Normalized pass@8 | Normalized distinct@8 |
|---|---:|---:|
| 0.5 | 66.67% | 0.775 |
| 1.0 | 70.83% | 0.867 |
| 1.5 | 72.50% | 1.050 |
| 2.0 | 70.83% | 1.042 |

Temperature 1.5 is the best observed aggregate point on both axes, not an
established optimum. The decline in per-response accuracy between endpoint
temperatures does not imply a decline in `pass@8`. Open, unconnected markers
reuse the same prompt subset from the original medium cohort, which reaches
97.50% `pass@8` and 1.550 modes. This historical reference is not a randomized
reasoning treatment and does not establish temperature robustness with medium
reasoning. The source artifacts are
`artifacts/frontier_temperature_20260911/GPT56_PASS8_FRONTIER.{json,md}`;
the earlier per-response report is preserved separately.

Both appendices retain the complete Grok/Kimi paired-temperature results,
including Kimi's 696/960 token-limited outputs at 1.5, and the selected-validity
Opus 5 Python diagnostic. Its three additional requests all succeeded without
adding new modes; it does not replace the fixed 3,072-response main cohort.
Builders are `ops/plot_paper_gpt56_temperature_curve.py`,
`ops/build_frontier_temperature_paper.py`, and `ops/build_frontier_retry_paper.py`.
Each export authenticates its independent source audit and frozen condition
files. Numerical uncertainty and provider outcomes accompany the figures.
