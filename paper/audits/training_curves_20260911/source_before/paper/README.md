# ModeBench and ReplayMaxRL papers

The [long ICLR manuscript](main.pdf) is built from [main.tex](main.tex).
The [short MATH-AI manuscript](mathai2026/main.pdf) has a distinct four-page
main text plus the expanded supplement. Both versions share six main figures
and eight supplementary figures, with the same admitted observations.

Start with the [figure inventory](FIGURE_MANIFEST.md), the
[coverage and provenance audit](FIGURE_DATA_AUDIT.md), or the
[September 11 numerical report](results/current_campaign_results_20260911.md).
The inventory identifies compiled assets; older figures on disk remain available
for provenance and are not part of the current manuscript.

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

## Reproduce the figures and results

From the repository root:

```sh
make -C paper latest-results      # rebuild dated tables/report from retained audit
make -C paper figures             # render all 14 compiled PDFs
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
  --audit-directory paper/audits/figure_refresh_20260911/workshop_sync \
  --reason 'Refresh figures and September 11 result coverage in both papers'
```

Add `--apply` to copy the reviewed assets, preserve replaced files and the old
snapshot, and record the new parent/workshop hashes. Choose a new audit directory
for each subsequent synchronization. `make -C paper sync-workshop` runs this
apply step with the Makefile defaults. Then:

```sh
make -C paper/mathai2026 bundle
```

The workshop build validates four main-text pages, all fourteen figures,
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
