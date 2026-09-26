# Paper figure manifest

This is the placement and evidence contract for the current ModeBench and
ReplayMaxRL paper. It follows the three-experiment narrative and deliberately
separates primary figures from supporting comparisons and historical artifacts.
Coverage below reflects the September 8, 2026 audited snapshots.

## Compiled figures

| Placement | Role | Asset | Evidence contract |
|---|---|---|---|
| Main, Fig. 1 | Collapse teaser | `figures/modecollapse_story.pdf` | Mechanically selected matched Qwen2.5-3B illustration; 32 fixed samples per checkpoint; caption discloses post-hoc status through the concrete trajectory rather than claiming an aggregate. |
| Main, Fig. 2 | ModeBench examples | `figures/modebench_examples.pdf` | One executable instance from each of the five domains, showing verifier outcome and canonical solution key. |
| Main, Fig. 3 | ReplayMaxRL mechanism | `figures/verified_support_story.pdf` | Conceptual diagram: identical policy samples feed MaxRL and key-balanced replay; not an empirical result. |
| Main, Fig. 4 | Experiment 1: retention and alternatives | `figures/experiment1_retention_comparator_matrix.pdf` | Panel A: 3 scales x 5 domains ReplayDr.GRPO-minus-Dr.GRPO grid with 74 admissible pairs. Fourteen blocks use five seeds; Falcon Countdown and its common-seed Average use four, marked by daggers. Panel B: all Qwen-0.5B references and alternatives use five matched seeds, including the before-training reference. Unadjusted 95% intervals apply only to complete blocks and are not encoded as cell-level significance decisions. Pass effects are probability points; distinct effects are expected verified-key counts. The green Average is a descriptive within-seed domain macro-average. |
| Main, Fig. 5 | Experiment 2: objective x memory | `figures/e118_all_scale_factorial_progress.pdf` | Qwen2.5-0.5B and Falcon3-1B show both completed objective/replay tracks; Qwen2.5-3B shows its completed Dr.GRPO/replay track. Averages are across domains within each track's common paired seeds. Falcon Dr.GRPO uses four common admissible seeds and a dashed track labeled `n=4`; all other displayed tracks use five. Faint paths expose seed averages and bold paths their means. Diamonds show initial references on the same seed sets. The breadth axis includes every plotted seed endpoint. Incomplete Qwen-3B MaxRL domains receive no cross-domain aggregate. |
| Main, Fig. 6 | Experiment 3: matched-level admission and interim comparison | `figures/modebench_level_admission.pdf` | Panel A retains frozen Qwen2.5-0.5B admission. Panel B averages seeds within each domain and then the five domains equally at checkpoints matched across both levels and all four methods. The September 8 snapshot has 21 domain-seed pairs: Graph/Python/MathIR use all five seeds at step 3,072; Countdown seeds 43--47 use 3,072/3,072/2,592/2,304/2,304; Pantry uses only seed 45 at step 0. This remains interim. The sidecar separately records 76/100 terminal Level-2 endpoints and three complete four-arm blocks: Graph, Python, and MathIR. |
| Appendix | Experiment 2 per-domain detail | `figures/e118_scale_extensions_appendix.pdf` | Five-domain Qwen2.5-0.5B and Falcon3-1B panels underlying Figure 5. MaxRL/replay pairs use five seeds everywhere; Falcon Countdown Dr.GRPO/replay uses four admissible seeds and receives no five-seed interval. Qwen2.5-3B Python has a separate complete five-seed appendix table. Its Graph and MathIR prefixes have `n=3` and `n=1` respectively; Countdown and Pantry have no paired MaxRL endpoints. These prefixes remain in the machine-readable record without a Qwen-3B MaxRL cross-domain average. |
| Appendix | Problem precheck: collapse without an intervention | `figures/baseline_collapse_precheck.pdf` | All 150 registered Dr.GRPO/GRPO runs across 3 scales x 5 domains x 2 objectives have four fixed K=8 draws at both step 0 and step 3,072; every model-domain-objective block contains five seeds. Bars split `distinct@8` into `pass@8` and extra verified modes. Retained-breadth badges appear only above a .05 initial extra-modes floor. Pass 0 is measured per arm; the two Qwen2.5-3B cells where the arms do not share it are recorded in the sidecar. |
| Appendix | Sustained retention | `figures/sustained_auc_effects_qwen05b.pdf` | Complete five-seed Qwen2.5-0.5B trajectory-AUC effects; no inferential domain pooling. AUC is a sensitivity analysis and can include long flat intervals after control collapse. |
| Appendix | Direct alternatives | `figures/direct_comparator_endpoint_effects.pdf` | GRPO, UCPO, and sparse RLEP-Dr minus matched Dr.GRPO; exact terminal `n`; intervals only at `n=5`. |
| Appendix | Direct-alternative trajectories | `figures/direct_baseline_learning_curves_static_strip.pdf` | Exact complete blocks and incomplete prefixes; unsupported cells blank. |
| Appendix | Semantic-MaxEnt comparison | `figures/verified_support_discovery_two_scale_effects.pdf` | The single compiled Semantic-MaxEnt comparison: 49 integrity-valid Qwen/Falcon pairs. It is supporting evidence, not a ReplayMaxRL component. |
| Appendix | Replay actuation | `figures/replay_mechanism_telemetry_qwen05b.pdf` | All 25 terminal Qwen replay runs and exact-zero controls; validates that the registered intervention was active. |

## Dated endpoint and table evidence

[`results/latest_results_20260908.json`](results/latest_results_20260908.json)
binds the current census to captured ledger copies and an endpoint audit:

| Campaign | Admitted endpoints | Complete blocks | Current evidence boundary |
|---|---:|---:|---|
| E118 MaxRL factorial | 124/150 | 11/15 | 59 matched pairs. Qwen-3B Graph seeds 70, 72, 73 and MathIR seed 74 are descriptive prefixes; Python is the only complete Qwen-3B MaxRL block. |
| E119 Level 2 | 76/100 | 3/5 | Graph, Python, and MathIR use all five four-method seeds. Countdown has two four-arm seeds; Pantry has no four-arm terminal intersection. |
| E120 replay weights | 33/45 | 6/9 | All six complete blocks pass the persisted frequency-weighting telemetry audit. The original Qwen-0.5B primary estimates retain their September 4 frozen input. |

The newly completed Python and MathIR effects are compiled from
[`results/latest_results_20260908_effects_table_body.tex`](results/latest_results_20260908_effects_table_body.tex).
Their `pass@8` and raw `distinct@8` replay means are positive under both fresh
objectives; all corresponding unadjusted paired 95% Student-t intervals include
zero. This statement does not extend to every secondary metric. Per-seed
values, arm means, exact intersections, and partial-block effects remain in the
JSON. The E120 uniform-minus-frequency summaries use exhaustive paired
percentile bootstrap intervals for complete five-seed blocks; their inference
convention is distinct from the E118/E119 Student-t summaries.

## Retired from the compiled manuscript

| Asset | Reason |
|---|---|
| `figures/terminal_pass8_distinct8_frontier.pdf` | The omnibus frontier mixed the primary retention contrast with supporting comparators. Experiment 1 now has a dedicated effect figure; alternatives move to the appendix. |
| `figures/replay_maxrl_qwen05b.pdf` | The Qwen-only snapshot is superseded by Figure 5's completed objective/replay tracks and its two-scale per-domain appendix. |
| Fixed/adaptive Semantic-MaxEnt figure families | The fixed no-replay arm appears once in Figure 4 as a direct benchmark; the detailed estimator and one verified-support comparison remain in the appendix rather than forming a parallel main storyline. |
| Historical adaptive, open-bank, dose, DAPO, and campaign-monitor figures | These are development or provenance artifacts and are not evidence in the current three-experiment narrative. |

Retired files remain on disk for provenance. They must not appear in
`paper/main.tex`.

## Regeneration

From the repository root, reproduce the current dated report and changed
figures from their retained snapshots:

```bash
python ops/exp_scaling/build_paper_latest_results.py --date 2026-09-08 \
  --from-audit paper/audits/results_refresh_20260908/latest_endpoints.json
python ops/exp_scaling/plot_paper_e118_all_scale_progress.py \
  --endpoint-audit paper/audits/results_refresh_20260908/latest_endpoints.json
python ops/exp_scaling/plot_paper_modebench_levels.py
python ops/exp_scaling/build_paper_e120_primary_breadth.py
python ops/check_paper_current_contract.py
make -C paper main.pdf
```

The current Figure 6 input is
[`results/modebench_level_comparison_snapshot.json`](results/modebench_level_comparison_snapshot.json).
Its collector, `ops/exp_scaling/freeze_paper_modebench_levels.py`, records every
registered cell's complete checkpoint availability before selecting the latest
shared checkpoint. `ops/exp_scaling/build_paper_latest_results.py` without
`--from-audit` collects a new dated endpoint census; it preserves the historical
E120 primary input. Regenerating the current analysis does not run
`build_paper_e120_frequency_progress.py`, which reads live campaign data into
the historical output path.

Builders for the retained supporting assets are:

```bash
python ops/plot_paper_collapse_toy.py
python ops/exp_scaling/build_paper_baseline_collapse_precheck.py
python ops/exp_scaling/plot_paper_baseline_collapse_precheck.py
python ops/exp_scaling/build_paper_e78_python_collapse_diagnostics.py
python ops/plot_paper_modebench_examples.py
python ops/plot_paper_support_story.py
python ops/exp_scaling/plot_paper_cross_scale_endpoint_effects.py
python ops/exp_scaling/plot_paper_sustained_auc_effects.py
python ops/exp_scaling/plot_paper_direct_comparator_endpoint_effects.py
python ops/exp_scaling/plot_paper_experiment1_composite.py
python ops/exp_scaling/plot_paper_aligned_domain_strips.py
python ops/exp_scaling/build_e112r1_two_scale_exploratory_results.py
python ops/exp_scaling/plot_e112r1_two_scale_exploratory_effects.py
python ops/exp_scaling/plot_paper_replay_mechanism_telemetry.py
```

## Fail-closed rules

- A complete efficacy block requires all five registered paired terminal seeds.
- An incomplete prefix prints exact `n`, receives no five-seed interval, and is
  never described as a completed result.
- Models and levels are not pooled for confirmatory claims. Figures 4 and 5
  use descriptive within-seed domain averages; Figure 6 averages observed seeds
  within each domain and then weights all five domains equally.
- Figure 6 admission is not a treatment effect. Its adjacent comparison remains
  interim because checkpoints differ across domain-seed pairs and PantryPlan
  contributes only an observed initial checkpoint. Missing evaluations are not
  imputed or replaced by admission measurements.
- Complete four-arm factorial claims use the common admissible seed intersection,
  rather than differences of aggregates with different seed sets. Eleven E118
  MaxRL/replay blocks are complete; the full factorial has ten complete blocks
  and one four-seed Falcon Countdown block.
- Conflicting terminal retries are never selected or averaged. The source
  exclusion for Falcon Countdown ReplayDr.GRPO seed 59 applies to every reused
  efficacy endpoint and aggregate; the primary replay comparison has 74 pairs.
- The dated census separates training-completion receipts, admitted endpoints,
  complete paired blocks, and E120 mechanism eligibility. Historical frozen
  result files remain fixed even when operational job ledgers change.
- Semantic-MaxEnt appears in the main text only as one row of Figure 4's direct-comparator panel; its definition and supporting analysis remain in the appendix.
- The categorical proof is conditional: MaxRL alone shares the binary-reward
  collapse flow; replay protects discovered recurrently replayed modes; uniform
  full-support convergence additionally assumes complete verified coverage.
