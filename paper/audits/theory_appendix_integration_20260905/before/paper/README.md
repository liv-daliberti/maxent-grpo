# Mode Collapse under GRPO

[`main.tex`](main.tex) is the ICLR 2027-format source for **Mode Collapse under
GRPO: ModeBench and x-Mode Dr.GRPO**. [`main.pdf`](main.pdf) is the built
manuscript.

The vendored `iclr2027_conference.sty`, `iclr2027_conference.bst`,
`fancyhdr.sty`, and `natbib.sty` are byte-exact copies from the official July
28, 2026 ICLR 2027 archive at
<https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>. The
archive SHA-256 is
`0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`;
the manuscript contract checks all four extracted file hashes on every build.

## Paper in one paragraph

A binary verifier distinguishes correct from incorrect responses but does not
distinguish redundant correct responses from genuinely different correct
outcomes. Under Dr.GRPO, frequent correct outcomes receive more on-policy
updates while a correct outcome that disappears from the sampled group receives
none. ModeBench makes this loss of verified outcome support measurable with
validators that both execute a response and assign a canonical outcome key.
x-Mode Dr.GRPO augments GRPO with verified replay: it stores one policy-generated,
validator-positive exemplar for each observed key and revisits the bank with a
deterministic recurrent schedule and uniform teacher-forced likelihood. The
reported experiments instantiate its task update with Dr.GRPO.

## What is measured

For prompt `x`, validator-positive response `y` receives an executable outcome
key `kappa_x(y)`. The target distribution is the policy distribution over keys
conditioned on verifier success, not token strings.

- `pass@1`: greedy success probability.
- `pass@8`: probability that eight samples contain at least one correct answer.
- `distinct@8`: expected number of distinct correct executable keys in eight
  samples.

The primary collapse diagnosis compares `distinct@8` with `pass@8`: near
equality means successful eight-sample sets almost never contain a second
correct mode. A falling `distinct@8` alongside stable correctness shows support
contraction. Token entropy alone is not evidence of outcome diversity because
formatting aliases can map to the same executed key.

## Theory guarantee

The appendix analyzes prompt-isolated, continuous mean flow with independently
trained categorical logits, finite on-policy groups, and zero reference KL.
Under these assumptions a unique initially most probable correct mode becomes
dominant. This is a conditional mechanism, not a general theorem about finite
PPO or Adam training in neural networks.

For a fixed verified bank and constant positive replay dose, a potential bound
keeps each banked categorical probability uniformly positive, including at
infinite time. The bound can be numerically tiny. With complete coverage,
independent logits converge to the replay target; it is uniform for equal
categorical weights. The implemented length-normalized exemplar loss requires
a separate weighting argument and does not guarantee uniform learned key
probabilities. Discovery, capacity truncation, cross-prompt interference, and
finite stochastic optimization remain outside these guarantees.

## ModeBench

The five benchmark domains are:

- Graph coloring: the complete executed coloring assignment.
- Countdown: the normalized, executed arithmetic tree.
- Executable Python factors: the function's return vector on hidden cases.
- MathIR: the exact equation-state trajectory produced by an executed action
  menu.
- PantryPlan: the ingredient support of a constraint-feasible plan.

Every admitted key is both correct and executable. There is no gold catalogue
of modes and no clustering of free-form text.

## Evidence map

| Evidence block | Status | Claim licensed |
|---|---|---|
| Qwen2.5-0.5B x-Mode Dr.GRPO versus matched Dr.GRPO | Terminal | Five-domain paired pass-8 effects, intervals, raw seeds, and trajectory AUC; no pooled effect |
| Plain GRPO versus matched Dr.GRPO | 75/75 terminal | All 15 model--domain blocks are complete at paired `n=5`; the five Qwen2.5-3B blocks are inconclusive and no domain pooling or model-size trend is inferred |
| Replay-bank and compute telemetry | Terminal | All 25 replay and 25 exact-zero control logs; occupancy, capacity hits, realized dose, and descriptive paired update time; identity-level survival unavailable |
| Bank occupancy versus retained breadth | Terminal observational diagnostic | All 25 replay seeds joined to same-seed normalized-AUC and terminal correctness-adjusted breadth effects; domain means only, with no pooled, causal, or identity-level survival claim |
| Qwen2.5-0.5B and Falcon3-1B fixed semantic-MaxEnt factorials | Terminal | Five-domain $2\times2$ paired effects, interactions, and intervals at both scales; domain-specific ablations, not pooled or general MaxEnt claims |
| Qwen2.5-3B fixed semantic MaxEnt + ReplayDr.GRPO | Terminal descriptive slice | Completed seed 70 on all five static domains; exact pass@8, mean@8, and distinct@8 endpoints and matched ReplayDr.GRPO deltas are reported with `n=1`, without intervals, significance, pooling, or a model-scale claim |
| Adaptive semantic MaxEnt + ReplayDr.GRPO | Terminal outcomes; Qwen mechanism gate failed | All five Qwen and Falcon domains are terminal `n=5`; Qwen2.5-3B is exact `n=1`. The Qwen E89 mechanism failure forbids a successful-adaptation claim. |
| Adaptive ReplayDr.GRPO dose ablation | Terminal; 25/25 | All five paired `n=5` domain effects; no cross-domain pooling |
| Historical decoding robustness control | Terminal historical control | Frozen pass-12 matched Dr.GRPO and superseded multi-component checkpoints across temperature, sample budget, and nucleus truncation; support claim for those policies only, not x-Mode or current-component evidence |
| UCPO versus shared Dr.GRPO cells | 50/75 terminal | All ten smaller-model paired `n=5` blocks are complete; Qwen Countdown has positive pass and adjusted-breadth intervals, while the newly completed Qwen MathIR block is neutral |
| Sparse prompt-matched RLEP-Dr repair | 47/75 terminal | Nine complete `n=5` blocks (all five Qwen domains; Falcon Graph/Countdown/MathIR/Pantry) plus Falcon Python `n=2`; Qwen MathIR has a positive paired pass@8 interval, Qwen Countdown remains inconclusive, and remaining extensions are nonterminal or pool-gated |
| DAPO versus completed Qwen/Falcon controls | Custom R3 excluded; official-verl R4-R2 gate passed; 4/50 science cells terminal; 46 nonterminal; 0 failed; 0 standardized endpoints | R3 is an immutable adapter diagnostic. R4 hit the Ray socket-path limit; R4-R1 fixed Ray but exposed a vLLM scheduler-cap incompatibility. R4-R2 preserves the scientific recipe and released all 50 cells. Qwen Graph seeds 43--46 completed 24 accepted updates and have exact upstream `acc@1` diagnostics only; without standardized pass@8/breadth, DAPO licenses no efficacy claim |
| Falcon3-1B and Qwen2.5-3B cross-family results | Terminal | Five terminal five-seed domain effects at each added scale (25 pairs each), with no pooling or model-size trend |
| Corrected semantic-estimator program | E105 retired; E111 15/15 mechanism cells terminal; E109 15/15 terminal; E112-R1 53/75 operationally | E105 is audit-only; E111's gate passes; E109 is the complete repaired-Python comparator; E112 reports 49 integrity-valid frozen smaller-model pairs after repeated 14/33/50-cell looks, with Qwen2.5-3B and one conflicting comparator pair excluded |
| E102 open-bank discovery bundle | Terminal follow-up outside canonical matrix | All 25 cells pass mechanism audit; domain-specific paired effects and admissions/priority telemetry, with no component attribution or realized proposal-token efficiency claim |
| Earlier broad comparison | Historical multi-component treatment | Motivation and provenance only; not an estimate of x-Mode Dr.GRPO |
| Earlier rehearsal comparison | Historical component evidence | Motivation for replay; not evidence for any coefficient rule |

The completed clean comparison uses Qwen2.5-0.5B-Instruct, seeds 43--47,
exactly eight passes, and checkpoints every half pass. Both arms perform the same bank
maintenance, recurrent traversal, exemplar scoring, and backward pass. The
control applies an exact-zero replay derivative; the x-Mode Dr.GRPO arm applies the
fixed replay weight `.10`. Pass 8 is the sole primary endpoint, and intermediate
checkpoints are trajectory measurements rather than model-selection candidates.
The aligned Falcon3-1B and Qwen2.5-3B comparisons preserve the same replay
intervention under model-specific optimizer recipes fixed before replay-arm
outcomes. Falcon and Qwen2.5-3B are complete across all five static domains,
each with five paired seeds per domain. The fixed Semantic MaxEnt +
ReplayDr.GRPO seed-70 treatment is terminal on all five domains and is reported
descriptively. The current endpoint artifact is rebuilt directly from immutable
campaign ledgers: missing cells are not imputed, arms retain every independently
available pass-8 endpoint, and paired effects use only their seed intersection.

## Reproduce the paper

From the repository root:

```bash
python ops/plot_paper_modebench_examples.py
python ops/plot_paper_verified_replay_mechanism.py
python ops/plot_paper_modecollapse.py
python ops/plot_paper_collapse_toy.py
python ops/exp_scaling/build_paper_baseline_collapse_precheck.py
python ops/exp_scaling/plot_paper_baseline_collapse_precheck.py
python ops/exp_scaling/build_paper_e78_terminal_results.py
python ops/exp_scaling/build_paper_maxent_factorial_results.py
python ops/exp_scaling/build_paper_e87_qwen3b_fixed_semantic_results.py
python ops/exp_scaling/build_paper_core_terminal_endpoints.py
python ops/exp_scaling/build_paper_program_status.py
python ops/exp_scaling/build_paper_e113r4_dapo_progress.py
python ops/exp_scaling/plot_paper_inference_frontier.py
python ops/exp_scaling/plot_paper_inference_pass_at_k.py
python ops/exp_scaling/plot_paper_fixed_semantic_factorial_effects.py
python ops/exp_scaling/plot_paper_qwen3b_endpoint_progress.py
python ops/exp_scaling/build_e112r1_two_scale_exploratory_results.py
python ops/exp_scaling/plot_e112r1_two_scale_exploratory_effects.py
python ops/exp_scaling/plot_paper_sustained_auc_effects.py
python ops/exp_scaling/plot_e72_decoding_frontier.py
python ops/exp_scaling/plot_paper_comparison_families.py \
  --comparison all --scale all --evidence terminal \
  --output-dir paper/figures/comparisons
python ops/exp_scaling/plot_paper_replay_mechanism_telemetry.py
python ops/exp_scaling/plot_paper_bank_occupancy_outcomes.py
python ops/exp_scaling/plot_paper_adaptive_dose_gate.py
python ops/exp_scaling/plot_paper_adaptive_mechanism_outcomes.py
python ops/exp_scaling/plot_paper_replay_dose_progress.py
python ops/exp_scaling/audit_e102_full_open_bank_maxent_replay.py
python ops/exp_scaling/plot_paper_open_bank_bundle_effects.py
python ops/exp_scaling/plot_paper_starvation_fallback_isolation.py
python ops/exp_scaling/plot_paper_admission_retention_funnel.py
python ops/exp_scaling/plot_e106_group_centered_mechanism_diagnostic.py
python ops/exp_scaling/plot_paper_aligned_domain_strips.py
python ops/exp_scaling/plot_paper_cross_scale_endpoint_effects.py
python ops/exp_scaling/plot_paper_direct_comparator_endpoint_effects.py
make -C paper
```

The plot scripts regenerate:

- [`figures/verified_replay_mechanism.pdf`](figures/verified_replay_mechanism.pdf):
  the appendix verify, store, recurrent revisit, and uniform replay mechanism;
- [`figures/modebench_examples.pdf`](figures/modebench_examples.pdf): one
  execution-checked response/verify/key row per benchmark domain;
- [`figures/baseline_collapse_precheck.pdf`](figures/baseline_collapse_precheck.pdf):
  the main-text problem precheck. All 150 registered pass-0 and pass-8 cells of
  unmodified Dr.GRPO and unmodified GRPO across five domains, five seeds, and
  three scales, with each bar splitting `distinct@8` into the first correct mode
  and the verified modes beyond it. Retained-breadth percentages appear only
  where pass-0 extra modes exceed .05 per prompt, and pass 0 is measured per arm
  rather than assumed shared;
- [`figures/modecollapse_story.pdf`](figures/modecollapse_story.pdf): the
  mechanically selected equal-correctness Graph Coloring contrast from four
  terminal Qwen2.5-3B matched pairs;
- [`figures/terminal_pass8_distinct8_frontier.pdf`](figures/terminal_pass8_distinct8_frontier.pdf):
  the main-text end-of-training frontier with pass@8 on the x axis and
  distinct@8 on the y axis. It focuses on matched Dr.GRPO, ReplayDr.GRPO,
  standalone fixed/adaptive Semantic MaxEnt, and fixed/adaptive Semantic
  MaxEnt + ReplayDr.GRPO. Qwen2.5-3B endpoints retain exact `n`; standalone
  adaptive MaxEnt is blank because no without-replay cell is registered.
  GRPO, UCPO, and RLEP-Dr are kept readable in the matched-baseline forest;
- [`figures/direct_comparator_endpoint_effects.pdf`](figures/direct_comparator_endpoint_effects.pdf):
  the appendix paired endpoint forest for GRPO, UCPO, and RLEP-Dr against
  matched Dr.GRPO. It prints exact terminal `n` in the invariant 3x5 grid and
  reserves means and paired 95% Student-t intervals for complete five-seed
  blocks. DAPO is intentionally absent because custom R3 is excluded and the
  four training-terminal R4-R2 cells have only incompatible upstream `acc@1`
  diagnostics, not standardized pass@8 or breadth endpoints;
- [`figures/e112r1_two_scale_endpoint_effects.pdf`](figures/e112r1_two_scale_endpoint_effects.pdf):
  the author-requested exploratory E112 terminal-effect forest for Qwen2.5-0.5B
  and Falcon3-1B. It reports 49 integrity-valid frozen pairs, prints Falcon
  Countdown at `n=4`, excludes Qwen2.5-3B and the conflicting comparator
  value, and makes no confirmatory, component-isolated, or trajectory-AUC claim;
- [`figures/e72_decoding_frontier.pdf`](figures/e72_decoding_frontier.pdf):
  the restored historical five-domain accuracy--breadth frontier over six
  temperatures. It shows matched Dr.GRPO against the superseded
  multi-component treatment with paired `n=5`; rings mark temperature one,
  and the caption and provenance forbid attribution to current x-Mode;
- [`figures/xmode_adaptive_cross_scale_distinct_at_k.pdf`](figures/xmode_adaptive_cross_scale_distinct_at_k.pdf):
  distinct-mode@K for full x-Mode Dr.GRPO against matched Dr.GRPO in the invariant
  3x5 model grid, derived from the four registered $K=8$ draws; each populated
  cell states its constant seed count and terminal/progress status;
- [`figures/xmode_adaptive_cross_scale_pass_at_k.pdf`](figures/xmode_adaptive_cross_scale_pass_at_k.pdf):
  the matched pass@K companion on the identical 3x5 grid, frozen seed sets,
  checkpoints, and K budgets, showing whether added sampling preserves or
  improves correctness alongside breadth;
- [`figures/fixed_semantic_factorial_effects.pdf`](figures/fixed_semantic_factorial_effects.pdf):
  raw paired terminal distinct@8 effects, means, and paired 95% Student-t
  intervals for fixed semantic MaxEnt without replay, on replay, and their
  interaction across all five Qwen and all five Falcon domains;
- [`figures/qwen3b_exact_endpoint_progress.pdf`](figures/qwen3b_exact_endpoint_progress.pdf):
  every available Qwen2.5-3B terminal pass@8--distinct@8 endpoint across the
  five static domains, with exact method-specific `n` and no aggregate,
  interval, imputation, or cross-domain pooling;
- [`figures/sustained_auc_effects_qwen05b.pdf`](figures/sustained_auc_effects_qwen05b.pdf):
  paired five-seed ReplayDr.GRPO effects in correctness, distinct-mode, and
  correctness-adjusted normalized trajectory AUC over every registered
  checkpoint from 0 through 8 passes, separately for all five Qwen2.5-0.5B
  domains;
- [`figures/modecollapse_training_compact.pdf`](figures/modecollapse_training_compact.pdf):
  the compact historical training surface retained for provenance;
- [`figures/comparisons/`](figures/comparisons/): separated comparison
  families in the Figure 4 visual language. Every static trajectory asset is
  a physical 3x5 grid: Qwen2.5-0.5B, Falcon3-1B, and Qwen2.5-3B rows by the
  invariant Graph/Countdown/Python/MathIR/Pantry columns. Cells that miss the
  figure's evidence gate are blank; rows never disappear and no result is
  pooled. The completed core comparison is rendered in GRPO-family distinct@8,
  pass@8, and mean@8 training grids;
- [`figures/cross_scale_terminal_endpoint_effects.pdf`](figures/cross_scale_terminal_endpoint_effects.pdf):
  fifteen balanced Dr.GRPO--ReplayDr.GRPO model--domain cells and
  all five Falcon GRPO contrasts, with paired points and intervals only at
  `n=5`;
- [`figures/replay_mechanism_telemetry_qwen05b.pdf`](figures/replay_mechanism_telemetry_qwen05b.pdf):
  all-update terminal bank occupancy, capacity, realized replay dose, and
  descriptive compute-matched timing telemetry;
- [`figures/bank_occupancy_retained_breadth.pdf`](figures/bank_occupancy_retained_breadth.pdf):
  the time-matched observational join between bank occupancy and
  correctness-adjusted breadth for all 25 clean replay seeds, plus per-seed
  capacity-hit frequency against the terminal effect. It shows raw seeds and
  descriptive domain means, with no pooled fit or survival claim;
- [`figures/adaptive_semantic_gate_e88.pdf`](figures/adaptive_semantic_gate_e88.pdf):
  the closed 5% adaptive semantic-dose mechanism gate;
- [`figures/adaptive_mechanism_outcomes_qwen05b.pdf`](figures/adaptive_mechanism_outcomes_qwen05b.pdf):
  all nine failed 5%-target records as mechanism-only context, followed by all
  25 terminal reachable-target cells. It records the failed E89 mechanism gate
  and shows same-seed adaptive-minus-fixed outcomes without claiming successful
  adaptation;
- `replay_dose_qwen05b_progress_static_strip` under
  [`figures/comparisons/`](figures/comparisons/) shows all 25 terminal
  Adaptive ReplayDr.GRPO cells with paired seeds 43--47 in every domain;
- [`figures/open_bank_bundle_endpoint_effects.pdf`](figures/open_bank_bundle_endpoint_effects.pdf):
  terminal E102 effects versus ReplayDr.GRPO and Dr.GRPO plus verified-mode
  admissions and priority replay groups for all five domains;
- [`figures/starvation_fallback_isolation.pdf`](figures/starvation_fallback_isolation.pdf):
  all 25 paired E103-minus-E102 endpoints beside fallback activations,
  admissions, and extra proposal groups;
- [`figures/admission_retention_funnel.pdf`](figures/admission_retention_funnel.pdf):
  the passed ten-cell E108 outcome-blind funnel and bounded-priority actuation;
- [`figures/e106_group_centered_mechanism_diagnostic.pdf`](figures/e106_group_centered_mechanism_diagnostic.pdf):
  the completed 15-cell v6 identity, centering, eligibility,
  semantic-pressure, and replay-actuation diagnostic; v6's later outcome-blind
  supersession and E111 v7 progress are recorded separately;
- the standalone training and decoding figures described in the manuscript
  appendix.

The pass@8--distinct@8 frontier is the empirical result figure in the main
text. Its balanced blocks and exact incomplete prefixes have different marker
semantics. The frozen 15-panel Figure 4 wall remains provenance only and is not
compiled. The three completed training-metric grids, terminal cross-scale
effects, factorial, and direct-baseline trajectories are separated in the
appendix.
[`FIGURE_MANIFEST.md`](FIGURE_MANIFEST.md) is the placement and evidence
contract.

## Source-of-truth artifacts

- Clean x-Mode Dr.GRPO protocol:
  [`preregistration/e78_verified_replay_only_05b_20260804.md`](preregistration/e78_verified_replay_only_05b_20260804.md)
- Submitted clean-comparison jobs:
  [`../var/artifacts/e78_verified_replay_only_05b_jobs.json`](../var/artifacts/e78_verified_replay_only_05b_jobs.json)
- Historical interim Figure 4 provenance (not compiled):
  [`figures/figure4_interim_20260806.json`](figures/figure4_interim_20260806.json)
- Terminal comparison provenance: adjacent JSON records under
  [`figures/comparisons/`](figures/comparisons/), generated by
  [`../ops/exp_scaling/plot_paper_comparison_families.py`](../ops/exp_scaling/plot_paper_comparison_families.py)
  and reflowed with source hashes by
  [`../ops/exp_scaling/plot_paper_aligned_domain_strips.py`](../ops/exp_scaling/plot_paper_aligned_domain_strips.py)
- Cross-scale endpoint provenance:
  [`figures/cross_scale_terminal_endpoint_effects.json`](figures/cross_scale_terminal_endpoint_effects.json)
- Replay mechanism telemetry provenance:
  [`figures/replay_mechanism_telemetry_qwen05b.json`](figures/replay_mechanism_telemetry_qwen05b.json)
- Bank occupancy/outcome provenance:
  [`figures/bank_occupancy_retained_breadth.json`](figures/bank_occupancy_retained_breadth.json)
- Adaptive semantic-dose gate provenance:
  [`figures/adaptive_semantic_gate_e88.json`](figures/adaptive_semantic_gate_e88.json)
- Adaptive controller-to-outcome provenance:
  [`figures/adaptive_mechanism_outcomes_qwen05b.json`](figures/adaptive_mechanism_outcomes_qwen05b.json)
- Direct-comparator endpoint provenance:
  [`figures/direct_comparator_endpoint_effects.json`](figures/direct_comparator_endpoint_effects.json)
- DAPO launch provenance:
  [`preregistration/e113_dapo_direct_baseline_20260818.md`](preregistration/e113_dapo_direct_baseline_20260818.md),
  [`../ops/exp_scaling/launch_e113_dapo_direct_baseline.py`](../ops/exp_scaling/launch_e113_dapo_direct_baseline.py), and
  [`../var/artifacts/source_snapshots/e76_tuned_scale_f941cd52a8a9927c/SNAPSHOT_IDENTITY.json`](../var/artifacts/source_snapshots/e76_tuned_scale_f941cd52a8a9927c/SNAPSHOT_IDENTITY.json).
  The released ledger is
  [`../var/artifacts/e113_dapo_direct_baseline_jobs.json`](../var/artifacts/e113_dapo_direct_baseline_jobs.json);
  the Qwen and Falcon smoke logs are
  [`../var/artifacts/logs/e113-q05-dapo-smoke-30736381.out`](../var/artifacts/logs/e113-q05-dapo-smoke-30736381.out)
  and
  [`../var/artifacts/logs/e113-f1-dapo-smoke-30736382.out`](../var/artifacts/logs/e113-f1-dapo-smoke-30736382.out).
  Both smokes exhausted ten all-zero dynamic-sampling batches, leaving no
  scientific endpoint. The exact 50 original scientific placeholders never
  started, were canceled after the official successor launched, and are frozen
  in [`../var/artifacts/e113_original_dependency_placeholders_retirement.json`](../var/artifacts/e113_original_dependency_placeholders_retirement.json).
- DAPO recovery and implementation-boundary provenance:
  [`preregistration/e113r1_dapo_recovery_smokes_20260819.md`](preregistration/e113r1_dapo_recovery_smokes_20260819.md),
  [`../ops/exp_scaling/audit_e113r1m1_dapo_effective_gate.py`](../ops/exp_scaling/audit_e113r1m1_dapo_effective_gate.py), and
  [`preregistration/e113r3_dapo_full_relaunch_20260819.md`](preregistration/e113r3_dapo_full_relaunch_20260819.md)
  retain the custom-adapter history. Static audit then established that R3's
  one-prompt retry sampler differs materially from published DAPO's
  multi-prompt filter-and-buffer sampler, so R3 is excluded from named-DAPO
  efficacy. The authoritative successor is frozen in
  [`preregistration/e113r4_official_verl_dapo_20260819.md`](preregistration/e113r4_official_verl_dapo_20260819.md).
  The exact R3 cancellation record is
  [`../var/artifacts/e113r3_retirement_for_official_dapo.json`](../var/artifacts/e113r3_retirement_for_official_dapo.json),
  and the amended official-R4 launch ledger is
  [`../var/artifacts/e113r4_official_verl_dapo_jobs.json`](../var/artifacts/e113r4_official_verl_dapo_jobs.json).
  Smokes 30800804--30800805 failed before training on the Ray socket-path limit,
  and their zero-runtime dependents were canceled. The infrastructure-only
  recovery is frozen in
  [`preregistration/e113r4r1_ray_socket_recovery_20260823.md`](preregistration/e113r4r1_ray_socket_recovery_20260823.md).
  Replacement smokes 30855240--30855241 reached vLLM construction but failed
  before rollout or training because the 448-token aggregate scheduler cap was
  below the pinned 1,024-sequence limit; their 50 dependents were canceled at
  zero runtime. The compatibility-only successor is prospectively frozen in
  [`preregistration/e113r4r2_vllm_scheduler_recovery_20260824.md`](preregistration/e113r4r2_vllm_scheduler_recovery_20260824.md)
  and released its two smoke jobs, 30865563--30865564, at 08:40 EDT on
  August 24. Both passed, releasing the frozen 50-cell science cohort. At the
  August 25 audit, Qwen Graph seeds 43--46 (jobs 30869111--30869114) had
  completed all 24 accepted updates; 46 cells were nonterminal and none had
  failed. Their exact per-cell upstream diagnostics and input hashes are in
  [`results/e113r4_dapo_progress.json`](results/e113r4_dapo_progress.json).
  These one-response-per-prompt `acc@1` values are not substituted for the
  paper's standardized pass@8 or executed-mode breadth endpoint.
  The R4-R2 runtime snapshot is
  [`../var/artifacts/source_snapshots/e113r4_official_verl_dapo_vllmsched_d2b6f70532c0c34d/SNAPSHOT_IDENTITY.json`](../var/artifacts/source_snapshots/e113r4_official_verl_dapo_vllmsched_d2b6f70532c0c34d/SNAPSHOT_IDENTITY.json).
  The preceding R4-R1 runtime snapshot is
  [`../var/artifacts/source_snapshots/e113r4_official_verl_dapo_raytmp_9419ba707c93b23d/SNAPSHOT_IDENTITY.json`](../var/artifacts/source_snapshots/e113r4_official_verl_dapo_raytmp_9419ba707c93b23d/SNAPSHOT_IDENTITY.json).
- Corrected semantic-estimator provenance:
  [`preregistration/e105_v6_superseded_by_e111_20260818.md`](preregistration/e105_v6_superseded_by_e111_20260818.md),
  [`../var/artifacts/e105_superseded_v6_retirement_jobs.json`](../var/artifacts/e105_superseded_v6_retirement_jobs.json),
  [`preregistration/e111_verified_support_discovery_mechanism_gate_three_scale_20260818.md`](preregistration/e111_verified_support_discovery_mechanism_gate_three_scale_20260818.md), and
  [`../var/artifacts/e111_verified_support_discovery_mechanism_gate_audit.json`](../var/artifacts/e111_verified_support_discovery_mechanism_gate_audit.json).
  The compiled terminal mechanism record is
  [`figures/e111_verified_support_mechanism_diagnostic.json`](figures/e111_verified_support_mechanism_diagnostic.json).
- E112-R1 exploratory disclosure and audit provenance:
  [`preregistration/e112r1_user_requested_private_interim_unblinding_20260820.md`](preregistration/e112r1_user_requested_private_interim_unblinding_20260820.md),
  [`preregistration/e112r1_user_requested_private_interim_unblinding_20260823.md`](preregistration/e112r1_user_requested_private_interim_unblinding_20260823.md),
  [`preregistration/e112r1_user_requested_private_interim_unblinding_20260826.md`](preregistration/e112r1_user_requested_private_interim_unblinding_20260826.md),
  [`preregistration/e112r1_two_scale_exploratory_public_disclosure_20260828.md`](preregistration/e112r1_two_scale_exploratory_public_disclosure_20260828.md),
  [`preregistration/e112r1_final_analysis_a7_response_free_identity_erratum_20260828.md`](preregistration/e112r1_final_analysis_a7_response_free_identity_erratum_20260828.md), and
  [`preregistration/e112r1_two_scale_49pair_terminal_integrity_amendment_20260828.md`](preregistration/e112r1_two_scale_49pair_terminal_integrity_amendment_20260828.md).
  The public result is [`results/e112r1_two_scale_exploratory_results.json`](results/e112r1_two_scale_exploratory_results.json);
  its figure JSON records all 49 included effects and the exact excluded-log hash.
  Private 14/33/50-cell artifacts remain forbidden for campaign selection.
- Open-bank bundle and mechanism-audit provenance:
  [`figures/open_bank_bundle_endpoint_effects.json`](figures/open_bank_bundle_endpoint_effects.json) and
  [`../var/artifacts/e102_full_open_bank_maxent_replay_05b_audit_latest.json`](../var/artifacts/e102_full_open_bank_maxent_replay_05b_audit_latest.json)
- Historical decoding-control provenance:
  [`figures/e72_decoding_frontier.json`](figures/e72_decoding_frontier.json),
  generated from the frozen 660-cell E72 summary by
  [`../ops/exp_scaling/plot_e72_decoding_frontier.py`](../ops/exp_scaling/plot_e72_decoding_frontier.py)
- Frozen interim four-metric table provenance:
  [`results/figure4_interim_20260806_table.json`](results/figure4_interim_20260806_table.json),
  generated by [`../ops/exp_scaling/build_figure4_interim_table.py`](../ops/exp_scaling/build_figure4_interim_table.py)
- Live figure generator: [`../ops/exp_scaling/plot_figure4_with_falcon_preview.py`](../ops/exp_scaling/plot_figure4_with_falcon_preview.py)
- Opening model trajectory:
  [`../var/artifacts/paper_graph_collapse_toy.json`](../var/artifacts/paper_graph_collapse_toy.json)
- Terminal paired-effects provenance:
  [`results/e78_terminal_05b.json`](results/e78_terminal_05b.json) and
  [`results/e78_terminal_05b_table_body.tex`](results/e78_terminal_05b_table_body.tex),
  generated by
  [`../ops/exp_scaling/build_paper_e78_terminal_results.py`](../ops/exp_scaling/build_paper_e78_terminal_results.py)
- Fixed semantic-MaxEnt factorial provenance:
  [`results/maxent_factorial_05b.json`](results/maxent_factorial_05b.json),
  with table bodies generated by
  [`../ops/exp_scaling/build_paper_maxent_factorial_results.py`](../ops/exp_scaling/build_paper_maxent_factorial_results.py)
- Canonical ten-method coverage registry:
  [`results/paper_program_status.json`](results/paper_program_status.json) and
  [`results/paper_program_status_table_body.tex`](results/paper_program_status_table_body.tex),
  generated from the 750-cell matrix by
  [`../ops/exp_scaling/build_paper_program_status.py`](../ops/exp_scaling/build_paper_program_status.py)
- Full x-Mode breadth-frontier provenance:
  [`figures/xmode_adaptive_cross_scale_distinct_at_k.json`](figures/xmode_adaptive_cross_scale_distinct_at_k.json),
  generated by
  [`../ops/exp_scaling/plot_paper_inference_frontier.py`](../ops/exp_scaling/plot_paper_inference_frontier.py).
  The older `figures/inference_frontier_05b.*` render is superseded and is no
  longer regenerated.
- Historical replay protocol retained for provenance:
  [`preregistration/e58_global_verified_replay_canonical_05b.md`](preregistration/e58_global_verified_replay_canonical_05b.md)

## Implementation map

- Environments and executable keys:
  [`../src/oat_drgrpo/canonical_actions.py`](../src/oat_drgrpo/canonical_actions.py),
  [`../src/oat_drgrpo/python_modebench.py`](../src/oat_drgrpo/python_modebench.py),
  [`../src/oat_drgrpo/mathir.py`](../src/oat_drgrpo/mathir.py), and
  [`../src/oat_drgrpo/pantry_plan.py`](../src/oat_drgrpo/pantry_plan.py)
- External Python execution boundary:
  [`../src/oat_drgrpo/python_modebench_process.py`](../src/oat_drgrpo/python_modebench_process.py)
- Verified replay loss and online bank:
  [`../src/oat_drgrpo/canonical_replay.py`](../src/oat_drgrpo/canonical_replay.py)
  and [`../src/oat_drgrpo/online_canonical_bank.py`](../src/oat_drgrpo/online_canonical_bank.py)
- Retired custom DAPO-adapter history:
  [`../src/oat_drgrpo/dapo.py`](../src/oat_drgrpo/dapo.py),
  [`../src/oat_drgrpo/learner/grpo.py`](../src/oat_drgrpo/learner/grpo.py),
  and [`../src/oat_drgrpo/learner/run.py`](../src/oat_drgrpo/learner/run.py).
  These files describe excluded E113-R3, not the named-DAPO comparator.
- Official-verl DAPO recipe and launcher:
  [`../var/cache/verl_dapo_4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/main_dapo.py`](../var/cache/verl_dapo_4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/main_dapo.py),
  [`../var/cache/verl_dapo_4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/dapo_ray_trainer.py`](../var/cache/verl_dapo_4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/dapo_ray_trainer.py), and
  [`../ops/exp_scaling/launch_e113r4_official_dapo.py`](../ops/exp_scaling/launch_e113r4_official_dapo.py).
  Local R4 runtime code is limited to exact data/reward adaptation, the
  scheduler wrapper, and terminal receipts.
- Clean experiment launcher:
  [`../ops/exp_scaling/launch_e78_verified_replay_only_05b.py`](../ops/exp_scaling/launch_e78_verified_replay_only_05b.py)

## Claim boundary

Verified replay is a retention mechanism, not a discovery oracle: it cannot
protect a mode before the policy produces it and the validator accepts it. The
primary cross-family result is the five-domain, five-seed Falcon3-1B block;
Qwen2.5-0.5B and Qwen2.5-3B provide two additional complete five-domain
uncertainty estimates, reported separately.
Fixed semantic MaxEnt is a separate discovery-amplifier ablation: its completed
factorial is strongly positive on PantryPlan, modest on Countdown, and otherwise
near zero after adjusting for correctness; no interaction interval excludes
zero.
DAPO remains a running comparator rather than efficacy evidence: custom R3 is
retired and excluded, R4/R4-R1 exposed and isolated two infrastructure
incompatibilities, and the R4-R2 gate passed. Four of 50 science cells are
training-terminal, but they provide only exact upstream `acc@1` diagnostics;
no standardized pass@8 or breadth endpoint exists, so DAPO licenses no
numerical efficacy claim yet.
Historical results from broader objectives remain auditable but are not
attributed to x-Mode Dr.GRPO; incomplete scale extensions license no final claim.

The September 5 review also adds a deterministic finite-step categorical replay lemma with an explicit step-size bound. It proves convergence to the fixed bank weights and makes the incomplete-bank limitation explicit: unbanked correct modes have zero limiting mass in this idealized model. See the [independent proof review](audits/finite_step_categorical_replay_independent_review_20260905.md); this does not extend the guarantee to stochastic neural optimizers.
