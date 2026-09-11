# Five-domain confirmation — live frozen-checkpoint results

Generated `2026-07-28T00:52:59.399880+00:00`. Campaign audit: `fail`; terminal runs `9/54`; integrity violations `1`.

Only three-seed-complete frozen checkpoints are summarized. AUC is emitted only after every registered checkpoint lands; partial AUC is descriptive and cannot pass a gate.

## Confirmation readiness

| Requirement | Live evidence | Terminal requirement |
|---|---:|---:|
| Independent registered runs | 9/54 terminal | 54/54 |
| Fixed arm/domain checkpoint cells | 89/174 | complete surface |
| Required seed-metric values | 1050/2046 | complete surface |
| Evaluation cadence | pass (0 violations) | pass, ≤1 epoch gap |
| E66/E68 pre-intervention paired seeds | 3/12 (`in_progress`) | 12/12, pass |
| E68 durable support-separation checkpoints | 3/12 (`in_progress`) | 12/12, pass |
| Primary interpretation gates | e58_cross_domain: `pending`, e68_repair: `pending`, plumbing_consistency: `pending`, math500_realism: `pending` | all evaluated only at the frozen terminal surface |

**Objective-correction disclosure:** E65R1 is excluded from all confirmatory gates because its runtime forced E58's novelty beta from `0.50` to `0.0` before the singleton actuator could fire. Those trajectories remain archived as engineering evidence. E67 is also excluded: its shared proposal/on-policy bank was rejected before optimization, with `0` optimizer metric files. E68 is the prospectively frozen separated-support treatment and must pass an explicit runtime same-objective audit against E66. Machine-readable invalidation audits: E65R1 `confirmed`, E67 `confirmed`.

**Infrastructure disclosure:** after scheduler preemption, the nine pending non-Math E66 controls were moved from single-node `lowprio` placement to broader same-accelerator-family `pvl-lowprio` pools. Run IDs, checkpoints, frozen source, objective, seeds, and all scientific settings are unchanged; the E66 audit fails if any trace regresses below its recorded pre-amendment step. The same hash-bound, same-family resume rule was first applied to three pending E61-R1 jobs. After eight additional RTX 3090 jobs were preempted, a second hash-bound placement-only amendment moved those pending checkpoint resumes to the same accelerator-family `pvl-lowprio` pool; the E61-R1 audit verifies both amendment identities and rejects trace regression. After both registered A6000 nodes entered a health-check drain for overheated GPUs, all six paired E66/E68 Graph jobs were moved together to non-MLTheory `lowprio` nodes that advertise the same A6000 accelerator family. This second hash-bound amendment changes placement only and both prospective audits reject trace regression. When two zero-step E68 jobs were then assigned a node206 GPU already holding 48.3 GiB from foreign processes, a third paired amendment removed that pool and moved all six Graph jobs together to healthy `mltheory/pvl-lowprio` A6000 nodes. Only the exact pre-amendment log prefixes are classified as infrastructure interruptions; later OOMs remain hard failures.

Pre-intervention E66/E68 trajectory equivalence audit: `in_progress`; ready paired seeds `3/12`; paired updates compared `960`; violations `0`.

E68 checkpoint separation audit: `in_progress`; latest durable checkpoints audited `3/12`; proposal-only outcomes checked `48`; graduated to neutral on-policy discoveries `255`; objective-count overlaps `0`; violations `0`.

Evaluation-cadence audit: `pass`; materialized runs with optimizer progress audited `44`; maximum permitted gap `1` prompt epoch; violations `0`.

Fixed paper-checkpoint coverage audit: `in_progress`; complete arm/domain checkpoint cells `89/174`; landed seed-metric values `1050/2046`; complete full comparison surfaces by domain: Countdown `0/10`; Graph coloring `0/10`; Held-out MATH-500 transfer `3/7`; MathIR action menu `8/10`; Python factors `0/10`.

The complete seed-level numerical surface is refreshed at [fixed-checkpoint CSV](e65_five_domain_confirmation_fixed_checkpoints_live.csv).

## Latest shared E58 versus Dr.GRPO checkpoint

These are interim three-seed means at the latest fixed checkpoint landed by both arms in each domain; they are not terminal claims.

| Domain | pass | Dr.GRPO pass@8 | E58 pass@8 | Δ | Dr.GRPO distinct@8 | E58 distinct@8 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Graph coloring | 10 | 0.372 | 0.968 | 0.595 | 0.393 | 2.344 | 1.951 |
| Countdown | 5 | 0.608 | 0.665 | 0.057 | 0.650 | 1.674 | 1.024 |
| Python factors | 6 | 0.172 | 0.740 | 0.568 | 0.172 | 0.956 | 0.784 |
| MathIR action menu | 8 | 0.364 | 0.779 | 0.415 | 0.364 | 0.805 | 0.441 |

## Latest shared separated-support E68 versus E66 checkpoint

This is the direct causal comparison at the latest fixed three-seed checkpoint shared by both prospective arms. It is interim only; the frozen repair gate uses pass 12.

| Domain | pass | E66 pass@8 | E68 pass@8 | Δ | E66 distinct@8 | E68 distinct@8 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MathIR action menu | 8 | 0.817 | 0.839 | 0.022 | 0.847 | 0.926 | 0.079 |

## Secondary paired prompt-level uncertainty

This post-specified analysis is descriptive and cannot change any primary gate. It resamples the three paired training seeds and the shared evaluation prompts as crossed units after averaging the four fixed K=8 draws. Complete domain-checkpoints: `8`/40; integrity violations: `0`. The [source-aligned method](../preregistration/e68_paired_prompt_uncertainty_secondary_v2_20260727.md) was frozen before integration into this report; it computes no p-values.

| Domain | pass | metric | E68 − E66 | descriptive crossed-bootstrap 95% | seed 43 / 44 / 45 deltas |
|---|---:|---|---:|---:|---:|
| MathIR action menu | 8 | greedy | 0.049 | [-0.016, 0.117] | 0.031 / 0.102 / 0.016 |
| MathIR action menu | 8 | mean8 | 0.023 | [-0.020, 0.064] | 0.034 / 0.034 / -0.001 |
| MathIR action menu | 8 | pass8 | 0.022 | [-0.027, 0.074] | 0.062 / -0.002 / 0.006 |
| MathIR action menu | 8 | distinct8 | 0.079 | [0.012, 0.154] | 0.135 / 0.055 / 0.049 |

Greedy repeatability sensitivity at the latest displayed checkpoints: the separate repeated temperature-zero call changed `7/768` prompt scores; per-run mean shifts ranged from `+0.000` to `+0.008`. The table is centered on the primary greedy result used by the paper figure.

These intervals include finite-prompt uncertainty but do not turn three training seeds into more than three independent runs. Intermediate intervals remain interim.

## Mechanism diagnosis and repair

These are live mechanism diagnostics, not selected performance endpoints. They make the repair hypothesis falsifiable: the controller must first detect an entropy deficit, and E68 must then demonstrate a clean, support-only intervention before its terminal gate can pass.

| Python E58 seed | step | entropy EMA / own reference | ratio | unprojected coefficient | projection active | mean bank support | outcomes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 43 | 2418 | 0.174 / 0.404 | 0.431 | 0.220 | 0 | 1.221 | 469 |
| 44 | 4608 | — / — | — | — | — | — | — |
| 45 | 4608 | 0.049 / 0.403 | 0.122 | 0.822 | 0 | 1.000 | 98 |

The prior E62R10 **one-seed engineering pilot only** showed actuator feasibility: maximum mean support 6.089, 793 admitted outcomes, 206.510 admissions per 100 updates, 0 proposal rows sent to PPO, and terminal pass@8 1.000 after one pass. Its breadth is precisely why it is not confirmatory evidence.

Separated-support E68 currently has `305` audited entropy-gated interventions over `9794` metric-bearing updates (3.114 per 100 updates); maximum admitted in any proposal group is `1` and the mechanism audit is `in_progress`. Zero is expected before a registered collapse condition occurs; the frozen terminal gate cannot pass without at least one clean intervention.

| E68 domain | metric-bearing updates | interventions | interventions / 100 updates |
|---|---:|---:|---:|
| Countdown | 0 | 0 | — |
| Graph coloring | 0 | 0 | — |
| MathIR action menu | 9794 | 305 | 3.114 |
| Python factors | 0 | 0 | — |

## ModeBench pass@8

| Domain | Arm | checkpoints | latest pass | latest mean [range] | terminal | AUC/12 |
|---|---|---:|---:|---:|---:|---:|
| Graph coloring | Dr.GRPO | 10/10 | 12 | 0.369 [0.344, 0.396] | 0.369 | 0.410 |
| Graph coloring | E58 | 9/10 | 10 | 0.968 [0.964, 0.971] | — | — |
| Graph coloring | E66 same-plumbing control | 2/10 | 1 | 0.865 [0.849, 0.883] | — | — |
| Graph coloring | E68 separated-support actuator | 0/10 | — | — | — | — |
| Countdown | Dr.GRPO | 6/10 | 5 | 0.608 [0.572, 0.648] | — | — |
| Countdown | E58 | 7/10 | 6 | 0.667 [0.664, 0.672] | — | — |
| Countdown | E66 same-plumbing control | 2/10 | 1 | 0.588 [0.486, 0.646] | — | — |
| Countdown | E68 separated-support actuator | 0/10 | — | — | — | — |
| Python factors | Dr.GRPO | 7/10 | 6 | 0.172 [0.172, 0.172] | — | — |
| Python factors | E58 | 7/10 | 6 | 0.740 [0.219, 1.000] | — | — |
| Python factors | E66 same-plumbing control | 0/10 | — | — | — | — |
| Python factors | E68 separated-support actuator | 0/10 | — | — | — | — |
| MathIR action menu | Dr.GRPO | 8/10 | 8 | 0.364 [0.293, 0.408] | — | — |
| MathIR action menu | E58 | 8/10 | 8 | 0.779 [0.742, 0.820] | — | — |
| MathIR action menu | E66 same-plumbing control | 9/10 | 10 | 0.818 [0.789, 0.850] | — | — |
| MathIR action menu | E68 separated-support actuator | 8/10 | 8 | 0.839 [0.768, 0.889] | — | — |

## Held-out MATH-500

The named paper figure, table below, and primary realism gate remain frozen to the seven even-pass anchors `0, 2, 4, 6, 8, 10, 12`. The separate [all-epoch diagnostic figure](../figures/e61r1_e58_vs_grpo_05b_12ep_all_epoch_diagnostic_live.pdf) displays every complete integer epoch from the already-retained quarter-epoch evaluation traces.

| Arm | metric | checkpoints | latest pass | latest mean [range] | terminal | AUC/12 |
|---|---|---:|---:|---:|---:|---:|
| Dr.GRPO | greedy | 3/7 | 4 | 0.343 [0.326, 0.362] | — | — |
| Dr.GRPO | mean8 | 3/7 | 4 | 0.290 [0.286, 0.292] | — | — |
| Dr.GRPO | pass8 | 3/7 | 4 | 0.583 [0.582, 0.584] | — | — |
| E58 | greedy | 3/7 | 4 | 0.348 [0.342, 0.352] | — | — |
| E58 | mean8 | 3/7 | 4 | 0.276 [0.269, 0.282] | — | — |
| E58 | pass8 | 3/7 | 4 | 0.585 [0.564, 0.600] | — | — |

Supplemental raw-trace verifier sensitivity: `in_progress`; 113 caught timeout diagnostics, 0 identical-response reward conflicts, and step-0 six-run response/reward redundancy `pass`. This audit does not alter primary scores and must pass at all seven fixed checkpoints for a terminal campaign audit.

## Frozen interpretation gates

| Claim | status |
|---|---|
| E58 cross-domain positive | `pending` |
| Separated-support E68 singleton repair | `pending` |
| E66 versus historical E58 plumbing sensitivity | `pending` |
| Held-out MATH-500 realism | `pending` |

A pending gate is not evidence of success or failure. Full seed values, paired deltas, raw AUCs, and gate diagnostics are in the machine-readable JSON artifact.
