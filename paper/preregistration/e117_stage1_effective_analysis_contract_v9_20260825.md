# E117 Stage-1 effective analysis contract v9

Consolidated: 2026-08-25T12:23:30-04:00 while all twelve E117-R1 jobs
remained pending, with zero realized optimizer updates and before any Stage-1
seed, job, ledger, sampled response, or outcome existed.

Status: analysis-only development contract. This document consolidates the
original Stage-1 contract and amendments A1--A8. A8 closes the previously
unmet untouched-evaluation-split precondition; no causal or advancement rule
changes. This document does not authorize a launch. Execution remains
conditional on a passing E117 mechanism audit and activation-readiness gate.
PointMaze is excluded.

## Minimal causal design

Use the same C/P/F system on exactly four sentinels:

- `qwen05b/countdown`;
- `qwen05b/graph_coloring`;
- `qwen05b/python_factors`;
- `falcon1b/mathir`.

`C` issues the proposal-shaped request, validates and canonicalizes it, then
discards the payload before mutation; semantic coefficient is zero. `P` is C
with validator-positive novel proposal admission and uniform replay enabled;
semantic coefficient remains zero. `F` is P with v7 semantic PPO enabled at
the fixed nominal coefficient 0.10.

The estimands are total effects:

- `P-C`: enabling proposal admission and replay;
- `F-P`: enabling semantic v7 on the proposal/replay system.

Neither contrast holds realized post-treatment proposal consumption, support,
retention, or replay fixed. `F-P` is not a controlled direct semantic effect.
If it advances, a semantic-without-proposal arm is required for the next
decomposition.

## Frozen grid and execution

Use exactly three fresh paired training seeds, at least 16 registered distinct
common-random-number K=8 evaluation draws, all fixed checkpoints from step
zero through the eight-pass terminal horizon, and all three arms on all four
sentinels. The future execution protocol must freeze the seeds, draw schedule,
checkpoint grid, source snapshot, node blocks, resource envelopes, output
paths, and complete export/audit machinery before submission.

Within each sentinel, assign the three seed ranks to the cyclic actual start
orders exactly once: `C-P-F`, `P-F-C`, and `F-C-P`. Enforce those orders with
dependencies or a block wrapper and audit realized start order. Within a
sentinel/seed block, C/P/F must use the same physical node class, resource
envelope, frozen source, data, and runtime plumbing.

## Untouched evaluation separation

Retain the existing source training banks. Do not use their historically
reused evaluation prompts for Stage-1 advancement. The only permitted Stage-1
evaluation paths are the `multi_answer` splits under:

- `var/data/e117_evaluation_reserve_v1/development/countdown/eval`;
- `var/data/e117_evaluation_reserve_v1/development/graph_coloring/eval`;
- `var/data/e117_evaluation_reserve_v1/development/python_factors/eval`;
- `var/data/e117_evaluation_reserve_v1/development/mathir/eval`.

Each path has exactly 128 deterministic prompts disjoint from every historical
train/evaluation identity. The reserve identity is
`var/data/e117_evaluation_reserve_v1/identity.json`, SHA-256
`a5b9eb4289cca8f78d90249fabbd85343c1c3eb3b5c3df6a2235bbd124446289`.
The analogous `confirmation` paths are disjoint from both historical and
development identities and are sealed from development analysis. They may be
opened only under a later confirmation protocol after a candidate advances.

## Fail-closed table identity

The analyzer requires the exact registered seed/arm/checkpoint/draw grid,
exactly the four sentinels above, exactly K=8, strict identifier types, finite
endpoints, no duplicate or extra row, and no missing row. Step-zero primitive
endpoints must agree exactly across C/P/F within each sentinel/seed/draw block.

Every row carries two response-free lowercase SHA-256 digests:

- prompt projection: `answer_keys`, `answer_mode_count`, `option_ids`,
  `prompt`, `prompt_index`, `reference`;
- request projection: `option_ids`, `prompt_index`,
  `request_seeds_by_option`.

The prompt digest must be exact within a sentinel across every row. The request
digest must be exact within a sentinel/draw across seeds, arms, and
checkpoints, and all registered request digests must be distinct across draws.
Metrics, responses, rewards, and endpoints are excluded from both identity
projections.

## Endpoints and uncertainty

The primitive endpoint vector is `(pass@8, raw distinct correct modes@8)`.
Correctness-adjusted breadth is the derived decomposition
`raw distinct@8 - pass@8`; it is reported with both uncertainty axes but has no
independent advancement veto.

Compute `P-C` and `F-P` within every seed and common draw before averaging.
Report terminal effects and normalized trapezoidal AUC over the complete fixed
horizon. Retain every paired seed/draw effect and report separately:

- training-seed SE over draw-averaged paired seed effects;
- evaluation Monte Carlo SE over seed-averaged common-draw effects;
- per-seed evaluation Monte Carlo SE.

Do not pool the two axes into one sample size, confidence interval, or p-value.

## Development advancement gate

For a component to be actionable within a sentinel, both terminal and
normalized-AUC raw-distinct effects must each:

1. be strictly greater than +0.05;
2. exceed two corresponding evaluation Monte Carlo SEs;
3. be positive in at least two of three paired training seeds.

At both terminal and normalized AUC, pass@8 must be at least -0.03 on the
three-seed mean and at least -0.10 in every paired seed for both the component
numerator versus its denominator (`P-C` or `F-P`) and the candidate numerator
versus C (`P-C` or `F-C`).

A broad development candidate requires the same component to be actionable in
at least three of four sentinels and to include both registered model families.
Exactly Countdown alone is a domain-specific candidate, with Graph retained as
its registered negative boundary. Every other pattern is `do_not_advance`.

## Interpretation and next decision

Stage 1 is a development screen, not confirmation. A surviving candidate must
use the sealed confirmation split with at least five fresh paired training
seeds and no tuning on Stage-1 outcomes.

- If `P-C` advances and `F-P` does not, retire semantic PPO.
- If `F-P` advances, add semantic-without-proposal before direct attribution.
- If mechanisms activate but neither contrast advances, work on conservative
  retention and replay rather than increasing eta or proposal attempts.
- If the E117 mechanism audit or activation-readiness gate fails, repair
  plumbing before launching Stage 1.

The executable analysis is
`ops/exp_scaling/e117_stage1_statistics.py`, schema
`e117_stage1_paired_vector_statistics_v8`. The reserve materializer is
`ops/exp_scaling/materialize_e117_evaluation_reserves.py`. A future launcher
and table builder must bind the v9 manifest, analysis implementation and tests,
reserve identity and tests, and latest chained A8 evidence by SHA-256.
