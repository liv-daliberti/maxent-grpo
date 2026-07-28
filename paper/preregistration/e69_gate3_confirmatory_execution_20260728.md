# E69 Gate 3 confirmatory execution contract

Date frozen: 2026-07-28, after Gate 2 launch and before any terminal Gate 2
outcome was available.

This document instantiates Gate 3 of
`e69_verified_route_successor_protocol_20260728.md`. It does not alter Gate 2,
inspect MATH-500, or authorize a Gate 3 launch unless the frozen Gate 2 audit
returns `pass` with zero integrity violations.

## Cohort and reuse

The confirmatory comparison contains the frozen compute-matched control and
successor at seeds 43, 44, and 45 for six prompt passes:

- Graph: `grpo_compute_matched` versus `verified_route_successor`;
- Countdown: `grpo_compute_matched` versus `verified_route_successor`;
- Python factors: `grpo_compute_matched` versus
  `verified_route_successor`;
- MathIR: `grpo_compute_matched` versus `verified_route_successor`; and
- MATH12K-384/MATH12K-route-dev-128: `grpo_compute_matched` versus
  `verified_first_global_replay_canonical`.

Gate 1 selected free-form-MATH route abstention, so the endpoint-only MATH arm
is the E69 successor for this area. It uses no trace prompt, route novelty,
proposal actuator, or cross-prompt route replay.

The exact Gate 2 seed-43 control and successor jobs are reused. They are not
rerun or averaged with duplicate attempts. Gate 3 therefore launches only
seeds 44 and 45: 16 new executable-domain jobs and four new MATH jobs. The
complete confirmatory identity contains 30 unique physical runs: ten reused
seed-43 runs and 20 newly submitted runs.

## Frozen execution

The model revision, data roots, prompts, verifiers, response limits, optimizer,
learning rate, rollout group size, six-pass stopping rule, pass-by-pass
evaluation cadence, fixed evaluation seeds, checkpoint retention, compute
controls, placement classes, and watchdog policy are identical to Gate 2.

Every arm issues one neutral group and three discarded proposal-shaped control
groups of 16 rows per training prompt. Every arm enables one capacity-16
teacher-forced replay group per optimizer update. The compute-matched Dr.GRPO
arm traverses the identical two-score-pass replay path with an exactly zero
replay derivative. The same telemetry and forbidden-feedback checks from Gate
2 remain hard integrity requirements.

Pass 6 is the sole terminal checkpoint. No best-checkpoint selection,
early-stopping choice, replacement seed, outcome-dependent restart, or
hyperparameter change is permitted. A scheduler or hardware restart may resume
only the same logical job from its latest checkpoint and remains part of the
same manifest identity.

## Frozen analysis

For every domain, arm, seed, and pass 0 through 6, report:

- greedy verified accuracy, mean@8, pass@8, and verified distinct@8 where
  defined;
- the paired successor-minus-control seed delta;
- verifier acceptance and failure counts;
- route-bank growth, proposal attempts and admissions, cross-prompt replay,
  and post-replay neutral reproduction;
- neutral, discarded-control, replay-realized, and replay-charged token
  accounting; and
- optimizer updates and integrity diagnostics.

Aggregate curves are arithmetic means over the three paired seeds. Terminal
uncertainty uses 10,000 deterministic crossed-bootstrap replicates with seed
`690301`: resample the three paired seeds with replacement and, within each
selected seed, resample matched development prompts with replacement using the
same prompt indices in both arms. Percentile 2.5% and 97.5% quantiles are
descriptive 95% intervals. All three raw paired seed deltas are shown because
three seeds do not support a precise population-level variance estimate.

The internal confirmatory result is classified as follows:

- **task-quality noninferior** in a development area only if terminal
  three-seed mean greedy and pass@8 deltas are each at least `-0.02`; MATH
  route-dev additionally requires mean@8 at least `-0.01`;
- **positive support** in an executable domain only if terminal mean
  distinct@8 is positive, at least two of three paired seed deltas are
  positive, and the pass-5 mean delta is also positive;
- **mechanism present** only if at least three executable domains have a
  positive post-replay neutral-reproduction count in at least two seeds; and
- **internal success** only if all five development areas are task-quality
  noninferior, at least three executable domains have positive support, MathIR
  has positive terminal pass@8 and distinct@8 mean deltas, the mechanism is
  present, and every integrity check passes.

These classifications do not control whether completed frozen checkpoints are
reported. If Gate 3 is complete with valid integrity, the algorithm,
checkpoint rule, and held-out analysis code are frozen and Gate 4 evaluates
MATH-500 once, regardless of the direction of the Gate 3 result.

## MATH-500 seal

Gate 3 training and analysis use only MATH12K route-dev for free-form-MATH
outcomes. No MATH-500 prediction, score, prompt text, aggregate, or checkpoint
comparison is read. The only previously permitted MATH-500 operation remains
the normalized-problem overlap firewall recorded by the materialization
manifest.
