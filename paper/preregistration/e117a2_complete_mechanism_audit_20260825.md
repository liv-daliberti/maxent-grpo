# E117-A2 complete mechanism-audit amendment

Frozen: 2026-08-25 while the E117-R1 analysis job is dependency-pending at
zero runtime and before E117 efficacy endpoints or contrasts exist. This
amendment replaces only the pending analysis job. It does not alter, restart,
stop, reprioritize, or otherwise change any of the 12 training cells.
PointMaze remains excluded.

## Why this amendment is needed

The registered E117 preflight promises more than the first executable audit
proved. In particular, the audit must fail closed on absent telemetry, verify
the physical node and complete C/P/F launch environment after scheduler-only
repairs, prove that C leaves all proposal-derived admission and retention state
untouched, and test counter continuity if Slurm restarts a job. It must also
distinguish correct identity from a short preflight that never exercises the
components needed for a Stage-1 screen.

No endpoint value, terminal evaluation, arm contrast, or efficacy statistic
was inspected to make these corrections.

## Frozen executable checks

The replacement audit retains every E117 and E117-A1 rule and adds the
following fail-closed checks.

1. The ledger must retain seed 117, C/P/F, four registered sentinels, 64 rows,
   one pass, 64 updates, checkpoints at 32, one common step-zero draw, one
   fixed proposal group and attempt, proposal temperature 1.20, replay weight
   0.10, F coefficient 0.10, no efficacy gate, no release-outcome inspection,
   and PointMaze exclusion.
2. Slurm accounting must show that every completed C/P/F member ran on the
   registered effective physical node. Within each sentinel, the complete
   exported launch environment must be exact across arms after removing only
   `SAVE_PATH`, `RUN_STAMP`, the C admission-discard flag, and the F semantic
   coefficient. The two arm-varying values must themselves equal the frozen
   C/P/F assignment.
3. Required numeric telemetry may not be absent, nonfinite, Boolean, or
   nonnumeric. Missing fields are failures, never implicit zeros.
4. Step-zero coverage must contain exactly draw index zero with evaluation kind
   `fixed_seed_sampled_k_neutral`, and the complete draw record must be exact
   across C/P/F within a sentinel.
5. Every optimizer step must preserve the common neutral request seed,
   fixed-control group/row/token accounting, and proposal request/row
   accounting across C/P/F. Step one must additionally preserve the
   pre-intervention neutral reward and proposal-validation surface.
6. Every arm must have zero control, proposal-conditioned, and transform rows
   sent to PPO; zero proposal objective-support delta; and zero gold-support,
   desired-mode-count, evaluation, or adaptive-retention feedback.
7. C must discard exactly every produced novel candidate at the admission
   boundary. It must have zero admitted outcomes, stored exemplars, cumulative
   proposal outcomes, tracked admissions, rollout conversion, or score/joint
   retention lifecycle state.
8. P/F may not discard at the C boundary. Every produced novel candidate must
   be admitted, stored, and reflected in the cumulative proposal and online
   retention counters. Per-step admissions must sum to the terminal cumulative
   counter.
9. Online retention telemetry is mandatory on every step. Replay-scoring
   retention telemetry exists only when replay rows are scored; whenever that
   conditional view is present, it is subject to the same completeness,
   no-feedback, no-adaptive-priority, and continuity checks. Its legitimate
   absence in a step with no replay rows is not imputed as data.
10. C/P must report coefficient and effective semantic RMS equal to zero. F
    must report coefficient 0.10 (allowing only float32 representation error)
    and positive effective RMS on every update with positive raw eligible RMS,
    per E117-A1. Eligibility remains descriptive when raw RMS is exactly zero.
11. Neutral request seeds must not repeat within a job. Proposal cumulative and
    every observed retention-admission counter must be monotone across the
    complete 1--64 trace. Duplicate or missing optimizer steps fail. If Slurm
    reports a restart, passing these checks records resume continuity as
    verified; otherwise resume fidelity is explicitly `unexercised`.

## Identity versus execution readiness

The audit output uses two separate gates.

- `passed` means every registered identity, isolation, leakage, state, and
  continuity check passed. It never incorporates an endpoint contrast.
- `stage1_execution_readiness.ready` additionally requires at least one
  sentinel with a positive P admission count and at least one sentinel with a
  positive F effective-pressure update. A correctly plumbed but entirely
  unexercised preflight may pass identity while remaining unready for Stage 1.

The readiness fields report which sentinel blocks exercised proposal replay
and semantic pressure. They do not rank blocks or authorize an efficacy claim.
Endpoint means, confidence intervals, tests, rankings, and pooling remain
prohibited for E117.

## Installation boundary

Install the corrected audit in a new content-addressed source snapshot. Submit
one no-requeue analysis job with `afterany` dependencies on exactly the same 12
effective E117-R1 training job IDs, inspect its held scheduler record, and only
then cancel the superseded zero-runtime dependency-pending audit. Record both
job IDs, records, digests, and the facts that training configuration was
unchanged and outcomes were not inspected.
