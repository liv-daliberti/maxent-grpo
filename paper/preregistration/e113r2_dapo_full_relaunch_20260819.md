# E113-R2: gated full DAPO direct-comparator relaunch

**Frozen:** 2026-08-19 15:48 EDT, after the original E113 gate failure and
while both E113-R1 Graph recovery smokes were scheduler-pending at zero
accepted updates. No E113-R1 outcome and no E113-R2 training result had been
observed.

## Purpose and immutable history

E113-R2 is the fresh scientific successor to the failed E113 launch. The
original E113 ledger, two failed Countdown smokes, and 50
`DependencyNeverSatisfied` science placeholders remain immutable and are not
reclassified as training outcomes. E113-R1 is a two-job, non-scientific
operational gate; it cannot release or modify those original jobs.

The requested comparative estimand remains DAPO minus the already completed
plain Dr.GRPO control at the same model family, domain, seed, data, decoding,
placement, and training horizon. E113-R2 restores the complete registered
surface rather than selecting domains from recovery-smoke outcomes.

## Fail-closed release gate

No E113-R2 job may be submitted unless
`audit_e113r1_dapo_recovery_smokes.py` passes both model-family smokes. Each
must have a terminal completion receipt at exactly 32 accepted optimizer
updates, 32 accepted DAPO metric records, finite loss/gradient/token telemetry,
the registered `.20/.28` clips, dynamic sampling enabled, at most ten
generation batches per update, and at most 5,120 sampled rows.

If either smoke fails, E113-R2 remains unlaunched. There is no force flag and
no partial-family release.

## Frozen scientific matrix

- Models: Qwen2.5-0.5B-Instruct and Falcon3-1B-Instruct.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, PantryPlan.
- Seeds: the five matched E78/E79 seeds for each model family.
- Cells: `2 * 5 * 5 = 50`, all submitted in one audited release.
- Horizon: 384 training prompts for eight passes, or 3,072 accepted optimizer
  updates per cell.
- Sampling: 16 responses per prompt, temperature 1, top-p 1.
- DAPO: GRPO critic, dynamic rejection of constant-reward groups, asymmetric
  clips `.20/.28`, token-level loss aggregation, overlong buffer ratio `.20`,
  overlong penalty factor `1.0`, and at most ten generation batches per
  accepted update.
- Query ceiling: `3,072 * 16 * 10 = 491,520` sampled responses per cell.
- All replay, MaxEnt, UCPO, RLEP, collision, DIAYN, discovery-tracking, and
  entropy interventions remain disabled exactly as in E113.
- Runtime: the immutable E113 source/ops snapshot recorded in the original
  ledger.

Each cell writes to a new `e113r2` run directory and receives a new job ID.
No original E113 path, job, receipt, or ledger field may be reused or edited.

## Operational failure semantics

The ten-generation-batch limit is the scientific DAPO rule, not a scheduler
retry unit. E113-R2 therefore disables wrapper-triggered requeue after a
nonzero learner exit. Successful trajectories are unchanged; a cell that
cannot find a non-constant group within ten batches fails once and remains a
registered feasibility failure. Scheduler/node requeue and checkpoint resume
remain available under cluster policy, but the experiment wrapper may not
turn DAPO exhaustion into six additional unregistered attempts.

No failed cell may be restarted with a higher generation-batch cap, altered
temperature, different prompt order, warm start, or replacement seed under
this cohort. Such a method would require a separately named prospective
protocol.

## Reporting rule

The paper and monitor must always report the full denominator of 50. Efficacy
estimates are paired only for terminal E113-R2 cells with their frozen E78/E79
controls, with exact per-domain/per-family `n`. Failed or incomplete cells are
shown as such and are never silently omitted. No pooled DAPO efficacy claim is
licensed from an incomplete surface; operational feasibility is reported
separately from endpoint effects.

The E113-R1 Graph outcomes are launch validation only and never enter a
scientific DAPO effect estimate.
