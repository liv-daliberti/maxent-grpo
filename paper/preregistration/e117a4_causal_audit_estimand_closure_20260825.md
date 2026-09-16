# E117-A4 causal-audit estimand closure

Frozen: 2026-08-25 while all 12 E117-R1 training jobs and audit job 30874565
remain pending at zero runtime, before any E117 run artifact, endpoint, or
contrast exists. PointMaze remains excluded. This amendment changes only the
dependency-pending audit; it does not alter a training job, placement, or
scientific configuration.

## Trigger

The final source-to-telemetry review separated two quantities that the A2/A3
audit had treated as one. Every arm unconditionally issues the same fixed
proposal-shaped request, but whether the explorer consumes that frozen group
is a policy-dependent mediator: after C/P/F policies diverge, an arm may have
no valid anchor and legitimately consume no proposal group. Requiring proposal
consumption to remain equal at every update would therefore condition the
identity proof on a post-treatment event and could reject a correct run.

The same review closed terminal-evaluation completeness, scientific
provenance, telemetry-algebra, and executable success-path gaps. All changes
were derived from frozen source, scheduler records, schemas, and synthetic
fixtures. No E117 output or efficacy outcome was available or inspected.

## Frozen corrections

E117 and amendments A1--A3 otherwise remain unchanged.

1. At every optimizer step, compare across C/P/F the neutral sampling request
   seed and the unconditional fixed-control request identity: one group,
   16 rows, positive charged response-token budget, equal request-seed
   min/max, common request seeds, common sampling temperature, and zero fixed
   rows sent to PPO. Require fixed-request seeds not to repeat within a run.
2. Within each arm and update, require consumed plus discarded fixed groups
   to reconcile exactly, proposal groups to equal consumed groups, and
   proposal rows to equal 16 times proposal groups. Proposal request seeds
   must equal that arm's fixed-request seed when a group is consumed and must
   be absent when no group is consumed.
3. Compare proposal consumption, proposal request seeds, proposal/result row
   accounting, validation surface, and neutral reward exactly across C/P/F
   only at step 1, before any treatment can affect the policy. Do not require
   equality of those post-treatment mediators at steps 2--64.
4. Require exactly 16 neutral rows to reach PPO at every update. Reconstruct
   the neutral task-positive row count from `actor/rewards * 16` within only
   serialization tolerance. Proposal, fixed-control, and transformed rows
   remain prohibited from PPO, and every registered no-feedback/no-objective-
   support leakage field remains exactly zero.
5. Require complete evaluation records at exactly steps 0 and 64: one greedy
   companion and the one registered sampled draw, each with nonempty metric
   and prompt payloads. Step-zero records, including outcomes, must be exactly
   common across C/P/F. At step 64 compare only request metadata after removing
   `metrics` and `prompts`; terminal efficacy values are neither compared nor
   copied into the audit.
6. Verify each job's completed physical node against the effective registered
   node. Recompute the raw frozen scheduler-export digest, require the complete
   objective environment, source and ops roots, seed, row/pass/update/eval and
   checkpoint settings, and complete C/P/F environment equality after removing
   only the four intended path/arm-varying fields. Verify the immutable
   training snapshot identity and original E117 ledger digests.
7. Reject a JSONL row that is not an object or contains a nonfinite value at
   any recursive depth. Retention counts must also obey their runtime subset
   relations, reported fractions must reconstruct from their numerators and
   denominators, and total refresh requests must equal rollout plus score
   refresh requests.
8. Exercise the complete audit entry point with a synthetic 12-cell, 64-update
   success fixture, including step-zero and terminal evaluation files, frozen
   provenance, launch environments, and completed scheduler accounting.
   Retain the separate integration test proving failed dependencies emit a
   durable failed audit artifact.

## Interpretation boundary

`passed` remains an identity, isolation, leakage, state, provenance, and
continuity result. `stage1_execution_readiness.ready` remains a distinct
mechanism-activation condition requiring at least one P admission block and at
least one F effective-pressure block. Neither field is an efficacy endpoint,
and this audit computes no C/P/F endpoint contrast.

## Installation boundary

The A4 audit must pass formatting, compilation, its complete E117 tests, and
the focused E105/E109/E112/E117 regression suite. Freeze it in a new
content-addressed source snapshot. Submit one no-requeue `afterany` audit with
exact dependencies 30873695--30873706, validate the held record and batch
script, and only then cancel zero-runtime audit 30874565. Preserve the full
A1--A4 amendment and superseded-job chain, immutable protocol/source digests,
`training_jobs_changed=false`, and `outcomes_inspected=false`.
