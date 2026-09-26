# E117-R2 same-plumbing runtime and audit repair

Frozen: 2026-08-29, after the mechanism-only E117-R1 audit diagnosis and before
E117-R2 submission or inspection of any E117 efficacy endpoint.

Status: prospective repaired mechanism preflight. This amendment does not alter
or reinterpret E112-R1, does not make E117-R1 an efficacy experiment, and does
not authorize Stage 1. PointMaze remains excluded.

## Evidence boundary and diagnosis

All 12 E117-R1 cells completed 64 genuine optimizer updates. Inspection was
limited to scheduler, identity, leakage, proposal, semantic-pressure, retention,
and logging telemetry. No endpoint contrast, arm ranking, confidence interval,
or efficacy statistic was computed or used.

E117-R1 identified two implementation defects:

1. the explicit semantic zero-coefficient control passed argument validation but
   learner initialization constructed the estimator only when the coefficient
   was positive, so C/P did not traverse the registered estimator or emit its
   required telemetry; and
2. the audit keyed rows only by `misc/global_step`, causing the post-training
   bookkeeping row (`trainer/step=65`, optimizer/global/policy step 64) to be
   mistaken for a second optimizer update.

All nonsemantic identity, isolation, admission, leakage, and retention checks
passed in a diagnostic that did not modify the official artifacts. P admission
and F semantic pressure were exercised in every sentinel. These facts motivate
only the repairs below, never an efficacy interpretation.

## Frozen repair

The new content-addressed source snapshot differs from the E117-R1 runtime only
through the following generic correctness repairs and contemporaneous source
already present in the repository:

1. construct `SemanticShannonTracker` when the coefficient is positive **or**
   `semantic_shannon_allow_zero_coefficient_control` is true;
2. keep a coefficient-zero semantic advantage positive bitwise zero while still
   executing the estimator, updating its checkpointed state, and emitting the
   complete signed/verified-support diagnostic namespace;
3. serialize that complete diagnostic namespace through one shared runtime
   adapter used by the learner and the pre-submission smoke; and
4. classify exactly one canonical optimizer row per step using
   `trainer/step == trainer/global_step == misc/global_step ==
   misc/policy_sgd_step`, while accepting exactly one terminal bookkeeping row
   only at step 64 with `trainer/step=65`. The terminal row must preserve every
   actor/train mechanism field plus policy, query, prompt-consumption, and beta
   state. A second canonical row, absent sentinel, unrecognized row, or mutated
   sentinel fails closed.

No proposal budget, replay weight, semantic coefficient, optimizer, sampler,
data, seed, evaluation request, model, or estimand changes.

## Snapshot-bound pre-submission smoke

Before any training job is submitted, run the smoke executable from the new
snapshot itself on one deterministic synthetic verified-support group. C/P/F
must construct the same estimator, emit exactly the same complete metric-key
set, and update identical estimator state apart from coefficient. C/P must have
coefficient, raw RMS, effective RMS, and every policy-advantage value positive
bitwise zero; adding that advantage must leave a nonzero task-advantage tensor
bitwise unchanged. F must have coefficient 0.10 and positive RMS. The smoke
writes `var/artifacts/e117r2_same_plumbing_runtime_smoke.json`. Any failure
prevents all Slurm submissions.

## Repaired cells

Rerun the complete 12-cell C/P/F matrix rather than combining repaired controls
with the old F cells. Use seed 117, 64 distinct training prompts, one pass, 64
optimizer updates, checkpoints at 32, and the exact E117 objective and request
surfaces.

| Scale | Domain | C/P/F physical node | GPU class |
|---|---|---|---|
| Qwen-0.5B | Countdown | node202 | A5000 |
| Qwen-0.5B | Graph coloring | node203 | A5000 |
| Qwen-0.5B | Python factors | node203 | A5000 |
| Falcon-1B | MathIR | node203 | A5000 |

Cells within a sentinel may serialize. Start time and wall-clock time are not
estimands. The only within-block environment differences remain run identity,
the C admission-discard flag, and the F coefficient.

Fresh run stamps begin with `e117r2_`. The immutable release ledger is
`var/artifacts/e117r2_same_plumbing_component_preflight_jobs.json`. It records
the R1 parent ledger and audit digests, protocol, source snapshot, smoke receipt,
held scheduler records, exact exported-environment hashes, physical-node map,
and audit dependency.

## Official audit and release boundary

Submit one no-requeue mechanism-only audit with `afterany` dependencies on
exactly the 12 R2 job IDs. It must run the repaired audit from the same frozen
snapshot and write
`var/artifacts/e117r2_same_plumbing_component_preflight_audit.json`.

The full E117, E117-A1, E117-A2, E117-A3, and E117-A4 rules remain binding.
Missing telemetry is never imputed. The official audit passes only if every
identity, request-stream, C admission/no-mutation, P/F admission, retention,
leakage, semantic coefficient/pressure, step coverage, terminal-sentinel, node,
and provenance check passes. Stage-1 readiness additionally requires positive
P admission and positive F effective pressure in at least one sentinel.

No endpoint values may be inspected to release, repair, stop, prioritize, or
interpret R2. Stage 1 remains unauthorized unless the terminal official R2
audit reports both `passed=true` and
`stage1_execution_readiness.ready=true`. A failed R2 audit requires another
prospective diagnosis; it is never waived by the synthetic smoke or by E117-R1.
