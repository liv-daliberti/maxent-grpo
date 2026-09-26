# E117-A3 executable-audit closure

Frozen: 2026-08-25 while all 12 E117-R1 training jobs and audit job 30874411
remain pending at zero runtime, before any E117 run artifact or endpoint exists.
PointMaze remains excluded. This amendment changes only the dependency-pending
audit; it does not alter a training job or scientific configuration.

## Trigger

Testing the A2 audit against the runtime's JSONL format revealed that every
sampled mode-coverage evaluation has a deterministic greedy companion row.
The A2 selector counted all step-zero rows before checking the registered
sampled-draw count, so a correct evaluation would falsely report two draws
instead of one. A subsequent fail-closed review found several places where the
executable proof could be made exact rather than implicit.

These issues were found from source, schemas, and non-E117 mechanism telemetry.
No E117 output, endpoint, contrast, or efficacy statistic existed or was used.

## Frozen corrections

E117, E117-A1, and E117-A2 otherwise remain unchanged.

1. At step zero, require exactly one
   `deterministic_greedy_trace_neutral` companion with `draw_index=null` and
   exactly one registered `fixed_seed_sampled_k_neutral` draw with integer
   `draw_index=0`. Reject any other step-zero evaluation kind. Compare the
   complete two-row record exactly across C/P/F within a sentinel. Only the
   sampled row counts toward `evaluation_draws=1`.
2. Common-arm identity fields use the same strict numeric parser as mechanism
   fields. A field missing in all three arms is a failure, not three equal
   nulls. Present optimizer steps and evaluation steps must be finite,
   non-Boolean integers; duplicate or incomplete 1--64 optimizer coverage
   remains a failure.
3. When an online or conditional replay-scoring retention view is present,
   require its complete runtime diagnostics schema, not only the subset used
   in a contrast. Counts must be nonnegative integers, fractions must lie in
   `[0,1]`, registered no-feedback/no-adaptive counters must be zero, and every
   lifecycle counter that is cumulative by construction must be monotone. The
   required field set is regression-tested against
   `AdmissionRetentionTracker.diagnostics()`; that source is byte-identical in
   the immutable E117 training snapshot.
4. Proposal candidates, admissions, stored exemplars, discards, and cumulative
   admissions must be nonnegative integers. The cumulative admission count
   must equal the running sum after every update, not only at step 64, and the
   online tracked-admission count must equal it in P/F.
5. The verified-support centered signal is coefficient-free in `[-1,1]` and
   then scaled by the registered coefficient. Therefore C/P raw and effective
   RMS must both be zero; F raw RMS must not exceed 0.10; and effective RMS
   must not exceed raw eligible RMS (allowing only float32 representation
   tolerance). E117-A1's positive-effective-RMS rule on every positive-raw-RMS
   F update remains unchanged.
6. If any dependency is failed, canceled, missing, or otherwise not completed
   successfully, write a durable version-3 failed audit artifact with
   `passed=false` and Stage-1 readiness false before returning nonzero. Do not
   leave failure evidence only in a Slurm log.
7. Report resume continuity as `verified` only when a restarted job passes all
   identity/state checks, `failed` when a restarted job does not, `unexercised`
   when none restarted, and `not_auditable` when training did not complete.

## Installation boundary

The corrected audit must pass its adversarial tests and the full focused
E105/E109/E112/E117 regression suite. Freeze it in a new content-addressed
source snapshot. Submit one no-requeue `afterany` audit depending on exactly
jobs 30873695--30873706, validate its held record and batch script, and only
then cancel zero-runtime audit 30874411. Preserve the A1/A2/A3 chain, all audit
job IDs and source/protocol digests, `training_jobs_changed=false`, and
`outcomes_inspected=false`.
