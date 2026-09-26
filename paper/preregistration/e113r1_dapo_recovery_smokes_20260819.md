# E113-R1: DAPO operational-smoke recovery

Date frozen: 2026-08-19, after both original E113 operational smokes failed
and before submitting any E113-R1 job.

## Status and boundary

The original E113 record is immutable. Qwen job 30736381 exhausted ten
all-zero groups before its first optimizer update; Falcon job 30736382 made
five accepted updates and then exhausted ten all-zero groups. Their 50
dependent scientific jobs are `DependencyNeverSatisfied`. None of those jobs
is an efficacy result, and E113-R1 does not reinterpret or replace them.

The failures were the registered DAPO dynamic-sampling guard, not a numerical
loss, clipping, verifier, or distributed-runtime error. The original smoke
used Countdown. Completed pre-E113 controls show step-zero sampled per-response
correctness of 0.010986 for Qwen2.5-0.5B and 0.036621 for Falcon3-1B on
Countdown, versus 0.199219 and 0.178223 on Graph coloring. A one-prompt
operational smoke on Countdown therefore tests rare reward support more than
it tests whether the DAPO optimizer path runs correctly.

## Recovery smoke

E113-R1 changes only the two non-scientific operational smokes:

- model families: Qwen2.5-0.5B and Falcon3-1B;
- domain: Graph coloring;
- seed: the first registered family seed (Qwen 43; Falcon 55);
- maximum accepted updates: 32;
- DAPO generation-batch ceiling: 10, unchanged;
- rollout group size: 16, unchanged;
- maximum sampled rows: `32 * 16 * 10 = 5120` per smoke;
- optimizer, clipping, token-level loss, overlong shaping, model revisions,
  data, prompts, verifier, and frozen runtime snapshot: unchanged from E113;
- output directories, run stamps, job names, and ledger: fresh E113-R1 paths.

The original 640-row smoke ceiling could cover only 40 generated prompt
groups, not the registered worst case for 32 accepted updates. E113-R1 binds
the ceiling to the actual 32-update smoke contract.

## Pass rule

A family smoke passes only if all of the following hold:

1. Slurm exits zero and writes an `oat_zero_training_complete_v1` receipt with
   terminal step 32.
2. Exactly 32 accepted DAPO update records are present.
3. Every accepted record reports dynamic sampling enabled, one accepted group,
   at most ten generation batches, positive token-level active-token count,
   DAPO clipping 0.20/0.28, and finite policy loss and gradient telemetry.
4. Sampled-row use does not exceed 5,120.

These smokes are operational and must never enter paper estimates. This
revision intentionally submits no scientific cell and rewires no original
dependency. A later scientific recovery requires a separate prospective
amendment after both smoke audits pass and after explicitly addressing the
near-zero initial support of the Python cells and the one-group adaptation's
long-run exhaustion risk on Countdown and MathIR.

## Integrity contract

- Submit both jobs held, audit the scheduler-expanded environments, write one
  atomic ledger, and only then release the two smokes.
- Reuse the exact E113 content-addressed runtime snapshot; do not silently
  patch the DAPO learner.
- Bind this amendment, the recovery launcher, and the original E113 ledger by
  SHA-256.
- Refuse a pre-existing recovery ledger or output directory.
- On partial submission or audit failure, cancel only newly submitted E113-R1
  jobs and do not write a released ledger.
- Do not submit or release scientific DAPO work from this launcher.

