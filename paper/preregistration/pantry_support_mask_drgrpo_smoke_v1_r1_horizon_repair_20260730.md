# PantryPlan six-bit Dr.GRPO smoke v1-r1 horizon repair

**Status: FROZEN AFTER THE V1 INFRASTRUCTURE FAILURE AND BEFORE ANY REPAIR RUN — 2026-07-30**

## Observed failure

The frozen v1 smoke retained job ID `30187473` and began normally on an
A5000. Its step-zero evaluation completed with greedy accuracy `0.359375`.
Before the first optimizer update, the learner-side canonical sampler stopped
at `src/oat_drgrpo/learner/run.py:1694` with
`RuntimeError: canonical learner sampler requires three supports`.

This is an infrastructure failure. The frozen Pantry policy has six binary
positions by construction, and no sampled training reward or optimizer update
occurred. Job `30187473` remains part of the provenance and cannot be reported
as a scientific Pantry result.

## Prospectively allowed repair

The replacement may change only generic finite-policy plumbing:

1. accept any positive canonical horizon whose positional supports are all
   nonempty, instead of requiring horizon three;
2. enumerate exact post-update entropy over every frozen position;
3. compute the expected prefix-row count as the sum of prefix products across
   every nonterminal depth (63 for six binary positions);
4. emit the same explicit Pantry task-identity telemetry as the actor path;
5. add regressions for six-step learner sampling, 64 leaves, 63 prefix rows,
   and the Pantry telemetry flag.

No prompt, data row, split, seed, learning rate, rollout count, optimizer-update
count, evaluation cadence, verifier, reward, pass threshold, or MaxEnt setting
may change. The replacement remains plain Dr.GRPO with seed `76201`, 16
rollouts per prompt, and 32 optimizer updates. It must use a fresh source
snapshot, identity, submission receipt, run directory, audit receipt, and Slurm
job ID while retaining the failed-v1 hashes and job ID as antecedents.

The replacement may directly apply the already-frozen placement-only amendment:
partition `all`, no fixed node, and one A5000 instead of the original pinned
node302 A100. CPU count, memory, wall time, account, and every scientific
environment variable remain unchanged. This placement change occurred before
runtime for both attempts and is recorded separately from the scientific
identity.

## Authorization boundary

The repaired smoke can authorize only the already-frozen paired Pantry
MaxEnt-mechanism smoke. It is not one of the final five seeds and cannot by
itself authorize the ten Pantry paper runs.
