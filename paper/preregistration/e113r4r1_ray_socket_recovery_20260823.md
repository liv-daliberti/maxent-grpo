# E113-R4-R1: Ray socket-path infrastructure recovery

Date frozen: 2026-08-23, after both R4 operational smokes failed and before any
R4 scientific cell ran or any replacement job was submitted.

## Trigger and diagnosis

Official-R4 smoke jobs `30800804` (Qwen-0.5B) and `30800805` (Falcon-1B)
allocated on node208 and failed before training. Both stderr logs report the
same deterministic infrastructure exception from `ray.init`:

`OSError: AF_UNIX path length cannot exceed 107 bytes`

The runner set `RAY_TMPDIR` below each long experiment output path. Ray then
appended its session and socket components, producing an invalid Unix-domain
socket pathname. The Qwen smoke ran for 66 seconds and the Falcon smoke for 12
seconds; neither wrote a completion receipt or scientific checkpoint. Their
failed joint `afterok` gate caused Slurm to cancel all 50 dependent science jobs
at zero runtime and without creating any science output directory.

## Authorized infrastructure-only change

R4-R1 may change only process-local temporary placement:

- create a unique directory matching `/tmp/e113r4-${SLURM_JOB_ID}-XXXXXX` on
  the allocated node;
- set both `TMPDIR` and `RAY_TMPDIR` to that short directory;
- bind that directory into the existing pinned Apptainer image; and
- remove it when the job exits.

The original immutable runtime snapshot remains preserved. A new immutable
snapshot must be copied from it and differ only in
`ops/run_e113r4_official_dapo.sh` plus snapshot identity metadata. In
particular, the upstream verl commit, image, verifier, models, data, reward
function, prompts, seeds, DAPO objective, response budgets, optimizer, batch
geometry, stopping rule, and 50-cell estimand must not change.

## Recovery graph

Submit two fresh one-step operational smokes, one per model family, on the
previously approved non-preempting A6000 pool in partition `all`, account
`allcs`, with 16 CPUs, 128 GiB, `Nice=0`, and a one-day limit. Submit fresh
copies of the exact 50 science cells with their original seven-day limit and
default priority, jointly gated by `afterok` on both replacement smokes. The
science jobs may be released into dependency-pending state immediately; no
science job may allocate unless both repaired smokes exit successfully.

The 50 canceled job IDs and two failed smoke IDs remain provenance only and
must not be counted as R4-R1 efficacy cells. The authoritative ledger may
replace its active `smokes` and `runs` records only after all 52 new jobs pass a
held-state audit, while retaining the complete superseded records under a
durable recovery amendment.

## Interpretation boundary

This repair responds only to an observed pathname failure and is blind to any
scientific endpoint because none exists. Smoke success licenses execution, not
an efficacy claim. Scientific reporting remains based only on terminal
replacement cells and their already frozen paired controls.
