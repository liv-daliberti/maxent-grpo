# E117-R1-S5 user-hold release

Frozen: 2026-08-28, before releasing the affected jobs.

Status: scheduler-only continuity amendment. This amendment does not change
E117's scientific cells, arm definitions, model/data assignments, seed,
training horizon, evaluation surface, analysis contract, or advancement gate.

## Trigger

The Falcon-1B MathIR C/P/F block is training-terminal. The other nine frozen
E117 scientific jobs are still present in Slurm with `PENDING`,
`Reason=JobHeldUser`, zero runtime, and zero restarts:

`30873695`, `30873696`, `30873697`, `30873698`, `30873699`, `30873700`,
`30873701`, `30873702`, and `30873703`.

The replacement audit job `30874713` is also user-held. Its surviving
dependency is `afterany` on the nine nonterminal scientific jobs, so releasing
it cannot run the audit before those jobs terminate.

## Authorized action

Release exactly the nine scientific job IDs above and audit job `30874713`
with `scontrol release`. Do not release, reprioritize, requeue, cancel, or
otherwise modify any E112, E113/DAPO, E116, or unrelated job. Afterward,
verify that the scientific jobs are no longer held and that the audit job
remains dependency-blocked until the scientific jobs terminate.

This action uses scheduler state only. No pending-cell endpoint exists to
inspect, and no endpoint value is used to justify the release.
