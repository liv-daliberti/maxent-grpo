# E104 Qwen-3B update-only capacity retry

**Frozen before this retry was submitted, before any Qwen2.5-3B E104
mechanism job started, and before any post-update E104 evaluation result was
inspected on 2026-08-17.**

Update-only preflight job 30638612 failed before model initialization because
its launcher encoded disabled recovery as `resume_steps=0`; the frozen
validator requires `-1` or a positive interval.  The job exited 1, created no
run directory or evaluation artifact, and executed no optimizer update.  Its
ledger, audit, logs, launcher, protocol, and v1 overlay remain unchanged.

This retry changes only that invalid sentinel.  Its separately versioned ops
overlay is byte-identical to the v1 update-only overlay except that the bounded
capacity-preflight branch exports `OAT_ZERO_RESUME_STEPS=-1`.  As registered
for the failed attempt, the same branch skips initial and terminal evaluation,
while export and recovery persistence remain disabled.  Model initialization,
rollouts, verification, replay, repaired semantic loss, backpropagation,
optimizer settings, and the single target update are unchanged from E104.

The same outcome-blind pass rule applies: normal completion after one finite
optimizer update, active v6 group centering, inactive legacy semantic and RMS
controller paths, centered and bounded semantic advantages, applied verified
replay, no evaluation artifact, and no registered failure marker.  It uses
partition `lowprio`, account `mltheory`, and one A6000 from
`node[103-104,205-208,805]`.  A pass supersedes both incomplete predecessors
as the placement proof only; it cannot replace an E104 outcome cell, does not
amend E105, and includes no PointMaze run.
