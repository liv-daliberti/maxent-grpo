# E104 Qwen-3B update-only capacity preflight

**Frozen before this replacement preflight was submitted, before any
Qwen2.5-3B E104 mechanism job started, and before any post-update E104
evaluation result was inspected on 2026-08-17.**

The original A6000 capacity preflight, job 30638185, was cleanly preempted
twice under partition `lowprio`: after 99 seconds on its first allocation and
after 3 minutes 38 seconds on its second.  Both allocations exited with code
zero before optimizer update 1; the outcome-blind auditor observed at most
training step 0 and no failure marker.  The second allocation spent its useful
window in the ordinary initial evaluation, which is irrelevant to the stated
capacity question.

This replacement therefore tests the narrow question directly: can the exact
frozen E104 Qwen2.5-3B model, Graph seed-70 training surface, optimizer, group
size, repaired semantic objective, and verified replay objective complete one
finite optimizer update on a 48GB A6000?  A versioned ops overlay differs from
the E104 snapshot in exactly two shell files.  When and only when the
capacity-preflight flag is set, it passes OAT's documented `--debug` switch
(`debug` means “skip the first evaluation”) and resolves `eval_steps=0`.
Terminal model export and recovery checkpoints are also disabled.  These
changes remove evaluation and persistence work only; they do not alter model
initialization, rollout generation, verification, replay, loss construction,
backpropagation, optimizer settings, or update 1.

The job uses partition `lowprio`, account `mltheory`, and one A6000 from
`node[103-104,205-208,805]`.  It passes only after normal scheduler completion
with update 1 recorded; all training values must be finite, v6 group centering
must be active, legacy semantic and RMS-controller paths must be off, centering
and magnitude bounds must hold, replay must be applied, no evaluation artifact
may exist, and neither log may contain a registered failure marker.  The audit
reads training telemetry and file names only, never evaluation values.

If this replacement passes, it supersedes the incomplete original capacity
preflight as the placement proof for the still-pending five Qwen2.5-3B E104
mechanism cells.  It cannot replace an E104 outcome cell, does not amend E105,
and includes no PointMaze run.

The incomplete original job may be held while this replacement is pending or
running, preventing duplicate capacity work.  It must be released if the
replacement fails and may be cancelled only after the replacement passes its
registered audit.  Holding or cancelling it changes no observed training
result because it has not reached optimizer update 1.
