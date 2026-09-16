# E113-R4-R2-S9: Qwen-0.5B host-memory rightsizing

Frozen: 2026-08-30 EDT after the user explicitly authorized the proposed
host-memory reduction, after inspecting scheduler memory accounting, and
before changing any live job. No task reward, validation accuracy, arm
contrast, or efficacy endpoint was inspected.

## Trigger and memory evidence

The five nonterminal Qwen2.5-0.5B Python-factors jobs `30977262`--`30977266`
are released, zero-runtime E113-R4 official-verl DAPO science jobs requesting
one A6000, 16 CPUs, and 128 GiB of host memory each. Contemporary A6000 nodes
have unallocated GPUs that cannot accept another 128-GiB allocation because
host memory, rather than accelerator count, is the packing constraint.

All 20 authoritative terminal Qwen2.5-0.5B E113-R4 cells completed with a
batch peak RSS below 64 GiB. The largest observed value is below 60,000,000
KiB (about 57.3 GiB). The earlier Qwen/Python filter-exhaustion attempts
`30869121`--`30869125` loaded the same model and runtime and exercised the
Python generation surface; their largest batch peak RSS is below 40,000,000
KiB (about 38.2 GiB). A 96-GiB request therefore retains more than 38 GiB of
headroom above the largest completed-Qwen observation and more than 56 GiB
above the largest prior Qwen/Python observation.

This evidence does **not** license a Falcon reduction. Historical Falcon
cells have reached approximately 127.6 GiB batch peak RSS, and the currently
pending Falcon/Pantry and Falcon/Python cells lack matching terminal evidence
for a lower bound. They remain at 128 GiB.

## Authorized scheduler-only transaction

For exactly jobs `30977262`--`30977266`:

1. Verify the authoritative ledger maps them to Qwen2.5-0.5B Python-factors
   seeds 43--47 and that each is pending, released, at zero runtime and zero
   restarts, with no dependency, no completion receipt, byte-exact scientific
   export, `Partition=all`, `Account=mltheory`, `QOS=long`, one A6000, 16 CPUs,
   128 GiB, and a 12-hour limit.
2. Temporarily user-hold all five jobs, then revalidate the same boundary.
3. Change only `MinMemoryNode` from 128 GiB to 96 GiB. Retain job IDs, account,
   partition, QOS, nice value, time limit, requeue policy, accelerator and CPU
   requests, command, output paths, run directories, and all exported
   scientific variables.
4. Validate the 96-GiB request while held, durably record the transaction, and
   release all five together. Accept pending or running post-release state.
5. If any pre-release check fails, restore 128 GiB for every changed job and
   release every temporary hold.

The temporary hold is a race-control mechanism only: it prevents a job from
allocating between preflight and its memory update. No running or completed
job may be held, requeued, canceled, or modified.

## Scientific invariants and exclusions

This amendment changes no model, family, domain, seed, official-verl source,
runtime image, data or hash, verifier, DAPO objective, filter cap, accepted
batch definition, optimizer, checkpoint/resume contract, evaluation request,
target update count, or stopping rule. It changes no GPU or CPU request and
does not inspect incomplete-cell outcomes.

Jobs `30977252`--`30977261` and `30978749`--`30978753`, every completed E113
cell, E112, E115, E116, E117, PointMaze, and unrelated campaigns are outside
this transaction.
