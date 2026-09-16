# E113-R1-S1: Falcon post-receipt shutdown classification

**Frozen:** 2026-08-19 16:10 EDT, after Falcon job 30790112 terminated and
before any Qwen M1 outcome or scientific DAPO relaunch.

## Observed ordering

The Falcon Graph recovery learner completed all 32 registered optimizer
updates, completed its step-32 evaluation, and wrote an
`oat_zero_training_complete_v1` receipt with outer runner terminal step 33 and
terminal policy/global optimizer step 32. This is the runner's established
target-plus-one receipt convention: for example, completed 3,072-update jobs
write terminal step 3,073. Falcon's cumulative query count was 704, safely
below 5,120. The metrics file contains 32 unique `trainer/global_step` values;
a 33rd row is the runner's duplicate terminal summary at `trainer/step=33`
with policy global step still 32 and represents no additional optimizer
update.

Only after these events, actor teardown raised `Trying to free a pointer not
allocated here` in `CUDAPluggableAllocator::raw_delete`, followed by `Fatal
Python error: Aborted`. Slurm therefore recorded job failure with exit 137 even
though the training completion receipt and full registered trajectory already
existed.

## Classification rule

For the non-scientific operational gate only, Falcon is classified as
**training complete, shutdown failed** if a fail-closed auditor verifies all of
the following:

1. the original immutable R1 job ID is 30790112;
2. its completion receipt has outer runner terminal step 33, while the
   terminal policy/global optimizer step is exactly 32;
3. exactly the unique optimizer-step set 1 through 32 passes every DAPO metric,
   clipping, finiteness, token-count, generation-batch, and query-ceiling rule;
4. the raw 33rd accepted-looking metric row duplicates global step 32;
5. step-32 evaluation/log completion precedes the allocator teardown error in
   the immutable log; and
6. scheduler accounting is `FAILED` with exit `137:0`.

This amendment does not relabel the Slurm job as successful and does not relax
any learner or DAPO criterion. It prevents a post-receipt process-cleanup bug
from discarding a fully completed operational trajectory. The Falcon smoke is
never an efficacy datum and is not rerun.

The downstream effective gate additionally requires a separately terminal,
exit-zero Qwen E113-R1-M1 smoke. No scientific job may launch from this
classification alone.
