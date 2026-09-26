# E105 amendment: matched Qwen-3B A6000 placement

**Frozen on 2026-08-17 before E105 submission, before any E105 training or
evaluation artifact existed, and without inspecting an E80-R1 evaluation
value or any post-update E104/E106 outcome metric.**

## Motivation and evidence boundary

The Qwen2.5-3B ReplayDr.GRPO comparator cohort E80-R1 is pinned to the single
A100 host `node302`. At freeze time, Graph and Countdown were complete, three
Python pairs had already completed or started, and the following ten matched
control/replay pairs were both still `PENDING` at runtime `00:00:00`:

- Python Factors seeds 73 and 74;
- MathIR seeds 71, 72, 73, and 74; and
- PantryPlan seeds 71, 72, 73, and 74.

These pairs comprise twenty existing E80-R1 jobs. Their selection used only
the immutable ledger and scheduler state, never an evaluation value. Moving
only ReplayDr.GRPO would confound E80-R1 by arm, so an eligible pair always
moves both its control and replay jobs or neither.

The subsequently frozen E109 amendment supersedes only the Python-comparator
part of that plan: E105 Python Factors now pairs to repaired-parser E109, not
to historical E80-R1 Python. Therefore the four historical E80-R1 Python jobs
for seeds 73 and 74 remain unchanged. Those two seeds are instead assigned
prospectively to the A6000 pool for both members of the actual pair—E109
ReplayDr.GRPO and E105 repaired Semantic MaxEnt—before either is submitted.
The eight MathIR/PantryPlan historical pairs remain eligible for the
transactional E80-R1 move described below.

The completed Qwen-3B A6000 capacity job `30638185` established that the exact
3B model, optimizer/offload recipe, rollout group size, 128-GiB request, and v6
loss stack can complete an optimizer update on a 48-GB A6000 without OOM. It
did not satisfy its separate one-update replay-application criterion, and this
amendment does not relabel that audit as a pass. Authorization instead waits
for the complete E104+E106 mechanism gate, including the real 64-update
Qwen-3B Python job `30640331` on the same A6000 pool. That combined gate must
be complete and passing, must show a live nonzero v6 semantic update at every
scale, must show verified replay and repaired Python admission in every E106
cell, and must remain outcome-blind.

## Conditional scheduler-only change

After, and only after, the combined gate passes, each non-Python candidate E80-R1 pair is
rechecked. A pair is eligible only if both jobs still have the original
scientific environment, remain `PENDING`, and retain runtime `00:00:00` on
`mltheory`, `node302`, and one A100. An ineligible pair is skipped as a whole.
Every eligible pair may be changed together to:

- partition `lowprio`, account `mltheory`;
- node pool `node[103-104,205-208,805]`;
- one `gpu:a6000`, 16 CPUs, and 128 GiB memory; and
- the existing three-day time limit, priority, checkpoint cadence, requeue,
  auto-resume, and output paths.

The application is transactional: any failed update or post-update audit rolls
all jobs changed by that invocation back to their original A100 placement. A
hash-bound amendment artifact records candidate, moved, and skipped pairs plus
the before/after scheduler records. It separately records the two prospective
E109/E105 Python assignments; these cause no update to an existing E80-R1 job.
It reads no training or evaluation file.

E105 then follows its already-frozen rule that each treatment inherits its
matched replay comparator's placement. A non-Python Qwen-3B E105 cell uses the
A6000 pool if and only if its exact E80-R1 pair appears in the validated
moved-pair artifact. Python seeds 73 and 74 use A6000 if and only if they appear
in the prospective E109/E105 pair assignment. All other Qwen-3B E105 cells
remain on `mltheory`/`node302`/A100.
Qwen-0.5B and Falcon placement is unchanged.

This changes no model, source or ops snapshot, data, domain, prompt, seed,
optimizer, decoding surface, group size, objective, coefficient, stopping
rule, checkpoint, evaluation cadence, paired estimator, or analysis rule.
Hardware is matched within every affected E80-R1 control/replay pair, every
E105 treatment/replay pair, and specifically each repaired E109/E105 Python
pair. PointMaze remains excluded.
