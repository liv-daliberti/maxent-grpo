# Amendment 4: resolve conflicted initial checkpoints by attempt provenance

This amendment follows amendments 1 and 2 of the September 11 audit and
amendment 3 of the September 12 complete-Pantry extension, whose cache,
receipt and result it leaves unchanged. It is frozen after that analysis and
before any concentration effect is computed from a newly admitted initial
checkpoint. It
is explicitly after results. It is not preregistration, and it is not an
independent confirmatory replication. It was adopted after observing which
cells lacked an admissible step-0 endpoint, and with knowledge that resolving
them would complete before/after blocks that the manuscript currently reports
as incomplete. That ordering is the reason for the disclosure requirements in
"Reporting" below.

## Diagnosis

A conflicted step-0 endpoint is not a missing evaluation. The job watchdog can
requeue a run; the restart reports `auto_resume=no checkpoint found; starting
from initialization`, evaluates the initial model again, and appends a second
step-0 block to the run's `eval_mode_coverage_draws.jsonl`. A run can also
carry several `debug_job*` directories, one per attempt.

The competing blocks disagree because separate attempts can execute on
different GPU models, which changes floating-point reduction order and
therefore the sampled tokens. The request contract does not differ: the
affected blocks record identical benchmark, evaluation kind, prompt count,
sample count, draw seed, temperature and top-p. The same hardware sensitivity
is why `ops/exp_scaling/launch_e72_decoding_frontier.py` pins every decoding
cell to the node that trained its checkpoint.

Re-running an evaluation does not adjudicate between two existing attempts; it
appends a third.

## Rule

For a cell whose step-0 endpoint is currently `conflicted_or_invalid`:

1. Consider each frozen source file of the cell. Discard any file that contains
   no training-step evaluation: it belongs to an attempt that trained nothing.
2. Within each remaining file, partition the step-0 rows into blocks at each
   deterministic greedy-trace row. A block is *continued* when the next
   evaluation row in file order is a training step.
3. Admit the cell's step-0 endpoint only when exactly one continued block
   exists across all of the cell's retained files, and admit that block.
4. Otherwise leave the endpoint unavailable, as now.

The rule reads only file order, step numbers, draw indices and the presence of
training steps. It never reads a metric value, a key, a collision estimate or
an effect, and it cannot prefer one attempt over another by outcome. The
admitted block is the evaluation of the initial model performed by the attempt
whose trajectory the analysis already uses.

## Fixed affected population

Determined by source structure before any effect is computed. Of the 400
registered cells, 363 already have an admitted step-0 endpoint and one lacks it
without a conflict. Of the remaining 36 conflicted cells, the rule resolves
**27** and leaves **9** unavailable because more than one continued block
exists. The 27 are fixed here and may not be extended or trimmed in response to
any subsequent estimate:

    level1/falcon1b/countdown/maxrl/56,58,59
    level1/falcon1b/countdown/replay_maxrl/56,57,58,59
    level1/falcon1b/graph_coloring/maxrl/55,58
    level1/falcon1b/graph_coloring/replay_maxrl/55,58,59
    level1/falcon1b/mathir/maxrl/55,56,58,59
    level1/falcon1b/mathir/replay_maxrl/55,58,59
    level1/falcon1b/pantry_plan/maxrl/56,57,58,59
    level1/falcon1b/pantry_plan/replay_maxrl/56,57,59
    level1/qwen3b/mathir/replay_drgrpo/71

The population spans four domains, both MaxRL arms and one Dr.GRPO-family cell,
including cells in domains whose conditional contrast the manuscript does not
report. It was not selected for the blocks whose completeness motivated the
amendment.

## What had already been observed when this was frozen

Stated so the disclosure is exact. Before freezing, the analyst had seen: the
block and file structure of the conflicted cells; the per-draw
`any_correct_at_k` of the competing step-0 blocks; the scheduler and learner-log
provenance, including `restart_count` and requested nodes; and the GPU model
recorded for each campaign.

The analyst had **not** computed, from any newly admitted step-0 block, a
conditional-collision estimate, a \pmd{} value, a before/after contrast, a
disjoint-stream orientation, or any aggregate derived from them.

## Reporting

Preserve the amendment-3 cache, receipt and result unchanged, exactly as
amendment 3 preserved its predecessors. Write a separate cache and receipt for
this admission. Report the before/after contrast under **both** admissions
wherever the manuscript states a MaxRL longitudinal result, together with the
defined-seed count under each. State in the manuscript that the rule was
adopted after the original analysis was specified. Do not describe the
post-amendment population as the originally analyzed one, and do not present
the change as a confirmation of anything.

If applying the rule reverses, weakens or leaves undetermined any direction the
manuscript currently reports, report that outcome under the same rule. The rule
may not be revised, narrowed or abandoned in response to what it produces.

## Constraints carried forward

Amendments 1 to 3, the estimator, the eligibility threshold, the
support bar, the orientation definitions and every uncertainty convention are
retained unchanged. No new model call, training run, checkpoint selection,
grading pass or outcome definition is permitted. Newly admitted blocks are
ingested through the existing frozen-prefix hashing and the unchanged raw
response normalizer.
