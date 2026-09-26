# Amendment 5: resolve conflicted registered checkpoints by attempt provenance

This amendment extends amendment 4 of the September 14 audit, whose binding,
cache, receipt and result it leaves unchanged, from the initial checkpoint to
the registered training grid. It is frozen after the training-curve figures were
drawn from the September 12 snapshot and before any curve, mean, seed range or
contrast is computed from a newly admitted mid-training checkpoint. It is
explicitly after results. It is not preregistration, and it is not an
independent confirmatory replication. It was adopted after observing which
checkpoints the published figures draw as gaps, and with knowledge that
resolving them would fill those gaps. That ordering is the reason for the
disclosure requirements in "Reporting" below.

## Diagnosis

Amendment 4 diagnosed a conflicted step-0 endpoint as a watchdog-requeue
duplicate: the restart reports `auto_resume=no checkpoint found; starting from
initialization`, evaluates the model again, and appends a second block to the
run's `eval_mode_coverage_draws.jsonl`. The competing blocks disagree because
separate attempts execute on different GPU models, which changes floating-point
reduction order and therefore the sampled tokens.

Nothing in that diagnosis is specific to step 0. A requeue that resumes from a
mid-training checkpoint re-evaluates that checkpoint and appends a second block
for it in exactly the same way, and a run can carry several `debug_job*`
directories, one per attempt. The frozen snapshot's checkpoint policy
invalidates the entire checkpoint when two blocks disagree, so every one of
these becomes a gap in the plotted curve.

The gap is then widened by the paired-cohort rule. A curve point is drawn only
when every seed in the fixed cohort reports, and paired arms are intersected, so
one seed's conflicted checkpoint removes that step from both arms of the pair.

Re-running an evaluation does not adjudicate between two existing attempts; it
appends a third.

## Rule

For a cell whose checkpoint at registered step *S* is currently
`conflicted_or_invalid`:

1. Consider each frozen source file of the cell. Discard any file that contains
   no evaluation row at a step later than *S*: it belongs to an attempt that
   never trained past *S*.
2. Within each remaining file, partition the rows at step *S* into blocks at each
   deterministic greedy-trace row. A block is *continued* when the next
   evaluation row in file order carries a step later than *S*.
3. Admit the checkpoint only when exactly one continued block of four draws
   exists across all of the cell's retained files, and admit that block.
4. Otherwise leave the checkpoint unavailable, as now.

At *S* = 0 this is amendment 4's rule verbatim, with "a training step" written
as "a step later than *S*". The rule reads only file order, step numbers, draw
indices and the presence of later training steps. It never reads a metric value,
a key, a collision estimate or an effect, and it cannot prefer one attempt over
another by outcome. The admitted block is the evaluation performed by the
attempt whose trajectory the analysis already uses.

## Fixed affected population

Determined by source structure before any curve is redrawn. The population is
fixed here and may not be extended or trimmed in response to anything the
redrawn figures show.

Of the 70 conflicted registered checkpoints outside step 0 in the September 12
snapshot, the rule resolves **27** and leaves **43** unavailable. The 43 are
unavailable for a reason worth stating plainly: more than one continued block
exists, because the run was requeued and more than one attempt trained past that
step. File order, step numbers and the presence of later training rows do not
distinguish them, and nothing else is permitted to. The 27 are:

    level1/falcon1b/countdown/replay_maxrl: seed 57 step 2304
    level1/falcon1b/pantry_plan/replay_maxrl: seed 55 step 2112
    level1/qwen3b/countdown/maxrl: seed 70 steps 960, 2880
    level1/qwen3b/countdown/replay_maxrl: seed 70 step 1728; seed 71 step 960; seed 72 step 1728
    level1/qwen3b/graph_coloring/maxrl: seed 71 step 1728; seed 72 step 1920; seed 73 step 1920
    level1/qwen3b/graph_coloring/replay_maxrl: seed 71 step 1536; seed 74 steps 960, 2688
    level1/qwen3b/mathir/maxrl: seed 74 step 1920
    level1/qwen3b/mathir/replay_maxrl: seed 72 step 960
    level1/qwen3b/pantry_plan/maxrl: seed 72 step 1920
    level1/qwen3b/python_factors/replay_maxrl: seeds 70, 72, 73, 74 step 1920
    level2/qwen05b/mathir/drgrpo: seed 47 step 384
    level2/qwen05b/pantry_plan/drgrpo: seed 44 steps 192, 384; seed 47 step 192
    level2/qwen05b/pantry_plan/replay_drgrpo: seed 44 step 1728
    level2/qwen05b/python_factors/drgrpo: seed 45 step 192
    level2/qwen05b/python_factors/replay_drgrpo: seed 46 step 384

Twenty are Level 1 and seven are Level 2, spanning five domains and both
objectives. Applied together with amendment 4's step-0 admissions, the
population completes six of the twenty-four broken Level-1 accuracy curves and
leaves eighteen broken. It does not close the Level-2 gaps that isolate a
pass-0 marker from its curve in Countdown and MathIR: every step blocking those
is one of the 43 ambiguous checkpoints. The amendment is therefore a partial
repair by construction, and the remaining gaps are reported as gaps.

## Composition with amendment 4

Amendment 4 expressed its admissions as a patch to a snapshot held in its own
audit directory, and the training-curve figures render from the snapshot in
`paper/results`, which never carried them. Its 27 step-0 admissions are
therefore applied here as well, by the rule of this amendment at *S* = 0, which
is amendment 4's rule verbatim. The population is amendment 4's, named in its
frozen binding and neither extended nor trimmed; applying the generalized rule
to it is a check as much as a mechanism, because a cell amendment 4 resolved
that this rule failed to resolve would mean the generalization is not faithful,
and the patch refuses rather than proceeding.

Amendment 4's own artifacts, including its patched snapshot, its cache, its
receipt and the conditional-concentration result computed from them, are left
untouched.

One defect in amendment 4's patch program is recorded here rather than
corrected there. Its `admitted_checkpoint` copies `prompt_count` from the raw
draw row, where that field does not exist; the snapshot derives it from the
length of the prompt list. Its output was validated through the collision
loader, which does not read that field, so the omission never surfaced. It
surfaces immediately against `validate_checkpoint`, which the training-curve
snapshot uses, and the program in this amendment derives the field correctly.
The amendment-4 program is left unedited because
`paper/results/conditional_concentration_20260914.json` records its source hash
as the code that produced the published result; editing it would make that
record describe code that never ran. Anyone re-running it against a snapshot
validator should take the derivation from this amendment's program.

## Scope of effect

This amendment changes the two figures that render from the training-curve
snapshot, and nothing else. It does not touch the success-conditional curves,
which are built by a separate program from the saved responses on their own
evaluation grid, so those retain their own gaps and their own per-series
cohorts. It computes no contrast, no interval and no aggregate. The fixed seed
cohorts are unchanged: the amendment changes which checkpoints a cohort seed
contributes, never which seeds are in the cohort.

## What had already been observed when this was frozen

Stated so the disclosure is exact. Before freezing, the analyst had seen: the
rendered figures and the location of every gap in them; the block and file
structure of the conflicted cells, including which registered steps each cell
lists as conflicted; the `debug_job` identifiers of the competing blocks and
each cell's registered job identifier; and the counts of conflicts whose
competing origins include the registered job, share a job, or include neither.

The analyst had also seen, for one cell
(`level1/qwen3b/python_factors/replay_maxrl/70` at step 1920), the recorded
metric payload of one competing block, printed as part of a stored
`conflicting_duplicate` issue record while diagnosing the conflict kind. That
cell is in the affected population. Its inclusion follows from the rule and from
source structure alone, and the rule admits whichever block the trajectory
continues from regardless of that payload, but the observation is disclosed
because it occurred.

The analyst had **not** computed, from any newly admitted mid-training block, a
curve, a fixed-cohort mean, a seed range, a paired contrast, a \pmd{} value or
any aggregate derived from them.

## Reporting

Preserve the September 12 snapshot and amendment 4's artifacts unchanged. Write
the admission to a separate snapshot and leave the source in place. State in the
manuscript that the rule was adopted after the figures were first drawn, and
that it extends a rule adopted after the original analysis was specified. Do not
describe the post-amendment curves as the originally analyzed ones, and do not
present the change as a confirmation of anything.

Report what the redrawn figures show under the same rule, including any respect
in which a filled gap weakens, reverses or complicates a reading the manuscript
currently gives. The rule may not be revised, narrowed or abandoned in response
to what it produces. Checkpoints the rule leaves unresolved remain gaps, and the
figures continue to draw them as gaps rather than interpolating across them.

## Constraints carried forward

Amendments 1 to 4, the registered 192-step grid, the four-draw checkpoint
contract, the fixed paired cohorts, the terminal census binding and every
uncertainty convention are retained unchanged. No new model call, training run,
checkpoint selection, grading pass or outcome definition is permitted. Newly
admitted blocks are ingested through the existing frozen-prefix hashing and the
unchanged raw response normalizer, and the patched snapshot is validated by the
same loader that validates its source.
