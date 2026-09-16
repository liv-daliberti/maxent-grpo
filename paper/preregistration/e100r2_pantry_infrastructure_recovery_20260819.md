# E100-R2 Pantry infrastructure recovery

Date: 2026-08-19. This execution-only amendment is written after observing
the E100 scheduler outcomes. It does not change the frozen sparse-RLEP
treatment, collection distribution, seeds, prompts, model revision, training
hyperparameters, endpoint, or estimand.

## Observed outcomes

PantryPlan seeds 56, 57, and 59 completed their frozen 384-prompt pool
collections and pool audits. Each receipt reports 228 replay-eligible and 156
ineligible prompts. Their original scientific jobs remained user-held after
the gates passed. Those exact jobs are released; they are not replaced.

PantryPlan seeds 55 and 58 failed during vLLM model initialization because the
allocated A6000 was out of memory. The watchdog terminated each job after one
hour. Neither output root contains a pool artifact, so no draw or outcome is
selected or discarded. Each collection is retried from its original audited
`SubmitLine`, changing only job name, scheduler priority, node eligibility,
and dependencies. The GPU class remains A6000. The retries are serialized to
reduce transient GPU contention. Their fresh pool audits use the frozen E100
auditor and require 384 prompts plus at least one replay-eligible prompt.

PythonFactors seed 58 completed collection, but its frozen audit raised
`sparse RLEP pool has no replay-eligible prompt`. It is therefore the same
scientific gate outcome as PythonFactors seeds 56 and 59: blocked, not a
training failure. Its unstarted cancelled training job remains cancelled and
is moved to `blocked_runs`. The pool is not regenerated and no
control-equivalent zero-replay treatment is run under the E100 label.

## Gates and release

A fresh CPU job re-audits the immutable completed E100 smoke receipt. The two
replacement scientific jobs depend on both their own successful pool audit
and this smoke re-audit. All new jobs are submitted held, their scheduler
records are checked against the frozen command surface, the primary and repair
ledgers are written atomically, and only then are the jobs released.

The admissible E100 outcome is therefore 22 executable Falcon sparse-RLEP
cells and three explicitly blocked zero-eligibility PythonFactors cells.
