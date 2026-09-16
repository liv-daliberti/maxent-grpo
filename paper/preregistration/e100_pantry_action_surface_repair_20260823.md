# E100 Pantry action-surface repair

Date: 2026-08-23 EDT. This amendment is frozen before cancelling or replacing
any E100 Pantry science job.

## Trigger and diagnosis

The five unfinished Falcon-1B PantryPlan sparse-RLEP-Dr cells all reached the
first learner update and raised the same deterministic exception: the offline
pool stores verified Pantry allocation witnesses, while the registered policy
acts on six binary support-mask tokens. The frozen E100 learner tokenized the
allocation text directly and then correctly rejected those text tokens as
outside the canonical action support. Seed 57 exhausted its retries and is
terminal failed; seeds 55, 56, 58, and 59 have the same exception in their
logs and are only pending, requeued, or awaiting watchdog termination.

This is an action-representation integration defect, not a failed verifier,
empty replay pool, numerical failure, model failure, or scientific outcome.
No unfinished Pantry cell has a checkpoint or completion receipt.

## Authorized repair

Create a new immutable snapshot by copying the frozen E100 snapshot and
replacing exactly the following three source files with the already-audited
E98-R1 Pantry action-surface repair, whose pre-repair files are byte-identical
to E100's:

- `src/oat_drgrpo/pantry_support_action.py`;
- `src/oat_drgrpo/canonical_actions.py`; and
- `src/oat_drgrpo/learner/grpo.py`.

The adapter first revalidates each stored allocation, maps its selected
ingredients to the public six-row inclusion mask, and maps each mask bit to
the registered one-token action at that position. It changes serialization
only: no replay row is added, removed, deduplicated, or reweighted. The frozen
pool, prompt matching, deterministic sampling, replay count, mixed advantage,
Dr.GRPO normalization, action support, reward, model, seed, optimizer,
training horizon, evaluation cadence, checkpoint cadence, and estimand remain
unchanged.

Validate the adapter with its fail-closed unit tests and by projecting every
validator-positive witness in all five frozen E100 Pantry pools. Then run a
fresh 32-step Falcon Pantry sparse-RLEP smoke from seed 57. A CPU audit must
observe the registered terminal step, replay use, finite metrics, and receipt
before any replacement science job becomes eligible.

## Replacement and preservation

Submit one held replacement for each Pantry seed 55--59 from that cell's
frozen `SubmitLine`, changing only the job name, source/ops snapshot roots,
dependency, and the already-authorized A6000 placement pool in partition
`all`. Preserve every run directory; each replacement writes a new
`debug_job<id>` subtree and finds no resumable checkpoint. Record every old
job ID, terminal/live state, failing-log digest, new held scheduler record,
and three-file patch digest before cancellation.

Only after the repair ledger and amended primary ledger are durable may the
four nonterminal doomed jobs be cancelled. Release the smoke, audit, and five
replacement jobs together; replacements remain `afterok`-gated on the smoke
audit. The admissible E100 cohort remains 22 executable cells plus the three
previously registered zero-eligibility blocked cells.
