# E85: PantryPlan semantic-MaxEnt repair and re-run

**Frozen before submission on 2026-08-09.**

## The defect

PantryPlan is a canonical-action task: rollouts are eight-token action
sequences, not free-form responses carrying a boxed answer. The semantic MaxEnt
term derived its outcome key by decoding response tokens and running the
free-form text extractor over them, which returns `None` for every such row.
Every PantryPlan row was therefore scored unparseable and received exact zero
semantic advantage, for the whole of training, in every affected cell.

The failure was silent because nothing else depended on that key path. The
verifier scored PantryPlan normally and the replay bank filled normally:

| quantity | PantryPlan | Countdown (for contrast) |
|---|---|---|
| reward-positive fraction | 0.544 | 0.561 |
| parseable fraction | **0.000** | 0.971 |
| eligible fraction | **0.000** | 0.561 |
| mean \|A_sem\| RMS | **0.0000** | 0.0069 |
| mean bank size | 7.1 modes/prompt | 1.8 modes/prompt |

Confirmed in 15 of 15 affected cells: E81 PantryPlan (5 seeds), E82 PantryPlan
(5 seeds), E83 PantryPlan (5 seeds).

The sibling outcome-collision path already bound the task's own
canonicalization surfaces at the same call site. The semantic path omitted that
branch. Three further key-derivation sites (DIAYN, mode-adaptive xDr, SEED)
carry the same latent gap; no arm in this campaign enables them, and they are
left unchanged rather than altering the runtime of treatments nobody is running.

## What this invalidates

The PantryPlan column of E81, E82, and E83 is **null by construction**. Those
arms' applied objectives were identical to their comparators, so:

- E81 PantryPlan `distinct@8` $+0.076$ is not a semantic-MaxEnt effect. It is
  run-to-run variance between two runs with the same objective.
- E83 PantryPlan `distinct@8` $+0.000$ is likewise not evidence of a null
  effect; it is the expected result of comparing an arm to itself.
- E82 PantryPlan carries the same defect.

The original 15 cells are retained as audit-only evidence of the defect. They
are excluded from every reported result and are not deleted.

Every non-PantryPlan cell of E81, E82, and E83 is unaffected: those domains are
free-form, their parseable fractions are 0.97, 0.79, 0.42, and 0.39, and their
semantic advantage was live throughout.

## Design

E85 re-runs exactly the 15 affected cells, changing one thing: the semantic term
binds the task's own canonicalization surfaces when canonical actions are
active, the same way the outcome-collision path already did.

- Cohorts, models, domains, seeds, schedule, decoding, replay dose, semantic
  coefficient, placement, and comparators are inherited unchanged from the
  parent cohorts. `eta = 0.10`, replay weight 0.10, eight passes, 3,072
  optimizer updates, evaluations and resumable checkpoints every 192 updates.
- Seeds 43--47 for the two Qwen2.5-0.5B cohorts, 55--59 for Falcon3-1B.
- 3 parents x 1 domain x 5 seeds = 15 runs.

E85 is a defect repair, not a new treatment. It does not change the coefficient,
the estimand, or the registered interpretation of any cohort.

## Runtime

Each parent's cells run from that parent's own base snapshot with three files
replaced: `src/oat_drgrpo/args.py` and `ops/run_experiment.sh` as before, plus
`src/oat_drgrpo/learner/grpo.py` for the key derivation. The new branch sits
inside `if semantic_shannon_tracker is not None:` and is therefore unreachable
for every arm that does not enable semantic MaxEnt — which is every comparator
these cells are read against, so E78, E79, and their control and replay arms
remain code-identical on every path they execute. The launcher walks both trees
and fails closed unless the divergence is exactly that patch set.

## Outcomes and estimands

Unchanged from the parent protocols, restricted to PantryPlan:

- E85/e81 against E78 `replay`: the effect of semantic MaxEnt given replay.
- E85/e82 against E79 `replay`: the same, in the second model family.
- E85/e83 against E78 `control`: the effect of semantic MaxEnt without replay.

Primary metric is the paired seed difference at pass 8 for `distinct@8` and
`pass@8`, all five seeds shown, no pooling across domains or families, no
checkpoint selection.

## Acceptance gate

Before any E85 result is reported, every cell must show a **parseable fraction
above 0.5 and an eligible fraction above 0.1**, and at least one update with a
finite nonzero applied semantic advantage. A cell that reproduces the original
zero-eligibility signature has not been repaired and fails closed. This gate
reads mechanism telemetry only and never evaluation behaviour.

## Integrity

All 15 cells are submitted held from hash-bound runtime snapshots and released
only after their scheduler environments pass an exact audit. The superseded run
of each cell is recorded in the E85 ledger by run stamp, directory, and job id,
so the replacement is traceable rather than silent. No failed scientific run is
deleted or excluded without that record.
