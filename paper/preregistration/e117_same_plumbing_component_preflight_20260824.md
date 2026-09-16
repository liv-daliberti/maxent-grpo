# E117 same-plumbing component preflight

Frozen: 2026-08-24, before E117 submission or outcome inspection.

Status: mechanism-only development preflight. E117 is not a confirmatory
efficacy experiment and does not modify, stop, select, or reinterpret any live
E112-R1 cell. PointMaze is excluded.

## Question

Can the verified-support treatment be reduced to two identifiable increments
on one execution surface?

1. `P - C`: admit independently verified proposal outcomes to uniform replay.
2. `F - P`: add v7 verified-support semantic PPO after proposal support is held
   fixed.

The preflight tests whether those contrasts are implemented and recoverable. It
does not use endpoint values to decide whether the method works.

## Frozen cells

Use training seed 117 and 64 distinct training prompts for one pass. Run three
arms on each of four sentinels, for 12 cells total:

| Scale | Domain | Reason | Frozen node |
|---|---|---|---|
| Qwen-0.5B | Countdown | repeated positive sentinel | node021 |
| Qwen-0.5B | Graph coloring | active-mechanism/null-outcome boundary | node021 |
| Qwen-0.5B | Python factors | cold-start/phase-transition sentinel | node022 |
| Falcon-1B | MathIR | low-realized-dose second-family sentinel | node022 |

Every C/P/F block is therefore pinned to one model, dataset, training seed,
node and GPU class. Scheduler availability may serialize cells; start time is
not an estimand.

## Exact arms

All arms use the same frozen source and the
`verified_replay_semantic_maxent_verified_support_discovery` runtime variant.
They share replicated free-form sampling, local actor synchronization, group
size 16, ReplayDr weight 0.10, one proposal group per update, one proposal
attempt, proposal temperature 1.20, isolated deterministic proposal request
seeds, validator, canonicalizer, token budget, optimizer, checkpoints, and
evaluation requests. Proposal rows never enter PPO, task-reward counts, or
evaluation.

- **C — compute-only admission control:** semantic coefficient 0.0. Generate,
  validate, and canonicalize the fixed proposal group, then discard the
  resulting candidate payload immediately before any bank, route-library,
  support, retention, replay, or policy-gradient mutation.
- **P — proposal replay:** identical to C, except validator-positive novel
  candidates may cross that admission boundary into uniform replay. Semantic
  coefficient remains 0.0.
- **F — full verified support:** identical to P, with fixed v7 semantic
  coefficient 0.10.

The only allowed environment differences within a sentinel are run identity,
the C admission-discard flag, and the F semantic coefficient. The launcher must
fail closed if any other field differs.

## Schedule and measurements

- Training: 64 rows x one pass = 64 optimizer updates.
- Checkpoint/resume cadence: 32 updates, with two resume slots retained.
- Evaluation: step 0 and terminal only, one common K=8 coverage draw. These
  values exist only to test exact arm identity and output completeness. They
  are not an efficacy gate and must not be used for component selection.
- Telemetry: neutral request identity, task rewards, replay state, fixed-control
  request and token accounting, proposal validation/canonicalization,
  admission/discard counts, proposal-to-PPO leakage, semantic eligibility and
  realized pressure, retention state, and checkpoint/resume state.

## Pass/fail audit

E117 passes only if all of the following hold within every sentinel block:

1. source snapshot, model/data/optimizer/sampler configuration, node, step-zero
   evaluation rows, neutral prompt order, and pre-intervention task rewards are
   exact across C/P/F;
2. fixed-control groups, request seeds, row counts, and charged token budgets
   are exact across C/P/F;
3. C validates/canonicalizes candidates but has zero admissions and zero
   mutation of proposal-derived bank, route, support, replay, or retention
   state;
4. C and P have exact-zero semantic coefficient and realized semantic pressure;
   F has coefficient 0.10 and finite, bounded, nonzero pressure whenever an
   eligible multi-support group occurs;
5. all arms have zero proposal/control rows sent to PPO, zero task-reward count
   leakage, and zero evaluation leakage;
6. P/F admission is possible when a validator-positive novel candidate is
   produced; lack of such a candidate in one short cell is descriptive, not an
   automatic plumbing failure; and
7. if any job resumes, the first post-resume neutral request stream and all
   bank/proposal/retention counters agree with the saved state. If no job
   resumes, resume fidelity remains covered by unit contracts and is marked
   unexercised rather than inferred.

Any failed identity or leakage condition blocks the causal screen. A failure
may motivate a repaired preflight, but never an efficacy interpretation of
these 12 cells.

## Statistical boundary and next experiment

E117 has one training seed per sentinel and one evaluation draw. Endpoint means,
confidence intervals, significance tests, rankings, and cross-domain pooling
are prohibited.

If the mechanism audit passes, freeze a separate Stage-1 protocol before
submission. It will use three new paired training seeds, a new development
evaluation seed block with at least 16 common-random-number K=8 draws, all eight
passes, and the same C/P/F matrix. The primitive endpoint vector is
`(pass@8, raw distinct@8)`; adjusted breadth `distinct@8 - pass@8` is derived
and retained but is not treated as an independent third coordinate. Analysis
must keep training-seed uncertainty separate from within-checkpoint evaluation
Monte Carlo error, and report terminal plus fixed-grid AUC.

Three seeds remain a development screen, not confirmation. Advancement requires
an effect larger than +0.05 adjusted breadth and two Monte Carlo standard
errors, positive in at least two of three paired seeds and in terminal and AUC,
with no family mean pass loss below -0.03 versus C and no paired pass loss below
-0.10. A broad successor requires an actionable component in at least three of
four sentinels. A Countdown-only effect can advance only as a domain-specific
claim with Graph retained as a registered negative boundary. Confirmation uses
fresh seeds and a previously untouched evaluation block.

## Prior lessons encoded here

- E66: request-stream changes are interventions, so controls use the same
  replicated sampler and local synchronization.
- E102/E103: more admissions or proposal attempts do not imply better outcomes,
  so proposal budget is fixed at one and is not tuned here.
- E108: admission is not neutral-policy conversion, so retention telemetry is
  observed without enabling adaptive priority.
- E89: a nominal coefficient is not a common realized dose and the prior RMS
  controller pinned at safety bounds, so F retains fixed eta=0.10 and no dose
  normalization is introduced before component value is identified.
- The E112-R1 private prefix is scheduler-selected and repeatedly viewed. It
  motivates the sentinels but supplies no E117 efficacy gate.

The desired outcome is a trustworthy decomposition, including a trustworthy
null. If P beats C and F does not beat P later, retire semantic PPO. If F beats
P, complete the missing semantic-without-proposal arm before normalization. If
neither contrast converts despite admissions, work on conservative retention
and replay rather than increasing sampling or entropy pressure.
