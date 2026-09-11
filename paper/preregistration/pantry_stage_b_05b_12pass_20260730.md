# PantryPlan Stage B: clean 0.5B verified MaxEnt versus compute-matched Dr.GRPO

**Status: FROZEN AFTER THE V3 QUALIFICATION PASS AND BEFORE STAGE-B SUBMISSION — 2026-07-30**

## Authorization and scope

The development-only Pantry integration pair at seed 76201 passed its
independent audit. Both arms completed exactly 96 updates with matched replay
traversal. The compute control exposed nonzero raw replay telemetry with an
exact-zero applied derivative; treatment applied the frozen novelty and replay
terms. The immutable receipt is
`var/artifacts/pantry_support_mask_paired_integration_v3_audit.json`, whose
decision is `eligible_for_ten_stage_b_jobs`.

This protocol prospectively authorizes exactly ten fresh PantryPlan paper
jobs. It does not include seed 76201, resume a qualification checkpoint, alter
the frozen data, or relabel any earlier failed probe.

## Frozen Cartesian product

- Arms: `grpo_compute_matched` and
  `verified_first_global_replay_canonical`.
- Seeds: 43, 44, 45, 46, and 47.
- Frozen initial model: Qwen2.5-0.5B-Instruct.
- Training pool: all 32 task-bound PantryPlan v2 training prompts.
- Schedule: 12 complete prompt passes, exactly 384 optimizer updates.
- Rollouts: 16 per prompt; `max_queries=(384-1)*16=6128` under the pinned
  runtime's inclusive learning-round convention.
- No resume, seed replacement, best-checkpoint selection, or result-dependent
  extension.

## Matched objectives and compute

The control uses plain Dr.GRPO task reward while traversing the same passive
canonical bank, one global replay group, forward scoring, and backward graph
as treatment. `online_canonical_replay_compute_only=1` makes every applied
replay derivative exactly zero. Semantic, novelty, and replay objective
coefficients are zero in the control.

Treatment uses the qualified E58-style online verified MaxEnt recipe:
semantic Shannon coefficient 0.10, online canonical novelty beta 0.50, replay
alpha 0.10, capacity 16, one persistent-hash global replay group per update,
and 64-step open-set/replay/mass warmups. There is no gold support, target
mode count, target entropy, coefficient projection, token-entropy objective,
E68 proposal actuator, or E69 successor replay.

All other optimizer, sampling, prompt order, request, verifier, execution, and
checkpoint settings are matched within seed.

## Frozen evaluation and reporting

Evaluate the fixed 128-prompt `multi_answer` split at pass 0 and every eight
updates (one quarter pass). Each evaluation contains greedy decoding plus four
deterministic temperature-one draws at K=8. The registered reporting anchors
are passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12. Report terminal pass 12 and
trapezoidal AUC over those anchors for greedy success, mean@8, pass@8, and
distinct@8. Missing seeds are never carried forward or averaged away.

## Fail-closed audit

The independent terminal audit requires the exact ten-cell manifest, exact
384-round streams, all 49 evaluation coordinates, four K=8 draws per
coordinate, finite metrics, matched per-seed replay traversal, task-bound
support identities, zero actor/verifier disagreement, exact-zero control
derivatives, nonzero raw control replay telemetry, and applied treatment
exploration/replay telemetry. It also verifies the qualification, protocol,
source, operations, data, model, manifest, submission, and scheduler
identities. Any missing or failed cell stops Pantry Stage B from becoming a
paper result.
