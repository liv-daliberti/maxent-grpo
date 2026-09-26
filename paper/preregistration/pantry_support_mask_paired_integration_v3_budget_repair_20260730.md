# PantryPlan task-bound MaxEnt integration v3: query-budget repair

**Status: FROZEN AFTER V2 JOBS 30201151/30201152 AND BEFORE V3 JOBS — 2026-07-30**

## Immutable v2 outcome

V2 remains failed and is not relabeled. Jobs 30201151 and 30201152 both
terminated successfully, but the launcher encoded `max_queries=1488`, equal
to only 93 groups of 16. The runtime therefore materialized learning rounds
1--94 plus its terminal duplicate instead of the protocol's explicit target
of exactly 96 optimizer updates. The corrected diagnostic audit retains only
these two schedule errors. All other checks pass: both arms have positive
verified reward and multimode online support; treatment has nonzero novelty
and replay derivatives; control exposes nonzero raw replay telemetry with
exact-zero applied derivatives; actor/verifier disagreement is zero; and
replay traversal matches.

## Prospective repair boundary

V3 is a fresh development-only pair from the same initial model and does not
resume or reuse v2 model state. It changes only the maximum-query arithmetic
from 1,488 to `1520 = (96 - 1) * 16`, the value used by the existing runtime
to permit exactly 96 learning rounds. It retains the v2 seed 76201, model,
Pantry v2 data and order, three prompt passes, six-bit action interface,
16 rollouts, optimizer, 96-update target, warmups, coefficients, replay
capacity/scheduler, task-bound canonical-surface bridge, checkpoint target,
information boundary, arm definitions, evaluation, and every pass threshold.
This is an execution repair to the v2 protocol's internally inconsistent
query ceiling and 96-update target, not a post-outcome threshold, prompt,
model, seed, data, or objective change.

## Pass boundary

The independent audit must observe exactly 96 learning rounds per arm and all
v2 checks. In particular, both arms need positive verified reward and online
two-mode support; treatment needs a nonzero verified exploration advantage
and applied replay derivative; control needs corresponding nonzero raw replay
telemetry and exact-zero applied semantic, novelty, mass, and balance
derivatives; actor/verifier disagreements must be zero; all finite/support/
no-gold/no-projection identities must pass; and fixed replay traversals must
match exactly. A pass authorizes only the ten fresh Pantry Stage-B jobs at
seeds 43--47. V3 is not a paper seed and cannot be included in final results.
