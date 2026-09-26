# E49E answer-blind symbolic singleton-repair V2

**Status: FROZEN BEFORE ANY V2 SINGLETON PROPOSAL OR AUDIT REQUEST — 2026-07-24**

The first answer-bound singleton pass correctly failed closed when its
policy-visible contracts contained derived numeric literals.  This
predeclared contingency retries only rows that still have zero certified
routes after the frozen raw trace bank, route-wise finite-kernel augmentation,
and terminal V1 singleton pass.

## Fixed V2 procedure

V2 removes the reference answer and gold derivation from the proposal request.
The Qwen2.5-72B proposer receives only the original problem and one fixed role
instruction.  The bounded slots, in order, are:

1. `symbolic_direct_route`, seed `492241`;
2. `symbolic_checked_route`, seed `492242`;
3. `natural_representation_route`, seed `492243`;
4. `constraint_first_route`, seed `492244`.

Each response has exactly one strategy and two to five ordered actions.
Operations are capped at 220 characters, the plan at 420 characters, and
generation at 1024 tokens.  Local parsing requires exact ordered use, action
closure, no evaluated result, and no numeric literal absent from the problem
except structural `0`, `1`, and `2`.  Number words and ordinals are mapped
back to their numeric values and receive the same check, so spelling a hidden
value cannot bypass the gate.  A spelled-out exact integer reference answer
is rejected even when that integer is otherwise structural or appears in the
problem.

The prompt requires the decisive symbolic identity, recurrence, invariant,
or event definition whenever it can be expressed within that leakage rule.
This avoids admitting a vague "solve and check" scaffold while still
forbidding worked numeric results.

The first locally admissible contract must pass both unchanged frozen E49E
soundness auditors and the independent exact-answer validator.  Completed
outputs are immutable within V2.  A successful V1 repair is validated from
its completed record and carried byte-for-byte; it is never resampled.
Up to four gap rows may be processed concurrently.  Concurrency changes only
wall-clock scheduling: every row retains its fixed role order, seeds, cache
keys, gates, and first-double-sound selection rule.

As before, repaired rows are coverage-only singletons.  They cannot count as
new strategy support, a novelty outcome, a pairwise distinction, or an
entropy observation.  Training remains blocked until every row has support
and the separately frozen blinded manual audit has pruned all nominal
multi-route pairs.
