# E49S — deterministic MathIR repair of the audited hard-MATH toy bank

**Status: FROZEN BEFORE REPAIR EXECUTION OR TRAINING — 2026-07-24**

## Why this successor exists

E49R's blinded manual audit rejected eight of twenty proposed additions. The
dominant failure was concrete: a 72B audit sometimes accepted text saying
"enumerate", "run dynamic programming", or "use the central-angle route"
without the displayed trace actually performing that operation. E49R remains
immutable and failed its advancement gate at 9 train and 7 evaluation
multi-route rows.

E49S does not ask another language model to reconsider those failures. It
repairs exactly four rows whose two intended algorithms were already labeled
genuinely distinct in the blinded E49R packet, but whose rejected route lacked
a faithful execution. Each repaired route is implemented below as a bounded,
deterministic, exact-arithmetic program. The program must execute the same
ordered action IDs exposed to the policy, produce an exact state trace, and
derive the frozen reference answer. Any program/action/menu mismatch fails
closed.

## Frozen repair cohort

- `train:0031:test/prealgebra/937.json`: LCM multiple count versus literal
  divisibility scan over integers 1 through 500.
- `eval:0027:cf127e68fa63742a3582`: closed-form binomial probability versus
  seven exact Bernoulli dynamic-programming updates.
- `eval:0041:9a01eabaf416b53ac8ba`: circular block contraction versus an
  exhaustive anchored permutation scan of all `7!` arrangements.
- `eval:0047:5310db58fbda4a6559d0`: regular-pentagon exterior-turn proof versus
  central-isosceles decomposition followed by exact line-angle transfer.

No other E49R rejection is restored. In particular, algebraically equivalent
tree/product and tuple/falling-factorial claims remain collapsed.

## Deterministic execution rule

For every strategy, the certifier binds:

1. the exact row ID;
2. the exact ordered action combo;
3. a fixed interpreter function;
4. its full exact output states; and
5. the frozen reference answer.

The certifier runs every function twice in fresh calls and requires identical
canonical JSON, exact reference-answer agreement, and exact agreement between
the function's action-state IDs and the menu combo. The finite scans must
materialize their complete passing set or count from the complete generated
space; the DP must materialize every probability vector. The resulting
certificate and menu hashes are bound into the materialized dataset.

This certifies that each menu option is executable and valid before training.
At policy time, the existing strict menu parser still requires an exact
strategy ID, exact action combo, one nonempty action-step block per action in
order, and a boxed answer. The frozen answer verifier and two 72B
execution-integrity audits remain active in both arms. Thus the repair neither
relaxes task validity nor awards novelty for a menu label alone.

## Advancement rule

The matched toy run is authorized only if:

- all eight repaired programs execute deterministically and match their
  reference answers;
- the four repaired menus reproduce the four pairwise distinctions already
  accepted under blinded E49R labels;
- all 100 rows have at least one retained strategy;
- at least ten train and ten evaluation rows have multiple strategies;
- every rendered prompt is at most 2048 tokens; and
- source records, E49R labels/packet/key, this protocol, the certifier, repair
  certificates, menus, and materialized train/eval trees are SHA-256 bound.

The matched three-epoch run uses the current E46 normalized canonical-bank
Haarnoja arm and a regular execution-gated Dr.GRPO arm with identical data,
seed, model, sampling, learning rate, validator, and evaluation cadence.
