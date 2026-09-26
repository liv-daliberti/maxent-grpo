# E49E finite-kernel diversity-augmentation amendment

**Status: FROZEN BEFORE ANY FINITE-KERNEL AUGMENTATION REQUEST OR POLICY TRAINING — 2026-07-24**

This successor retains the E49E execution gate and the unchanged E46
normalized canonical-bank Haarnoja treatment. It addresses a measured recall
failure in the answer-blind candidate bank: many rows contain multiple sound
proposals that merely repeat the same decisive calculation with different
wording, units, or representation. Such duplicates correctly collapse and do
not provide a usable Math strategy support.

## Finite route and operation language

Each problem receives exactly one answer-bound Qwen2.5-72B bank proposal at
temperature zero and seed `492201`. The structured response contains two or
three routes, no more than twelve actions total, and assigns every route one
distinct kernel from:

```text
direct_algebra, substitution_change_variable, factorization_roots,
inequality_bound, modular_invariant, recurrence_induction,
generating_function, combinatorial_bijection, inclusion_exclusion,
geometric_similarity, coordinate_geometry, complex_plane,
trigonometric_identity, calculus_extremum, symmetry_invariant,
exhaustive_casework
```

Every action uses one operation code from:

```text
READ_GIVENS, NORMALIZE, SUBSTITUTE, EXPAND, FACTOR, SOLVE_EQUATION,
ENUMERATE_CASES, COUNT_OBJECTS, APPLY_MODULAR_RULE, APPLY_THEOREM,
CONSTRUCT_OBJECT, TRANSFORM_REPRESENTATION, ESTABLISH_BOUND,
CHECK_CASES, CONCLUDE
```

The materializer renders these as machine-visible `[KERNEL:...]` and
`[OP:...]` prefixes in the ordinary finite menu. Kernel labels are never
sufficient evidence of novelty. The pair auditors and manual reviewer see the
executed mathematics.

## Answer binding without answer leakage

The proposer sees the problem and reference answer and, for frozen training
rows that contain one, the gold derivation. This information is auditor-only.
Local validation rejects a bank that includes the reference answer, a boxed
answer, an evaluated final value, worked numerical result, preassigned action
ID, duplicate kernel, duplicate combo, more than twelve actions, or an action
outside the finite operation enum. Both matched policy arms see only the
validated finite menu and the original problem.

## Soundness and equivalence

Every proposed route receives the unchanged independent
`literal_action_executor` and `adversarial_action_checker` audits at seeds
`492111` and `492112`, including the frozen MATH answer verifier with
`audited_math_answer_surface_v1`. Only double-sound routes reach pair review.

The two pair attacks at seeds `492121` and `492122` apply a strengthened
equivalence rule. Decimals versus fractions, dollars versus cents, unit
conversion before versus after, reordered equations, identical formulas,
redundant checks, and different labels for the same decisive operation are
explicitly equivalent. A distinct edge requires route-exclusive intermediate
facts produced by genuinely non-routine operations. The deterministic maximum
clique is retained.

For each row, the materializer keeps whichever has greater independently
certified support: the immutable original E49E bank or the new finite-kernel
bank. Ties prefer the original bank. A row with no surviving route remains a
gap for the separately frozen singleton repair; it is never promoted to
multi-route support.

## Frozen gates

Before the first augmentation request:

- the complete raw E49E records and E49D input are immutable;
- all five known-invalid controls and all three known-equivalent controls pass
  under conservative recomputation;
- the source cohorts, endpoint, control manifests, source snapshot, protocol,
  prior amendments, this amendment, preflight, launcher, and Slurm wrapper are
  content-addressed; and
- completed proposal/audit outputs are durable and cannot be resampled.

After augmentation and singleton repair, every retained multi-route pair still
requires a blinded manual soundness/self-containment/distinction decision.
False pairs are removed and support is recomputed. No policy training can
start unless the final artifact has zero accepted false-new pairs, at least
20/100 multi-route toy rows, at least 10/50 multi-route evaluation rows, and
all rendered prompts at or below 2048 tokens.
