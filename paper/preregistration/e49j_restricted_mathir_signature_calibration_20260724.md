# E49J restricted MathIR executable-signature calibration

**Status: FROZEN BEFORE ANY E49J 72B REQUEST — 2026-07-24**

## Motivation

E49I correctly recovered the direct-radix versus decimal conversion pair but
its free-text operator descriptions were internally inconsistent: judges
could call two signatures distinct while simultaneously saying a routine
translation existed, or invent different names for the same inverse-variation
invariant.  A free-text signature is therefore not canonical enough.

E49J makes the 72B judge select from a restricted executable MathIR operator
ontology.  The model performs semantic parsing and soundness checking; a
deterministic local comparator removes generic bookkeeping operators and
compares the remaining canonical operator-ID sequences.  Neither prose labels
nor the model's final relation field determines novelty.

## Canonicalization contract

The frozen enum is defined in
`ops/math_strategy_calibration/calibrate_e49j_mathir_signature_veto.py`.
It contains generic operators plus problem-solving primitives such as:

- `INVERSE_VARIATION_INVARIANT`;
- `INDEPENDENT_CHOICE_PRODUCT`;
- `RADIX_GROUP_MAP` and `RADIX_REPEATED_DIVISION`;
- `GCD_LCM_PRODUCT_THEOREM` and `PRIME_FACTOR_GCD_LCM`;
- `GEOMETRIC_SERIES_DIFFERENTIATION` and `SERIES_SHIFT_SUBTRACTION`;
- `INTERIOR_PRODUCT_COUNT` and `TOTAL_MINUS_BOUNDARY_COUNT`;
- `AM_GM_INEQUALITY` and `CALCULUS_GLOBAL_EXTREMUM`; and
- the finite geometry, recurrence, modular, polynomial, and counting
  primitives needed by the already frozen E49H cohort.

Every assessment returns only enum-valued ordered signatures, route
soundness, signature completeness, and a bounded rationale.  The local
canonicalizer deletes only frozen generic operators:
`READ_GIVENS`, `SUBSTITUTE_VALUES`, `ARITHMETIC`, `ALGEBRA_REARRANGE`,
`SOLVE_EQUATION`, and `VERIFY_RESULT`.  A vote is distinct exactly when both
routes are sound and complete, both remaining signatures are nonempty, and
the remaining ordered signatures differ.

The ontology deliberately maps the following to the same ID: ratio versus
named-constant inverse variation; multiplication principle versus Cartesian
product cardinality; inclusion-exclusion versus the algebraically identical
two-event complement calculation; decimal versus fractional magnitude
comparison; and any redundant unit detour around cubic scaling.

It deliberately keeps different IDs for different executed algorithms:
radix grouping versus decimal-plus-repeated-division; gcd-lcm theorem shortcut
versus prime-factor construction; differentiated generating function versus
series shift/subtraction; and direct interior count versus total-minus-boundary
count.

The parser classifies operations actually present in the execution trace,
not a decorative plan label. Thus a route that claims generating functions
but only evaluates binomial coefficients is `COMBINATION_PRODUCT`; a divisor
listing relabeled as modular arithmetic remains `DIVISOR_GCD_ENUMERATION`;
and expanded versus exponent-combined copies of one factorization remain
`PRIME_EXPONENT_ACCUMULATION`.

## Frozen assessments and decision

Each pair receives four temperature-zero parses:

1. MathIR canonical parser, A then B, seed `493301`;
2. MathIR canonical parser, B then A, seed `493302`;
3. adversarial MathIR parser, A then B, seed `493311`; and
4. adversarial MathIR parser, B then A, seed `493312`.

The pair is predicted distinct only by four-way unanimity of the deterministic
signature comparison.  A malformed, unsound, incomplete, empty, or
disagreeing parse is non-distinct.

Calibration passes only with zero false-new across all manually
non-distinct/unsound pairs, all hidden equivalent controls rejected, at least
three of four manually distinct pairs recovered, and all requests complete.
The unseen E49H cohort remains subject to independent trace execution,
four-way MathIR unanimity, and a sealed zero-false-new manual audit.
