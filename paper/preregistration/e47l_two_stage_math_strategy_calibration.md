# E47L-CAL: two-stage derivation validation and strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Reason for the prospective successor

E47K completed all 50 bounded problems with zero false-new events on 150
injected controls and 223/275 (81.09%) answer-matched policy responses
assigned. Its strengthened combined prompt also separated the direct-sign
and expanded-case routes that E47J had falsely merged. However, an integrity
spot check found that E47K still admitted two plainly invalid modular-GCD
derivations as distinct strategies. Their final answers matched, but their
written inferences did not establish those answers. The immutable witness is
recorded in
`var/artifacts/e47k_integrity_math_strategy_calibration_v1/integrity_failure.json`.
E47K is therefore blocked before manual pair scoring or training.

## Prospective two-stage change

E47L separates the two semantic tasks that E47K combined.

1. Two independently permuted, temperature-zero Qwen72 calls classify every
   answer-matched candidate derivation as `valid`, `invalid`, or `ambiguous`.
   The prompt explicitly requires checking the response's actual algebra,
   arithmetic, logical inferences, domain restrictions, and necessary cases.
   It forbids repairing a flawed response. A candidate proceeds only when
   both calls return `valid`.
2. Only the doubly valid candidates and previously admitted valid
   representatives enter the unchanged two-pass strategy partition and E47J
   component-incidence rule.

All responses still pass the ordinary full MATH final-answer validator first.
Thus a new bank entry now requires both final-answer correctness and two
derivation-validity approvals before two independent strategy partitions can
admit it. Any disagreement fails closed with reward zero.

The four calls use the same immutable
`Qwen2.5-72B-Instruct-AWQ@698703eae6604af048a3d2f509995dc302088217`,
permutation seeds `470721` and `470722`, temperature zero, finite strict JSON
schemas, item bounds, and private node105 service. The persistent executable
schema is `math_strategy_canonicalizer_two_stage_integrity_v10`.

The original 50 problems, 3,200 Qwen0.5B samples, frozen answer-validator
outcomes, injected controls, online group-16 schedule, component-incidence
rule, blinded audit construction, and numerical gates are unchanged. In
particular, E47L does not lower the 80% policy-key coverage gate in response
to stricter validation.

## Frozen gate

E47L advances only if:

- exact-duplicate false-new is zero;
- overall injected false-new is at most 5%;
- lexical-paraphrase false-new is at most 10%;
- at least 80% of answer-matched policy responses receive a key after the
  two-stage integrity screen;
- a newly constructed blinded manual strategy audit has same-pair false-new
  at most 5% and different-pair false-merge at most 20%;
- the two known E47K invalid modular witnesses receive no key; and
- every bounded problem completes without structural failure.

The witness check is a regression test, not a substitute for the blinded
pair audit. Passing supports use as a conservative sampled reward signal; it
does not make the 72B screen a formal proof checker.
