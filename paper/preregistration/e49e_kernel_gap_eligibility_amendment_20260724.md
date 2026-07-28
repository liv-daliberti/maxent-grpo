# E49E finite-kernel gap-eligibility clarification

**Status: FROZEN BEFORE ANY FINITE-KERNEL AUGMENTATION REQUEST OR POLICY TRAINING — 2026-07-24**

The finite-kernel amendment contains an internal wording ambiguity: it says
to keep the better independently certified support while also saying that a
row with no surviving original route remains a gap.

The executable rule is fixed here. For every row, the deterministic selector
compares:

1. the immutable original E49E trace-certified bank, when nonempty; and
2. the answer-bound finite-kernel bank after both execution audits and both
   equivalence attacks.

It keeps the bank with more retained strategies and prefers the original on a
tie. Therefore, a finite-kernel bank may fill an original zero-support row,
but only with the support that independently survives the same strict
soundness and equivalence contract. A row remains a gap only when neither
source retains a valid route; only those rows enter the separately frozen
singleton repair.

This clarification changes no judge seed, prompt, acceptance rule, support
threshold, E46 objective, or matched Dr.GRPO control.
