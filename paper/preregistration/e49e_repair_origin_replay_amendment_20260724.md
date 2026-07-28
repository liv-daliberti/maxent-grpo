# E49E singleton-repair origin replay amendment

**Status: FROZEN AFTER FAILED PREFLIGHT AND BEFORE ANY SINGLETON-REPAIR JUDGE REQUEST OR POLICY TRAINING — 2026-07-24**

The first singleton-repair preflight made zero judge requests and submitted no
Slurm job. It failed because the replay function applied the answer-bound
finite-kernel numeric-leak rule to rows whose augmentation record explicitly
selected the original, answer-unaware E49E trace bank. That rule examines the
whole answer-bound candidate proposal and is not the identity contract for an
original-bank selection.

Replay is now origin-specific and remains fail closed:

- `original_trace_bank` records are recomputed solely through the frozen raw
  E49E trace-certification path;
- the recomputed menu hash, canonical menu, and full certification hash must
  exactly match the augmentation record;
- `finite_kernel_augmentation` and
  `prior_v1_finite_kernel_augmentation` records continue to replay their
  answer-bound candidate and must pass the strict derived-numeral leakage
  check before their audits are recomputed; and
- no failed or zero-support record is promoted by this change.

The failed preflight directory is archived intact. A new repair source
snapshot, launcher hash, amendment hash, v2 augmentation hash, and zero
repair-request count are frozen before retry.
