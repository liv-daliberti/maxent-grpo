# E49E finite-kernel hardening amendment

**Status: FROZEN BEFORE ANY FINITE-KERNEL AUGMENTATION REQUEST OR POLICY TRAINING — 2026-07-24**

This amendment resolves two local-validation details before the first
finite-kernel request.

First, distinct kernel labels are not enough to create distinct finite
actions. Two proposed routes with the same ordered operation-code combo are
rejected locally even when their kernel labels or prose differ. The two
independent execution audits and two equivalence attacks remain mandatory for
every locally admissible route and pair.

Second, the answer-leakage check is strengthened beyond literal reference
matching. Operations and plans may contain numeric literals only when those
literals already occur in the problem or are the structural constants
`0`, `1`, and `2`. The literal reference answer remains forbidden. This
prevents an auditor-only gold derivation from exposing a derived intermediate
or evaluated result through the policy-visible menu.

The source snapshot, this amendment, launcher, Slurm wrapper, frozen raw
records, endpoint, source cohorts, and control manifests are content-addressed
before the first request.
