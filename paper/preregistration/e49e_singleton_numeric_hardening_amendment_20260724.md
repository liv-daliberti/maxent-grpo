# E49E singleton-repair numeric-leakage hardening

**Status: FROZEN BEFORE ANY SINGLETON-REPAIR JUDGE REQUEST OR POLICY TRAINING — 2026-07-24**

The singleton repair sees the reference answer and, when available, the gold
derivation only as auditor-side inputs. Before its first request, the local
leakage gate is strengthened to reject any policy-visible operation or plan
containing a numeric literal that is absent from the original problem, except
for the structural constants `0`, `1`, and `2`. Literal reference answers,
boxed answers, and explicit answer phrases remain forbidden.

This hardening prevents gold-derived intermediate values from becoming hints
in the finite policy menu. It does not change the two fixed proposal slots,
the two execution-audit seeds, the answer verifier, or the rule that a
singleton repair cannot count as new multi-route support.
