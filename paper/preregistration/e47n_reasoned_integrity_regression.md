# E47N-REG: reasoned integrity-classifier regression

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47M's live preflight fixed the known routine-equivalence false split but
failed before its 50-problem replay. In a three-item integrity context, it
admitted one invalid modular-GCD derivation; in a two-item context, it rejected
a manually written and answer-validated case proof whose arithmetic and case
restrictions are correct. E47M is preserved as a preflight failure.

E47N changes only the integrity response contract. For every candidate the
judge must emit a short `brief_check` identifying either a decisive
verification or the first material error before assigning
`valid`, `invalid`, or `ambiguous`. The strict finite JSON schema requires a
nonempty check for every ID. The prompt explicitly states that a longer case
proof remains valid when its calculations and case restrictions are correct,
even if a shorter proof exists. Downstream code consumes only the status; the
check is retained for audit.

The strategy-equivalence prompt and all E47M logic are unchanged. The
persistent schema is `math_strategy_canonicalizer_reasoned_integrity_v12`.

E47N advances to a full calibration successor only if the frozen E47M live
packet passes all four checks: valid anchor admitted, both invalid derivations
rejected, routine-equivalent clock solutions merged, and valid direct-sign
versus case proofs separated.
