# E49E repaired-bank blinded calibration amendment

**Status: FROZEN BEFORE MANUAL LABELS OR POLICY TRAINING — 2026-07-24**

The final 50-train/50-evaluation toy bank must pass an identity-bound blinded
audit before either matched policy arm can launch.

The packet contains every pair retained as distinct by the deterministic
maximum-clique selector. It also contains the three frozen equivalent-route
controls, rendered in the same schema and without a visible control label.
Route order is independently and deterministically blinded with seed
`492231`. The private key records pair provenance and expected relation but is
not consulted while labels are produced.

For every pair, the auditor records:

- whether route A is sound and self-contained;
- whether route B is sound and self-contained;
- whether the routes use genuinely distinct decisive mathematics; and
- a nonempty rationale.

Advancement requires all of the following:

- exact coverage of all 100 toy rows;
- at least 20 multi-route rows overall and at least 10 among the 50 evaluation
  rows;
- all five known-invalid controls rejected by soundness;
- all three automated known-equivalent controls rejected as new;
- zero retained pairs that manual review finds unsound, dependent, or
  equivalent;
- all three blinded equivalent controls manually recognized as sound but not
  distinct; and
- every rendered 0.5B prompt at or below 2048 tokens.

The finalizer reloads the frozen repair source snapshot and deterministically
replays original, finite-kernel, and singleton-repair evidence before reading
labels. A stored pass flag, mutable checkout, or packet without a matching
private-key and artifact hash cannot advance to training.
