# E49C menu-format retry amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49C TRAINING LAUNCH**

The E49C scientific protocol, matched arms, reward contract, menu soundness
criteria, audit seeds, training hyperparameters, and advancement gates are
unchanged.

During frozen offline menu generation, the first structural failure occurred
for toy row `train:0004:test/number_theory/574.json`: Qwen2.5-72B returned
duplicate action-ID sequences on all four attempts. No menu was admitted for
that row; its durable record has `pass=false`. The retry prompt previously
contained only the parser message “strategy action combos must be distinct,”
which did not expose which sequences collided.

Before any E49C training submission, the materializer was amended only to
include the returned `Sj=Aa>Ab>...` sequences in retry feedback and explicitly
request route-specific actions. It does not repair, deduplicate, accept, or
change a menu. The same parser still requires distinct combos, and the same
two temperature-zero audits must still unanimously certify every strategy as
sound and every pair as substantively distinct. Previously accepted records
remain byte-for-byte unchanged; only failed rows are retried.

This is a fail-closed structured-output elicitation repair, not a change to
the treatment or its canonical support.
