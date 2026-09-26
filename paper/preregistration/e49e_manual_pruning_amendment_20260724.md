# E49E blinded manual-pruning amendment

**Status: FROZEN BEFORE MANUAL LABELS OR POLICY TRAINING — 2026-07-24**

The repaired-bank blinded audit remains a fail-closed gate, but a false
distinctness claim does not force a sound unrelated route out of the
calibration. After every retained pair and all three hidden equivalent
controls have been labeled, the finalizer deterministically constructs the
policy-visible bank as follows:

1. Any route judged unsound or not self-contained in any displayed pair is
   ineligible.
2. An edge exists only when the blinded manual label says both routes are
   sound and their decisive mathematics is genuinely distinct.
3. The retained support is the largest clique in that manual distinctness
   graph, with the frozen strategy order as the deterministic tie-break.
4. Actions and all explicit action references are closed and renumbered after
   pruning. The pruned prompts and records are materialized as a new,
   content-addressed dataset; the unpruned repaired data remain immutable.

The advancement thresholds apply to the pruned bank: exact coverage of all
100 rows, no zero-support row, at least 20 multi-route rows overall, at least
10 multi-route evaluation rows, and no retained pair that lacks a positive
manual distinctness edge. All previous automatic invalid/equivalent controls,
the three blinded equivalent controls, and the 2048-token prompt limit remain
mandatory.

The decision report records both the number of rejected automatic
distinctness claims and the support remaining after pruning. It may advance
only the newly materialized audited dataset, never the pre-audit repaired
dataset.
