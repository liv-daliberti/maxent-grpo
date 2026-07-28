# E49F V5 repaired-bank calibration replay amendment

**Status: FROZEN BEFORE ANY V5 MANUAL LABEL OR POLICY TRAINING — 2026-07-24**

The E49E repaired-bank blinded calibration protocol and thresholds are
unchanged. The completed bank now uses the V5 nested repair evidence chain
rather than the original V1 repair schema, so the calibration replayer must
validate that exact chain before preparing the blinded packet.

For each V5 singleton, replay requires the V5 record, its embedded V4 record,
the embedded V3 record where applicable, every contract manifest, the exact
problem and reference-answer bindings, and the objective dual-audit gate.
For every non-singleton row, the frozen raw trace and finite-kernel
augmentation selection are recomputed as before. The V5 frozen input-tree
hashes replace V1's direct input-file fields; the materialization manifest
still binds each operative record file.

Python interpreter cache files under `__pycache__` are excluded from snapshot
replay hashing. They are generated only after the frozen source hash has
already passed and are not executable source inputs; every non-cache file
must still reproduce the exact frozen tree hash.

This adapter does not add routes, relabel a failed audit, weaken soundness or
distinctness, or alter the blinded audit. All 26 retained toy pairs and the
three hidden equivalent controls must be labeled before their private kinds
are inspected. Only the deterministically pruned, manually audited dataset
may be used for policy training.
