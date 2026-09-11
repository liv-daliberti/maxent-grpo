# E50G runtime-control schema-name repair

**Status: RECORDED 2026-07-26 AFTER THE FIRST E50G PROCESS EXITED DURING
ACTIVATION, BEFORE ANY E50G CANDIDATE, MENU, 0.5B SAMPLE, RESULT, OR
TRAINING OUTPUT EXISTED.**

The first E50G allocation exited before reading or evaluating the candidate
corpus.  Its activation check expected the two already frozen, passing E49T
runtime-control artifacts to declare these nonexistent schema names:

- `e49t_route_confusion_calibration_v1`
- `e49t_declaration_mismatch_calibration_v1`

The immutable artifacts actually and historically declare:

- `e49t_route_confusion_calibration_result_v1`
- `e49t_declaration_mismatch_result_v1`

This amendment changes only those two literal schema-name expectations to
the names present in the hashed artifacts.  It does not alter either
artifact, any candidate, signature, exact-answer rule, soundness/binding
audit, forced or unforced sampling count, natural-support threshold, ranking
rule, selection threshold, downstream toy, or training gate.

The repair file and repaired E50G script hashes must be recorded in the
fresh E50G identity.  The failed activation-only allocation remains
preserved in its scheduler and stderr records.
