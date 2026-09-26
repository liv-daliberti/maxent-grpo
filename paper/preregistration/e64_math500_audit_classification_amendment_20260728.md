# E64 MATH-500 audit-classification amendment

**Date:** 2026-07-28  
**Scope:** post-run process-audit repair only

## Trigger

The frozen E64 auditor classified the literal string
`Traceback (most recent call last)` as a fatal process failure. The
`math_verify` dependency prints that string for exceptions caught inside
`compare_single_extraction_wrapper`; it assigns the affected comparison a
negative result and continues. All six E64 runs subsequently reached optimizer
step 4608 and materialized every registered evaluation boundary.

The saved response-level verifier-sensitivity audit remains responsible for
checking that repeated identical response/reference tuples do not receive
conflicting rewards.

## Permitted repair

The matched-run auditor may classify a traceback as a caught verifier
diagnostic only when the same log context satisfies all of the following:

1. the traceback is immediately preceded by `Error during comparison`;
2. its stack enters `math_verify/grader.py`; and
3. its stack names `compare_single_extraction_wrapper`.

Such events must be counted and reported per run. Every other traceback remains
fatal. The existing CUDA OOM, non-finite, child-process, actor-death, and
segmentation-fault signatures remain fatal without exception.

## Frozen scientific surface

This amendment does not change training, checkpoints, examples, responses,
rewards, metric computation, seeds, evaluation cadence, matched arms, or the
registered advancement gate. It cannot select or alter a result. The original
identity and logs remain immutable; a separate machine-readable amendment
record binds their hashes to the repaired auditor.

