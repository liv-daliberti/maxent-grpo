# E105 amendment: bind the full cohort to the E106 Python surface repair

Frozen on 2026-08-17 before E105 submission and before any E104/E106
post-update outcome was inspected.  This amendment supersedes only the E105
release-gate and runtime-snapshot clauses; its 75-cell estimands, seeds,
domains, checkpoints, comparator pairing, and reporting rules are unchanged.

The E104 Qwen2.5-3B Python mechanism cell proved that the old runtime rejected
the model's native boxed `\lambda n:` surface before executable validation.
E106 repairs that formatting boundary and reruns Python at all three scales.
Accordingly, E105 now requires the passing
`e106_python_lambda_normalization_combined_gate_v1` audit: twelve non-Python
E104 cells plus the three superseding E106 Python cells.  The combined audit
must be complete, outcome-blind, PointMaze-free, and backed by the frozen
53-test E106 evidence.

All 75 E105 cells use the content-addressed E106 snapshot
`e106_python_lambda_b853595e3b158046`.  Relative to E104, its ops tree is
byte-identical and its only source change is
`src/oat_drgrpo/math_grader.py`.  That change is reachable only for the Python
factor verifier and normalizes the exact LaTeX spelling to the already-frozen
`lambda n:` language.  Thus non-Python cells remain execution-equivalent to
the original E105 design, while Python cells no longer repeat the known
admission bug.  PointMaze remains excluded.
