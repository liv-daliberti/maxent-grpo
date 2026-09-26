# E104 execution note: accidental pre-treatment log exposure

At 2026-08-17 15:17 EDT, while checking that the first E104 job had loaded the
intended estimator and replay configuration, a tail of the combined startup
log also displayed the aggregate correctness field from its step-0 evaluation.
The affected cell was Qwen2.5-0.5B Graph Coloring, seed 43, Slurm job 30637786.

No result from step 1 or later, no sampled breadth result, and no comparison
with a prior endpoint was read.  The displayed value was measured before any
optimizer update and therefore cannot reveal the effect of the repaired
estimator.  No protocol, coefficient, domain, seed, stopping rule, gate
threshold, or full-run setting is changed in response.

This note amends the blinding record, not the E104 mechanism criteria.  The
mechanism auditor remains restricted to training telemetry and runtime-failure
markers.  Its output must report both the pre-treatment exposure and that no
post-update outcome was inspected.  E105 may be released only if E104 passes
the originally frozen mechanism criteria and the audit contains exactly that
limited exposure record.
