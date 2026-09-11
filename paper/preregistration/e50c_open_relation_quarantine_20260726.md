# E50C open-relation quarantine

**Status: ACTION SPECIFIED BEFORE TERMINATION — 2026-07-26**

E50C was preregistered and launched before the later E50F2/E50F4
hard-MATH audits completed.  Those audits subsequently observed false-new
errors in the same open-ended 72B relation boundary.  E50C therefore cannot
authorize training regardless of its eventual internal natural-support
counts.

Continuing E50C beyond teacher sampling would spend substantial 72B and 0.5B
compute on a quarantined route partition.  Its unconstrained teacher samples
are not inputs to the replacement E50G path, which uses a separately seeded,
answer-blind, finite-pair conditioned corpus.  Terminate job 30124473 and
materialize a fail-closed terminal record before releasing its dependent
E50F job.

The terminal record must:

- have the original E50C schema but `pass=false`;
- report every fully logged teacher problem and its fixed sample count;
- state whether a private response corpus had been persisted;
- retain hashes for the E50C protocol/script/log, endpoint, cohort, and both
  false-new failure artifacts;
- select no source indices and authorize no training; and
- be written atomically before cancellation, so the dependency successor
  cannot observe a missing or partially written result.

No E50C policy update occurred; E50C was a calibration job only.
