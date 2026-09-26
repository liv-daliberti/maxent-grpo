# E49E curated V4 two-row symbolic correction

**Status: FROZEN BEFORE ANY V4 AUDIT REQUEST — 2026-07-24**

V3 certified nine of the 11 repaired toy rows and correctly rejected two
fixed contracts whose Qwen2.5-72B executions exposed missing symbolic detail.
V4 carries all nine validated V3 records byte-for-byte and replaces contracts
only for those two rejected rows:

- the complex-polygon contract now fixes the circumradius as
  `R=|1+i|^(1/4)` and the square area as `A=2R^2=2|1+i|^(1/2)`;
- the integer-sequence contract linearizes
  `r_(k+1)=2-1/r_k` with `s_k=(r_k-1)^(-1)`, states the resulting quadratic
  term parameterization, and requires the integer-difference argument before
  specializing at `a_13=2016`.

Neither contract states the reference answer or an evaluated derived numeric
result.  Both must pass the unchanged digit and number-word leakage gates and
the frozen objective dual-audit rule: both independent derived answers must
match exactly, and at least one execution must affirm every action, declared
action closure, and self-containment.

No V3 failure is reinterpreted.  V4 uses new menu hashes and fresh audit cache
keys, records the V3 carry provenance, and remains singleton-only coverage
with no diversity or entropy contribution.
