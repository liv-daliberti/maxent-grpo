# E49E curated V5 odd-index correction

**Status: FROZEN BEFORE ANY V5 AUDIT REQUEST — 2026-07-24**

V4 certified the corrected complex-polygon contract and rejected the
integer-sequence contract because one executor substituted odd index `13`
into the even-term product formula.  V5 carries all ten validated V4 records
and replaces only that rejected contract.

The V5 action explicitly requires solving `13=2k-1`, substituting only into
the odd formula `2016=C(m+k-1)^2`, setting `d=Cm`, and comparing prime
exponents in `(d+C(k-1))^2=2016C` under `d>0` before evaluating `a_1=d^2/C`.
It does not state the reference answer or any evaluated derived numeric
result and passes the unchanged numeric/number-word leakage gate.

Both independent executions must derive an exact reference match, and at
least one must affirm all actions, declared-action closure, and
self-containment.  The V4 rejection is not reinterpreted; V5 has a new menu
hash and fresh audit cache keys.  The result remains a coverage-only
singleton with no diversity or entropy contribution.
