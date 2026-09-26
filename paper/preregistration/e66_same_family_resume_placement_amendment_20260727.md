# E66 same-family resume placement amendment

Recorded 2026-07-27 after the E66 scientific protocol and trajectories were
frozen, after scheduler preemptions of the affected jobs, and before this
placement change.

The nine non-Math E66 controls are pending on `lowprio` under account `allcs`.
They are pinned to a single physical node per accelerator family: Graph to
A6000 `node103`, and Countdown/Python to RTX 3090 `node024`. At the time of
this amendment, eight jobs have no scheduler start estimate and the remaining
Graph job is estimated for August 3. Their matched E68 jobs have materially
earlier estimates on `pvl-lowprio`.

This infrastructure-only amendment moves jobs
`30128394--30128402` to `pvl-lowprio` under account `mltheory` and broadens
placement only within the original accelerator family:

- Graph remains A6000 and may use `node103,node104`.
- Countdown and Python remain RTX 3090 and may use
  `node020,node021,node022,node024,node026`.

Run stamps, job IDs, frozen source and execution snapshots, model, data,
seeds, objective, sampling, evaluation cadence, checkpoint cadence, and every
scientific setting remain unchanged. Running MathIR controls
`30128403--30128405` are untouched.

All affected jobs must be held and pending before mutation. Existing
checkpoint state and metrics are retained. Jobs resume under the already
registered watchdog/resume contract; the pre-intervention E66/E68 equivalence
audit and the E66 integrity audit remain fail-closed. The amendment document
and executable script are SHA-256-bound in a machine-readable artifact before
the jobs are released.
