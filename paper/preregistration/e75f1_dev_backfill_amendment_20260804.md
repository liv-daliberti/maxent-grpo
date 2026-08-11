# E75F1 development-gate backfill amendment

**Frozen:** 2026-08-04, after the memory/placement amendment and before
development job 30258125 started or produced any output.

E75R3's identical 32-map evaluator completed in 1 minute 57 seconds. E75F1
doubles the map count to 64 and performs no optimizer updates. Its original
eight-hour scheduler limit prevents useful backfill under current fair-share
priority even after its request became resource-feasible.

This operations-only amendment changes job 30258125's wall-time reservation
from eight hours to 30 minutes. The evaluator code has no internal time-based
stopping and must still finish all 64 maps and 512 trajectories or fail closed.
The 30-minute limit is more than fifteen times the observed E75R3 runtime and
more than seven times a linear map-count extrapolation. No model, data,
sampling, metric, gate, dependency, or downstream scientific setting changes.
No E75F1 development output existed when this amendment was frozen.
