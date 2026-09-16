# E117-A1 actuable-pressure audit correction

Frozen: 2026-08-24 while all 12 E117-R1 training jobs are pending at zero
runtime, before any run artifact or endpoint exists. PointMaze remains excluded.

The original mechanism audit required F's applied semantic RMS to be positive
whenever a group had at least two eligible verified-support members. That is too
strong: a symmetric predictor can assign equal probability to two modes, making
the raw predictor-centered contrast exactly zero. Correct behavior is then
zero applied pressure despite multi-support eligibility.

Replace that condition with:

- record multi-support eligibility independently;
- define an actuable update by strictly positive raw eligible semantic-advantage
  RMS;
- require strictly positive effective RMS on every actuable F update; and
- require C/P effective RMS to remain exactly zero.

Also make the already-registered isolation rules explicit in the executable
audit: every arm must have zero objective-support delta, gold-support feedback,
evaluation feedback, desired-mode-count feedback, proposal/control/transform
rows sent to PPO, and adaptive retention priority. C must additionally have
zero tracked proposal admissions.

This corrects a false-failure possibility and strengthens leakage detection. It
does not inspect or alter training, change an endpoint, or relax a real
mechanism requirement. Replace pending audit job 30873707 with a dependency on
the same 12 E117-R1 jobs and a newly content-addressed copy of the corrected
audit.
