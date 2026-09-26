# E69 Gate 2 pending executable-domain placement repair

Date frozen: 2026-07-28, before any replacement was submitted and before any
Graph, Countdown, or Python Gate 2 job started.

## Scheduler failure

All 12 Gate 2 jobs for Graph, Countdown, and Python remained pending with zero
runtime, no run directory, no optimizer record, no checkpoint, and no
evaluation result. Slurm's scheduler forecast placed their earliest starts
between 2026-07-31 and 2026-08-03, with six jobs having no estimated start,
while suitable low-priority accelerator nodes were idle.

The original requests were overconstrained to three A6000 nodes for Graph and
mostly drained RTX 3090 nodes for Countdown/Python. This is a placement failure,
not a scientific failure.

## Frozen replacement placement

Exactly the 12 never-started cells are replaced:

- all four Graph arms run on `node101`, `gpu:a40:1`, partition `lowprio`,
  account `mltheory`;
- all four Countdown arms and all four Python arms run on `node105`,
  `gpu:a5000:1`, partition `lowprio`, account `mltheory`.

Every arm within a domain therefore uses the same accelerator class. Cross-arm
compute matching, not cross-domain wall-clock speed, is the scientific
requirement.

No model, source, execution code, data, prompt, verifier, seed, rollout,
sampling-control, replay, optimizer, six-pass stopping, evaluation,
checkpoint, watchdog, CPU, or memory setting changes. The replacements use the
exact Gate 2 source and execution snapshots.

## Atomic attempt selection

The replacement cohort is submitted held and audited before the original 12
pending jobs are cancelled. The replacements are released only after every
original job is confirmed cancelled and every replacement is confirmed held
with its frozen environment.

The original pending jobs are permanently excluded; they contain no
outcome-bearing attempt. The repair identity records the exact one-to-one
mapping. The Gate 2 analyzer substitutes only these cells and never combines
original and replacement attempts.

This decision is based only on scheduler state and idle-node availability.
No Graph, Countdown, Python, terminal MathIR, terminal MATH-development, or
MATH-500 outcome informed it.
