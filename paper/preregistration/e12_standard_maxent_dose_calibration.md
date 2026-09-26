# E12 prospective calibration: standard sequence-MaxEnt coefficient

**Status: EXPLORATORY ENGINEERING CALIBRATION, FROZEN BEFORE E12 OUTCOMES
(2026-07-17).** E12 does not change E11's active objective or estimator:

```text
J(theta) = E[R] + alpha H(pi_theta),
```

with E10's exclusive old/new prefix-ratio estimator and exactly one shared
Dr.GRPO `1/T_max` update normalization. E11 established that literal
`alpha=0.05` strongly actuates raw entropy but fails by length 87 and 7/16
no-EOS rollouts at step 32. E12 calibrates the coefficient's raw-sequence
units before any controller or analytical grid is reconsidered.

## Fixed-dose design

Four independent fixed arms use graph coloring, Qwen2.5-0.5B-Instruct, seed
9005, group size 16, four-row backward microbatches, one PPO epoch, PPO clip
0.2, 128 updates, and quarter-pool evaluations:

```text
alpha in {0.0005, 0.0010, 0.0015, 0.0020}.
```

This bracket lies between engineering-equivalent historical endpoints:
E10 normalized `alpha=0.05` corresponds to standard `0.0002604`, while E10
normalized `alpha=0.50` corresponds to standard `0.0026042`. Historical
traces remain labeled E10 and are not pooled into E12 outcomes.

E12 is a single-seed engineering calibration, not comparative evidence. It
does not include a fresh `alpha=0` control, and its four-point grid does not
land exactly on either historical endpoint. Those limitations are accepted
because this gate only asks whether a finite standard-MaxEnt coefficient is
safe enough to justify a separately preregistered controller smoke.

Every arm must contain contiguous telemetry for steps 97--128 and the frozen
step-128 endpoint with finite loss, gradient, raw entropy,
per-`T_max` entropy, prefix-ratio, response, reward, and evaluation telemetry.
At every observed step, raw entropy divided by 192 must equal the separately
named normalized diagnostic, the fixed coefficient must equal the assigned
dose, and no adaptive-controller telemetry may be present.

An arm is **behaviorally safe** only when:

- it has a positive reward in its final 32 updates;
- its final-16 mean response length is at most 32 tokens;
- no final-16 batch has mean response length above 64;
- no final-16 batch has more than 2/16 no-EOS rollouts;
- final average evaluation accuracy exceeds 0.05.

It is **entropy-effective** only when final-16 mean raw entropy is at least
half the frozen raw target, `4.9840172700576` nats. The preferred dose is the
largest dose satisfying both definitions. Peak entropy, length, no-EOS,
accuracy, pass@8, and coverage@8 are reported descriptively and cannot rescue
a failed guard.

If no dose is both safe and entropy-effective, E12 selects no coefficient and
the next method must add an explicit expected-length constraint. Passing this
calibration authorizes a separately frozen adaptive smoke; it never
automatically submits proportional, dual, or analytical jobs.

The actual v2 jobs use separate immutable source snapshots made from the same
checkout. Their learner and MaxEnt source hashes were verified byte-identical
before any training outcome was observed. The abandoned v1/a0p0005 namespace
contains only a header-only manifest from a scheduler-inaccessible launch
attempt and produced no job or outcome.

## Frozen outcome

Jobs 30007803--30007806 completed all 128 updates with exit code zero. The
prospective checker selected no coefficient:

| `alpha` | final-16 raw entropy | final-16 mean length | max final-16 length | max no-EOS | accuracy | coverage@8 | decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0005 | 1.794 | 4.14 | 4.69 | 0/16 | 0.198 | 0.179 | safe; entropy-ineffective |
| 0.0010 | 2.113 | 4.63 | 13.19 | 0/16 | 0.203 | 0.170 | safe; entropy-ineffective |
| 0.0015 | 1.715 | 4.23 | 4.69 | 0/16 | 0.172 | 0.107 | safe; entropy-ineffective |
| 0.0020 | 48.464 | 60.76 | 121.94 | 10/16 | 0.172 | 0.104 | entropy-effective; unsafe |

Thus the tested unconstrained objective exhibits a sharp transition from
short, low-entropy concentration to length/EOS runaway. E12 is only a
single-seed engineering calibration, so it does not establish a population
threshold; it does establish that none of the prospectively tested doses is
safe enough to seed the adaptive smoke. The 54-cell analytical grid remains
held. Per the frozen rule, the next intervention must constrain expected
length explicitly rather than continue coefficient search.
