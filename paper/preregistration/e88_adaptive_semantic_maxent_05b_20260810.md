# E88: adaptive semantic MaxEnt on verified replay, Qwen2.5-0.5B

**Frozen before submission on 2026-08-10.**

## Question

A fixed semantic coefficient does not deliver a fixed intervention. Does
adapting it so the *realized* pressure is the same everywhere change the
outcome?

## Why this is not coefficient tuning

E81 ran at a fixed `eta = .10`. Its mechanism telemetry shows the realized
ratio of semantic-advantage RMS to task-advantage RMS was not fixed at all:

| domain | realized ratio at `eta = .10` |
|---|---|
| Countdown | 3.7% |
| Python factors | 1.9% |
| Graph coloring | 1.6% |
| MathIR | 1.0% |

Within Python factors the per-seed eligible fraction ranged .19--.79 and the
semantic-advantage RMS varied eightfold at an identical nominal coefficient.
The cause is structural: the centered surprisal of the applied advantage depends
on how many verified modes the prompt's bank happens to hold, so the same
coefficient buys an order of magnitude more pressure in one domain than another.

E88 therefore does not search for a better `eta`. It fixes the *dose* and lets
the coefficient move, so that "semantic MaxEnt at strength `rho`" means the same
thing in every domain and every seed. Whether that is better is the experiment.

## Design

- Model, domains, data, seeds, schedule, decoding, placement, and comparators
  are inherited from E81 unchanged: Qwen2.5-0.5B-Instruct, five domains, seeds
  43--47, exactly eight passes and 3,072 updates, group size 16, learning rate
  2e-7, evaluations and resumable checkpoints every 192 updates.
- 5 domains x 1 arm x 5 seeds = **25 runs**.
- The single difference from E81 is how `eta` is chosen on each update.

## Controller

Let `RMS_sem` and `RMS_task` be the RMS of the applied semantic advantage and of
the Dr.GRPO task advantage on an update, and let `f` be the eligible fraction.
With EMAs over both magnitudes,

    eta_{t+1} = clip( eta_t * ( rho * RMS_task_ema / RMS_sem_ema )^g ,
                      eta_min, eta_max )

registered as: `rho = .05`, `eta_0 = .10`, `eta in [.02, .40]`, gain `g = .5`,
EMA decay `.98`, per-update change capped at a factor `1.1`, warmup 64 updates
during which the EMAs fill and `eta` does not move.

`rho = .05` is chosen from E81's mechanism telemetry, never from any E81
outcome: it sits just above the top of the observed 1.0--3.7% band, so it is a
genuine intervention in every domain rather than a no-op in some and a large
change in others.

### The refusal rule is the design

The controller **freezes** `eta` and does not update its EMAs when the eligible
fraction is below .05, when `RMS_task` is below 1e-3, or when `RMS_sem` is
below 1e-6. A refused observation is not scored as a zero ratio.

This is not a safety afterthought. When a prompt holds one verified mode the
centered semantic signal is exactly zero, so a naive ratio controller divides by
approximately zero, drives `eta` to its ceiling while multiplying a signal that
is identically zero, and then delivers an oversized update the moment a second
mode appears. Every prior controller in this program failed in some version of
that shape. Refusing the observation is what makes RMS targeting admissible
here, and a run whose signal is never usable degrades to the fixed-`eta` arm
rather than to something unbounded.

The controller observes only `RMS_sem`, `RMS_task`, and the eligible fraction.
It never sees evaluation behaviour, gold support, a desired mode count, or a
target diversity. Its state is checkpointed and restored exactly, so a resume
does not silently rerun the warmup.

## Exact exclusions

As E81, plus: no second controller of any kind. The replay dose stays fixed at
.10 and is not adapted; adapting both at once would not be identifiable. Token
and sequence entropy, SEED, novelty, balance, bank entropy shaping,
counterfactual proposals, singleton escape, and reference KL all remain off.

## Outcomes and estimands

At every registered half-pass checkpoint, per domain: greedy pass@1, sampled
mean@8, sampled pass@8, mean distinct correct modes@8, excess multiplicity.

The **primary** comparison is the paired seed difference `adaptive - fixed` at
pass 8 for `distinct@8` and `pass@8`, against E81 cell by cell. That isolates
adaptation of the coefficient, because every other setting is inherited. The
secondary comparison is `adaptive - replay` against E78, which reports the
combined effect of semantic MaxEnt with an adapted dose.

Show all five paired seed differences and their mean and range.
Do not pool domains into one effect and
do not select a best checkpoint.

## Mechanism gate, checked before any outcome is read

1. Realized ratio reaches `rho` within a factor of two by pass 2 in at least
   three domains.
2. `eta` is pinned at a bound on no more than 20% of applied updates.
3. The frozen fraction is reported per domain; a domain that freezes on more
   than 80% of updates is reported as *not adapted*, and its outcome is read as
   the fixed arm rather than as evidence about adaptation.
4. Mean response length does not exceed the fixed arm's by more than 25%, and
   the no-EOS fraction stays below .05.

Failing 1 or 2 means the controller is mis-specified. The registered response
is to stop and re-derive, **not** to widen the bounds and rerun.

## Registered interpretation

1. Adaptive beats fixed in most domains: report equalized pressure as the
   better way to dose semantic MaxEnt, and the fixed coefficient as an
   incidental choice.
2. No difference: report that the realized-pressure variation seen at fixed
   `eta` does not matter for outcomes, which is itself informative and retires
   the adaptive machinery.
3. Adaptive is worse: report that the fixed coefficient was accidentally
   well-matched, and that equalizing pressure trades away a useful
   domain-dependent dose.

## Integrity

All 25 cells are submitted held from one hash-bound snapshot and released only
after their scheduler environments pass an exact audit. The snapshot extends
E81's patch set with the controller module and its wiring; the audit fails
closed on any divergence outside that declared set. A non-finite loss, a
traceback, a duplicate run directory, controller state that fails to restore,
or a missing pass-8 endpoint fails closed. No failed scientific run is silently
replaced or excluded.
