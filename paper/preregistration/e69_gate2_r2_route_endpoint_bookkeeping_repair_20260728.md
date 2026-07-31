# E69 Gate 2 R2 route/endpoint bookkeeping repair

Date frozen: 2026-07-28, after the E69-R1 Python successor failed at training
step 302, before its first step-384 checkpoint, and before any repaired-run
outcome or temporal-route observation existed.

## Trigger and disposition of R1

E69-R1 Python job `30168831` failed closed with:

```text
ValueError: verified route identity changed for one source prompt
```

The learner failed before producing a durable checkpoint. Its silent wrapper
was subsequently canceled; its log, scheduler accounting, run directory, and
R1 identity remain immutable failed-attempt provenance. The failure is not
classified as scheduler damage, and no R1 Python prefix may be spliced into
R2.

R1 therefore cannot pass. R2 is a prospective implementation repair under a
new identity, not an execution-only continuation or a retrospective
whitelist.

## Diagnosed contract mismatch

The frozen E69 protocol separates:

1. a prompt-local endpoint identity; and
2. a literal-abstracted route identity that may recur across prompts.

The route library stores at most one replay exemplar for each
`(verifier, route signature, source prompt)` tuple. Python factors can
legitimately produce two different correct endpoint vectors from two distinct
lambda expressions that share the same literal-abstracted route skeleton on
the same prompt.

The R1 implementation incorrectly treated a changed endpoint as a changed
route identity inside that one record. This is contradicted by the protocol's
two-level identity and deterministically reproduces outside training with one
prompt, one route signature, and two valid endpoint keys.

## Frozen correction

The record key, route signature, counters, recurrence definition, replay
budget, proposal budget, and gate remain unchanged.

For a repeated source-prompt/route observation:

- verifier and route signature must remain identical;
- distinct response token sequences may have distinct prompt-local endpoint
  keys;
- the lexicographically smallest response-token tuple is retained as the
  deterministic replay exemplar, and its endpoint key is updated atomically
  with it;
- if an identical response-token tuple changes endpoint key, fail closed,
  because deterministic verification of identical executable content may not
  change its endpoint;
- neutral route occurrences are still counted, but a second endpoint on the
  same source prompt does not create a second source prompt or a cross-prompt
  recurrence.

No coefficient, sampling setting, seed, dataset, prompt, validator, route
canonicalizer, optimizer, checkpoint cadence, stopping rule, outcome gate, or
temporal gate changes.

## Minimal rerun scope

R2 replaces only the failed Python route-successor cell with a fresh
initialization. Countdown, MathIR, and any terminal Graph successor trace that
never raised this invariant could not have entered the repaired branch; their
accepted input streams therefore have identical behavior under the correction.
The remaining control arms do not instantiate the route library.

If another non-Python successor cell raises the same invariant before
termination, this minimal-scope premise fails and R2 must not be used.

The replacement uses seed 43, the frozen Python data and model, six prompt
passes, 16 samples, learning rate `2e-7`, three proposal-control groups, one
route-replay group, the existing evaluation/checkpoint cadence, and the same
stable A5000 pool. It starts from initialization and may not resume R1.

## Advancement rule

Gate 3 and MATH-500 remain sealed. Advancement still requires:

- all effective Gate-2 cells terminal and integrity-clean;
- all exact temporal checkpoints present;
- nonzero Python post-replay neutral reproduction;
- clean Python terminal performance;
- the unchanged outcome and compute gates; and
- a final frozen audit status of `pass`.

R2 is frozen from the exception and identity semantics only. No repaired
training outcome was available when its repair rule and rerun scope were
chosen.
