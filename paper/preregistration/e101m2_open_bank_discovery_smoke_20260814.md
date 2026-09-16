# E101m2: open-bank discovery-opportunity smoke

Date frozen: 2026-08-14, before submission and after E101m completed.

## Motivation

E101m r1 completed eight updates in 121 seconds. It established a clean
singleton-only proposal opportunity, one untouched-original-prompt group, zero
proposal leakage, and four ordinary replay-actuation updates. Its only proposal
group contained 16 rows but zero task-positive or validator-positive rows, so
there was no candidate to admit. This is discovery failure, not evidence about
the bank-balance gradient.

E101m2 changes only discovery opportunity. It is separately named and is not a
performance arm or an after-the-fact replacement for E101m.

## Frozen cell

- Model/source: the same frozen Qwen2.5-0.5B-Instruct snapshot as E101/E101m.
- Objective: task-only Dr.GRPO plus replay mass 0.10 and known-bank balance
  0.10. Every semantic/PPO-advantage coefficient is zero.
- Isolation: separate proposal objective support, singleton-only activation,
  deterministic transforms off, zero proposal rows to PPO, and no gold,
  support-size, or evaluation feedback.
- Discovery: up to three fresh samples of 16 rows from the untouched original
  prompt for each eligible singleton update. The registered temperature sweep
  is 1.0, 1.2, and 1.4 and stops at the first novel verified candidate group.
- Data: the same frozen Countdown split, all 32 training prompts, seed 102, one
  pass, hence 32 optimizer updates.
- Evaluation: steps 0 and 32, batch size 8, K=4, one fixed draw. Accuracy and
  diversity are runtime-integrity readouts only.
- Sampling plumbing: replicated free-form sampling, one GPU per actor, and
  local actor weight synchronization, identically required by E101r1.
- Placement: `all`/`mltheory`, one generic GPU from the frozen healthy >=24 GB
  whitelist `node[007,020,022-023,101,103,202,204-206,302,403,805]`, 8 CPUs,
  22 GB host memory, and a hard 30-minute limit.

## Decision

1. No singleton-eligible updates means ordinary verified discovery is too rare
   in this tiny split.
2. Eligibility but zero validator-positive proposal rows confirms that
   unconditioned proposal quality is the bottleneck.
3. Validator-positive rows but no novel outcome indicates collision with the
   incumbent/known bank rather than generation failure.
4. Admission with any PPO/objective-support/feedback leakage is invalid.
5. Admission without replay actuation at or after the admission is a scheduling
   failure.
6. Admission, zero leakage, and later replay actuation validates the clean
   new-mode mechanism and unlocks the matched E101r1 comparison.

No coefficient, prompt, decoding, data, or attempt-count change is allowed after
metrics materialize. Any further explorer change receives a new experiment name.
