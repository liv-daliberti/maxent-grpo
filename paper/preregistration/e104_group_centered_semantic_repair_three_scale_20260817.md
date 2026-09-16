# E104: score-function repair for semantic MaxEnt on ReplayDr.GRPO

**Frozen before any repaired model was trained or evaluated on 2026-08-17.**

## Failure being repaired

The legacy semantic-on-replay estimator used an open-set predictive
distribution containing one structural unseen bucket.  It subtracted the
expected clipped surprisal under that distribution from every sampled correct
row.  The unseen bucket has no sampled policy-gradient row, and the
leave-one-out predictive distribution (hence its expectation) also changes
with the sampled outcome.  This is not a valid action-independent score-
function baseline.  In particular, an all-correct group containing one
observed mode could receive uniformly negative semantic advantages.  With a
zero Dr.GRPO task advantage, the only update then lowered the likelihood of
every sampled correct response and leaked probability to unsampled outputs.

E104 replaces only that estimator.  For the eligible verifier-positive rows
`E_x` sampled for prompt `x`, define

    u_i = min(-log qhat_-i(a_i), C) / C
    A_sem_i = eta * (u_i - mean_{j in E_x} u_j).

`qhat_-i` retains the catalogue-free prompt-local history, leave-one-out
successful peers, pseudocount one, and one structural unseen bucket.  The
bucket is used only to estimate the probability of a sampled new mode.  It is
not a baseline observation and has no gradient row.  Incorrect, inactive, and
unparseable rows receive exact zero and do not enter history.

This is the sampled score-function estimator for conditional answer entropy.
It has the following pre-registered invariants: all-wrong, singleton-success,
and all-same-success groups have exact zero semantic pressure; every prompt
group's eligible semantic advantages sum to zero; a sampled rare successful
mode is positive and is balanced by sampled common successful modes; and
`|A_sem_i| <= eta` because both `u_i` and its sampled mean lie in `[0,1]`.

The legacy estimator and its v5 checkpoint schema remain unchanged.  The
repair is a separate configuration mode with checkpoint schema
`semantic_shannon_tracker_v6_group_centered`.

## Mechanism cohort

This gate runs the repaired treatment at all three registered model scales:

- Qwen2.5-0.5B-Instruct, seed 43;
- Falcon3-1B-Instruct, seed 55; and
- Qwen2.5-3B-Instruct, seed 70.

Each scale runs Graph Coloring, Countdown, Python Factors, MathIR, and
PantryPlan: 3 models x 5 static domains = 15 cells.  PointMaze is excluded.
Each cell uses the paired replay cohort's model, data, prompt surface,
optimizer, group size 16, decoding, and fixed replay coefficient 0.10.  It
trains on the first 64 registered training prompts for one pass (64 optimizer
updates), with evaluations at steps 0, 32, and 64.  The semantic coefficient
is fixed at `eta = 0.10`; the legacy RMS controller is disabled.

The treatment is uniform verified-likelihood ReplayDr.GRPO at weight 0.10 plus
the v6 semantic advantage.  No token-entropy objective, adaptive coefficient,
quality gate, legacy signed estimator, novelty reward, counterfactual
proposal, support target, evaluation feedback, or PointMaze code path is
enabled.

## Outcome-blind mechanism gate

The full E105 cohort may be released only if all of the following are true:

1. the exact-value unit tests for all-wrong, singleton, all-same, and
   common-versus-rare groups pass, together with legacy semantic checkpoint and
   argument tests;
2. all 15 cells reach update 64 with finite telemetry and no traceback,
   assertion failure, CUDA error, or non-finite loss;
3. the v6 active flag is one and the v5 legacy-active flag and RMS-controller
   flag are zero on every recorded training update;
4. the absolute group-centered effective-advantage mean is at most `1e-8` on
   every update, and the maximum absolute applied semantic advantage is at
   most `0.1000001`;
5. verified replay produces an applied replay update in every cell; and
6. at least one cell at each model scale records a nonzero repaired semantic
   RMS, proving that the live learner—not only the unit test—applied the term.

Whether a particular domain produces a multi-mode eligible group in only 64
updates is descriptive and cannot fail the gate.  No evaluation correctness,
pass@K, distinct@K, model selection, coefficient selection, or comparison to a
prior endpoint is inspected by this gate.

## Full follow-up fixed now

If E104 passes, E105 runs the same frozen source snapshot and objective for all
five registered seeds at each scale: Qwen seeds 43--47, Falcon seeds 55--59,
and Qwen-3B seeds 70--74, across the same five domains. E105 contains 75
repaired treatment cells at eight passes and 3,072 updates per cell, evaluated
every 192 updates. Each cell is paired with its already registered
ReplayDr.GRPO cell.
No coefficient or domain-specific setting may change after E104 telemetry is
seen.  If E104 fails, the full cohort is not released; the failure and its
mechanism are reported before any new estimator is designed.
