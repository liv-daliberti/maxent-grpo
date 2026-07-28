# E52 scale-free multiplicity and self-retention gate

**Status: FROZEN DURING SENTINEL PASS 2, BEFORE ANY TERMINAL WINDOW — 2026-07-26**

## Motivation

The parent E52 protocol requires the hybrid to beat matched Dr.GRPO at six of
the last eight boundaries and in last-eight mean distinct-correct@8. Those
comparisons are necessary but do not by themselves distinguish two failure
modes already visible in the completed E51 diagnostic cohort:

1. a policy can solve more prompts than control while returning at most one
   verified correct outcome per solved prompt, for which aggregate
   `distinct-correct@8 == pass@8`; and
2. a policy can finish above a weak control after losing most of the diversity
   it had sustained earlier in the same run.

This amendment adds completion checks for those failure modes. It changes no
training process, coefficient, controller observation, entropy reference,
data, sampling seed, or evaluation schedule.

## Information boundary

The checks below use only the registered sampled evaluation statistics and
the run's own history. They do not use a valid-answer catalogue.
They do not use any gold mode count. They also exclude reference-answer
multiplicity, maximum possible distinctness, and every domain-specific target.
They therefore do not tell the policy how high its entropy or number of modes
should be.

For an evaluation boundary `t`, define the observed multiplicity excess

`X_t = distinct-correct@8_t - pass@8_t`.

This is zero when every solved prompt contributes at most one sampled verified
outcome and is positive only when repeated sampling recovers multiple
verified outcomes for at least one prompt.

## Added terminal checks

For each domain, using the exact final eight quarter-pass boundaries:

- the hybrid's mean `X_t` must exceed matched Dr.GRPO's mean `X_t`;
- the hybrid must exceed matched Dr.GRPO in `X_t` at least six of eight times;
- the hybrid must have strictly positive `X_t` at least six of eight times;
- the hybrid's final-eight mean distinct-correct@8 must retain at least half
  of the hybrid's own best rolling-eight mean distinct-correct@8 observed
  anywhere in that run.

The half-retention threshold is a scale-free no-majority-loss criterion. It is
not derived from ground-truth support size or from any domain's observed
terminal score.

These checks are conjunctive with every runtime, entropy-retention,
control-dominance, pass@8, and mean@8 check in the parent protocol. Both the
control and hybrid must write finite evaluations at their exact pass-50 boundary.
Before then, any rolling-eight calculation is diagnostic-only and must remain
`pending`; it cannot pass or fail Stage S.
