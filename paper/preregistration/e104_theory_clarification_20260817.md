# E104 theory clarification: verified-support entropy redistribution

This clarification was written on 2026-08-17 while E104 was running, before
any post-update evaluation result was inspected.  It changes no implementation,
coefficient, gate threshold, cell, or full-run setting.

The frozen E104 protocol overstates one point in its diagnosis.  Conditional
on a row's leave-one-out peers, the legacy predictor expectation is independent
of that row's sampled action and is therefore admissible as a score-function
control variate.  A structural unseen bucket also does not need its own sampled
gradient row merely to participate in such a baseline.

The implementation defect is instead a mismatch between that open-set model
and verified conditional entropy.  The unseen bucket was included in the
predictor as if it represented successful probability mass even though it had
no verified successful exemplar.  Consequently, an all-same verifier-positive
group could receive uniformly negative semantic pressure, lowering every
observed correct response in favor of unspecified outputs.  The empirical
failure this repairs is correctness leakage from known verified support, not
mathematical invalidity of constant baselines in general.

The v6 estimator is therefore best described as a conservative sampled,
verified-support conditional-entropy gradient surrogate.  It uses open-set
surprisal only to rank sampled verifier-positive modes and centers the applied
advantages across those sampled successful rows.  Thus all-same and singleton
successful groups are exact no-ops, a rare sampled verified mode is balanced
by common sampled verified modes, and every ineligible/failing row receives
zero semantic pressure.

In addition to the frozen exact-value invariants, an exhaustive finite-policy
test enumerates every group of size four.  Across binary and three-mode
successful distributions it requires cosine alignment above 0.95 with the
exact conditional-Shannon-entropy logit gradient, exact zero semantic update
on failure logits, and an exact zero expected update at uniform conditional
success mass.  This test is in
`tests/test_semantic_shannon_group_centered_theory.py`.
