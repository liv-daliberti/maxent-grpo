# E129 amendment 2 — a high-coefficient extension, and why it was added — September 19, 2026

Amends `e129_drgrpo_reference_kl_05b_20260918.md`. This amendment is written
**before** the extension it registers is submitted, and it adds arms to a design
whose first cells have already been read. Both facts are stated here because
they bear on how the extension should be weighed.

## What was seen first

Seven of the registered 75 cells have reached pass 8, all of them Countdown,
seeds 43, 46 and 47. Terminal PMD, computed with the published estimator
against the 32-disjoint-stream resample of the matched E78 arms:

| arm | pass@8 | PMD | n |
| --- | --- | --- | --- |
| Dr.GRPO control | .512 | .000 | 3 |
| KL, beta = .001 | .619 | .052 | 3 |
| KL, beta = .01  | .635 | .225 | 2 |
| KL, beta = .04  | .652 | .320 | 2 |
| Re:Dr replay    | .673 | .494 | 3 |

Registered prediction (2), coefficient independence of terminal PMD, is
disconfirmed on this evidence: the spread runs .052 to .320, is monotone in
beta, and is consistent across seeds. That prediction tested an asymptotic
statement, Proposition "The retained conditional is the reference's", with a
finite-time experiment; the flow's rate result makes a coefficient ordering the
expected finite-time behaviour, and the registered prediction was the wrong one
to write. The disconfirmation stands as recorded.

## What the extension tests, registered before it runs

The three registered coefficients all sit where the flow predicts no
correctness cost at all. With the frozen reference's measured per-sample
correctness `mu(C) = .0139` and the Dr.GRPO coefficient at `G = 16`,
`logit P* = logit mu(C) + c_G / beta` gives a stationary correctness of
essentially 1 for every beta at or below .04. The interesting region is above
it:

| beta | lambda | predicted stationary correctness |
| --- | --- | --- |
| .04 | 23.4 | 1.000 |
| .10 |  9.4 |  .994 |
| .15 |  6.3 |  .880 |
| .20 |  4.7 |  .605 |

Two arms are therefore added, `beta = .10` and `beta = .20`, at the same five
domains and the same seeds 43-47, 50 cells in total, submitted under the
identical configuration in every other respect.

**Prediction (4), correctness knee.** Terminal mean per-sample correctness at
`beta = .20` is materially below `beta = .10`, and below the registered
`beta = .04` arm, in a majority of domains. Verified replay has no analogous
knee: its full-coverage limit sends correctness to 1 for every positive dose,
so the E78 replay arm's correctness does not fall with its replay weight.

**Prediction (5), breadth ceiling.** Terminal PMD continues to rise with beta
but does not exceed the frozen reference's own success-conditional PMD, which
is .696 in Countdown. A KL arm exceeding its reference's breadth would
contradict the stationary-conditional result directly.

Disconfirmation of either is a result about the flow's applicability to neural
training and is to be reported as such.

## Status of the addition

These arms were chosen after seeing the table above, and would not have been
run had the three registered coefficients spanned the knee. They are an
extension driven by an observed trend, not part of the frozen design, and any
report must distinguish them from the registered three. What makes them
admissible is that predictions (4) and (5) are quantitative, derived from a
result written before any cell of this cohort existed, and recorded here before
submission.

Nothing in the registered design changes: the original three arms, their
domains, seeds, schedule, outcomes, estimands and failure policy all stand.
