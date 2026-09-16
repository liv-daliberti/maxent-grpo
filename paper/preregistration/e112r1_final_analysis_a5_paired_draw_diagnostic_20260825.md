# E112-R1 final-analysis A5 paired-draw diagnostic

Frozen: 2026-08-25T13:03:05-04:00 while E112-R1 was 49/75 terminal,
E109 was 13/15 terminal, and the complete official E112 result did not exist.
No additional task-evaluation endpoint was inspected to make this amendment.
The previously viewed private E112 prefix cannot be undone, so every quantity
added here is an exploratory disclosure and never a confirmatory test.

This amendment changes no training cell, comparator, endpoint definition,
registered contrast, aggregation, interval, threshold, or decision rule. The
original E112 decision remains the five-seed, draw-averaged historical
contrast on terminal `sampled_excess8`, with its registered pass safeguard.

## Trigger

The frozen builder retained raw distinct@8 in its machine-readable result, but
the registered forest showed pass@8 and the derived subtraction
`distinct@8 - pass@8`. It also averaged the four common evaluation draws before
reporting training-seed intervals. That is faithful to the original protocol,
but it can visually hide a simultaneous accuracy-and-raw-breadth gain and does
not reveal whether a family estimate is sensitive to the four generation
draws.

E112-R1 and its historical ReplayDr comparators also use different frozen
request/source plumbing. Exact prompt and response-free request identities are
audited, but any step-zero endpoint offset remains evidence about the bundled
historical-comparator surface rather than a training effect.

## Frozen secondary diagnostics

For sampled pass@8, mean correctness@8, raw distinct correct modes@8, and the
derived adjusted breadth, retain every paired training-seed/common-draw effect
at terminal and normalized trajectory AUC. Every four-draw mean must reconcile
exactly with the registered draw-averaged effect or the build fails.

For each family and summary, report separately:

- training-seed SE over the five draw-averaged paired seed effects;
- evaluation Monte Carlo SE over the four seed-averaged common-draw effects;
- evaluation Monte Carlo SE and all four draw effects within every seed.

Do not pool the two axes into one interval, p-value, or effective sample size.
Four draws provide a sensitivity diagnostic, not a precise sampling-error
estimate.

Also report the paired step-zero historical offset and the baseline-centered
sensitivity

`(E112(step) - E112(0)) - (ReplayDr(step) - ReplayDr(0))`.

For normalized AUC, subtract the paired step-zero offset from the raw
historical AUC contrast, which is algebraically the AUC of the two
gain-from-start trajectories. This sensitivity can reveal a fixed baseline
surface offset; it does not remove time-varying sampler/source differences and
is not an isolated semantic, proposal, or replay effect.

## Presentation and interpretation

Retain the two registered pass/adjusted-breadth forests unchanged. Add
supplementary terminal and AUC forests for the primitive vector
`(pass@8, raw distinct correct modes@8)` using the same paired five-seed means
and Student-t intervals. The primitive supplement and paired-draw diagnostics
cannot change either registered binary decision.

If the registered and baseline-centered signs disagree, lead with the
registered result and disclose the historical-plumbing sensitivity. If the
training-seed and evaluation-Monte-Carlo axes disagree materially, call the
estimate unstable rather than choosing the favorable axis. Component
attribution remains assigned to the same-plumbing E117 C/P/F successor.
