# Conditional-concentration protocol review (before outcomes)

## Recommended estimands

For fixed model/prompt/checkpoint/decoding law, let R be the number of correct
outputs, n_c their canonical-key counts, and C(q)=sum_c q_c^2. For R>=2 use
C_hat=sum_c n_c(n_c-1)/[R(R-1)]. Keep missing values undefined when R<2.
A positive before-to-after delta means increased concentration; a negative
replay-minus-control delta means less concentration with replay.

For each registered paired seed s, define E_s from complete matched prompt
receipts with R>=2 in BOTH conditions. Average the paired prompt differences
equally over E_s, then average those seed estimates equally. This is an
eligibility-conditioned target: it does not estimate all-prompt mean C.
Report all seed-level eligible counts/fractions, excluded categories, source
completeness, exact paired seed IDs, and each seed estimate. Different E_s are
allowed but explicitly produce seed-specific selected prompt populations.

Mandatory population sensitivity: E_cap=intersection_s E_s across the
prespecified admitted paired seed set; recompute each seed on that same fixed
prompt set. If E_cap is empty, report undefined, without weakening the rule.
An empty E_s likewise leaves that seed's effect undefined. Do not silently
turn five source-complete seeds into four supposedly complete effect seeds.

Before/after analysis uses each method's own initial checkpoint with matching
prompt/validator/decoding identity. Endpoint replay contrasts compare matching
fresh objectives. A difference-in-differences needs one common prompt
intersection across every condition/checkpoint entering the four-term
subtraction. It should remain secondary; it does not prove survival of named
banked keys or that concentration mediates improved correctness.

## What is mathematically unbiased, and what is not

Under iid outputs from ONE fixed policy law, the correct labels conditioned
on R=r are iid q. For every r>=2, E[C_hat|R=r]=C(q), because every unordered
pair agrees with probability sum q_c^2. Thus conditioning on that policy's
valid count does not by itself bias this promptwise collision estimator.

For a paired E_s selected on BOTH policies' counts, extending that identity
to conditional paired unbiasedness requires independent evaluation randomness
across the conditions, or the weaker joint conditional independence between
label composition and all eligibility counts. Shared sampling RNG streams
can break it. If that joint condition is unverified, retain the single-policy
identity and call the paired analysis a descriptive count-conditioned contrast.

Concrete counterexample: q_A=q_B=(1/2,1/2), P_A=1/2, P_B=3/4, and two draws
per policy using the same independent uniforms U_1,U_2 with standard category
intervals. Common eligibility requires both U_i<1/2. A's two correct labels
then collide with probability 1/2; B's restricted labels have probabilities
(3/4,1/4) and collide with probability 5/8. The conditional contrast is +1/8,
even though the true conditional concentrations are identical. This is a
warning about joint selection under coupled randomness, not a reason to
replace pairing with separately selected prompt populations.

Existing hosted collision pools pair counts within domain. Its population
analogue weights prompts by P_x^2 when draw budgets are equal; it is not an
equal-prompt mean C. Different correctness profiles can change that aggregate
without changing any prompt's q. Preserve it as the existing descriptive
summary, and distinguish it from the new equal-prompt paired estimand.

## 32-draw primary and 8-draw sensitivity

Pooling four K=8 groups is valid only within an unchanged fixed policy,
checkpoint, validator, task rendering, and decoding law, with distinct iid
request/sample identities. The pooled valid labels still follow q. Do not
pool across checkpoints, training seeds, model arms, or changed prompts.
Cross-group pairs are permitted under that identical-law assumption.

For the prespecified K=8 sensitivity, retain original matched prompt-draw
units with R>=2 in both conditions, average paired differences within prompt,
then average prompts. Never pool all eligible groups with equal group weight
if that changes prompt weights, and never treat four groups as training seeds.
Report its prompt and prompt-draw coverage. Also recompute the 32-draw effect
on the same eligible PROMPT set so a budget comparison does not silently mix
sampling variability with a changed prompt population. If only one direction
has sufficient eligibility, leave the other sensitivity undefined.

## Intervals and aggregation

Use the paper's existing two-sided Student-t interval over the five
registered paired seed means: mean +/- 2.7764451051977987 * sd(delta_s)/sqrt(5).
Label it nominal/descriptive seed variability, with no multiplicity correction,
and not a confidence guarantee for all prompts or all decoding randomness.
A partial seed intersection has exact seed counts and descriptive values,
with no five-seed interval. A zero estimated variance or all observed C=1
is not evidence of population equivalence or deterministic single-mode output.

Report every prespecified domain/model/level block. Cross-domain summaries
must first average domains equally within a common seed intersection, with
all required domains defined; then summarize seeds. Domain-specific seed sets
cannot be manufactured into a paired macro interval. Such a macro may be
reported only as a separately labeled descriptive average of domain means.
Cross-model or cross-level patterns do not identify a causal size/difficulty
effect. Keep hosted bootstrap intervals as prompt-cluster intervals; pairs
sharing responses and outputs sharing prompts are not independent replicates.

## Exact link to the mean-flow theorem

For dq_c/dtau=q_c(q_c-C),
  dC/dtau=2*sum_c q_c^2(q_c-C)
         =2*[sum_c q_c^3-C^2]
         =2*Var_{c~q}(q_c)>=0.
Equality holds when q is uniform on its positive support. A unique initial
maximum converges to C=1 in the categorical collapse theorem. This links the
empirical conditional concentration to precisely the same mathematical
quantity, while preserving the theorem's independent-logit/Euclidean
assumptions. Neither the C statistic nor the replay contrast identifies
which unobserved key disappeared or certifies positive probabilities.

## Agreed additional check after identifying coupled-RNG selection

The frozen analysis should add two disjoint-stream 16-draw orientations:
A draw groups [0,1] versus B [2,3], and A [2,3] versus B [0,1]. Each
orientation retains its own eligibility set; never intersect those sets.
Compute the across-seed fixed prompt intersection separately per orientation.
The source adapter must verify actual distinct sampling streams, not merely
different list positions or filenames. Separate groups cannot repair a
shared-law or duplicate-request failure.

Report both orientation estimates and their coverage. Their average within
seed is a useful descriptive summary when both are defined, but selecting
seeds where both are defined conditions on the opposite fold's counts again;
do not promote that complete-case average to a general conditional-unbiased
estimator under arbitrary shared-RNG coupling. Main interpretations should
consider concordance across the 32-draw, disjoint-stream, and fixed-population
views, rather than selecting the most favorable estimate.

Initial output receipts that are reused across training-seed labels also
share sampling error. Five-seed intervals then describe variability conditional
on those saved initial receipts, not the uncertainty of an independently
resampled initial policy. Preserve source identities and make that distinction
when source metadata reveals reuse.

## Synthetic implementation validation

`python -m pytest -q tests/test_paper_collision_statistics.py` passed all 26
tests. Tests use synthetic samples only and do not read experiment outcomes.
They cover direct unordered-pair oracles, exact conditional-count expectation,
shared-RNG selection bias and independent-stream repair, equal-prompt versus
pair-weighted aggregation, empty eligibility, disjoint orientations, five-seed
Student-t arithmetic, malformed inputs, and the categorical concentration
Lyapunov derivative. No manuscript was modified by this audit.
