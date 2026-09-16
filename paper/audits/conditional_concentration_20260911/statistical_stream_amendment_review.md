# Sampling-stream amendment: independent statistical review

This supersedes the earlier assumption that four saved K=8 groups provide
32 independent draws and that the two pairs of groups are independent
16-draw orientations. It is based on the verified sampler metadata, before
empirical concentration effects are computed.

## Admissible primary sampling units

For the certified vLLM source, child seed = parent seed + option index and
parent seed = base + draw index. The four groups therefore overlap in nominal
random streams. With four groups of eight, the usual stream multiplicities
are [1,2,3,4,4,4,4,4,3,2,1], giving 11 distinct streams rather than 32.
The actual certified stream IDs determine each endpoint's budget; do not
hard-code eleven for another sampler, including the Pantry canonical sampler.

Deduplicate within a fixed prompt/endpoint, selecting the earliest
(draw_index, option_index) receipt for each certified stream identity.
Selection is structural and must not inspect correctness, key identity,
response length, or agreement among repeated occurrences. Apply the rule
independently at each endpoint. Sorting/selection must be invariant to the
order in which input rows happen to be loaded. Never deduplicate across
prompts merely because their numeric seed values match.

Retain every discarded occurrence in provenance. Report verified-key
agreement including invalid=None as a diagnostic. Disagreement can arise
from batch/runtime nondeterminism; it neither turns repeated nominal streams
into independent draws nor justifies excluding them conditionally on outcome.
Distinct nominal seeds are a source property, not a proof of mathematical
independence under arbitrary runtime effects. State the fixed-law/independent
stream idealization explicitly.

The primary paired concentration estimate uses one representative per
certified stream at each endpoint and the already specified equal-prompt,
common-eligibility procedure. It remains descriptive when conditions share
streams. Report the actual representative count and valid count. Do not call
an 11-stream occupancy count pass@8 or distinct@8. Keep the original intact
K8 metrics separately as contextual endpoints.

## Disjoint orientations

For the prespecified matching common stream IDs, sort structurally and split
into lower and upper sets: lower five and upper six in the usual eleven-stream
case. Compare A(lower) with B(upper), then A(upper) with B(lower). Confirm
that the actual selected streams across each pair of conditions are disjoint;
different draw positions are insufficient. A and B must also share the
required policy/checkpoint-specific marginal law within their selected sets.

Each orientation keeps its own promptwise R>=2 eligibility and its own
across-seed fixed-prompt intersection. Do not intersect orientations. The
5-versus-6 budget difference changes variance and eligibility, but not
E[C_hat | R=r]=sum q_c^2 for r>=2 under the fixed-policy iid model. The two
orientations balance the budget direction descriptively. As already noted,
requiring both to be defined introduces a complete-case selection event;
report them individually and treat their mean as a descriptive summary.

If common certified stream IDs differ in size or identity across conditions,
report this before choosing a new split. The rule for non-eleven cardinality
must be frozen (for example lower floor(n/2), upper remaining IDs); never
choose a split to improve eligibility or the effect estimate. Empty or tiny
eligible populations remain explicitly unidentified. Low-success tasks can
have very low joint eligibility at budgets five and six.

Source certification should identify whether stream seeds are also reused
across prompts. Single-policy count-conditioned unbiasedness is a promptwise
identity; asserting a joint conditional-unbiasedness result for a random
eligible-population mean requires the appropriate joint count/label
independence across all selection variables. Descriptive reporting avoids
claiming more than the saved observations identify.

## What remains valid in the original outputs

If each intact K8 group has eight distinct streams from the same marginal
law, averaging its pass@8 and distinct@8 over the four groups still targets
the usual eight-draw expectations. Cross-group correlation does not change
linearity of that average, though it changes uncertainty. The same holds for
per-response mean accuracy. Thus the metadata finding invalidates independent
32-draw cross-group collision pairs, not automatically the existing marginal
K8 metric values. Four correlated groups are not four training seeds.

Keep full32 concentration only as an explicitly labeled reuse diagnostic.
For exact repeated stream outcomes with all outputs correct, 38 of the 496
unordered positions are duplicate-stream pairs. Naive full32 collision then
has expectation C + (1-C)*19/248 rather than C. The formula is a synthetic
illustration; runtime disagreements and validity conditioning mean it is not
a correction to apply mechanically to real data.

Each intact K8 group remains a useful marginal sensitivity. Retain the
original matching prompt-draw analysis and count those groups as correlated,
not independent replications. Compare unique-stream estimates on its same
eligible prompt population when separating budget from selection effects.

## Inference and interpretation

Preserve the previously frozen five-registered-seed descriptive t interval,
partial-seed rules, per-domain reporting, and cross-seed fixed-population
sensitivity. A change of the source sampling unit is an analysis amendment
before effects, not a license to select favorable populations or methods.
Interpret concordance across unique-stream, disjoint-orientation, fixed-
population, and K8 views. Divergence or low eligibility limits the empirical
claim. No estimate proves the literal disappearance or survival of a named
mode.
