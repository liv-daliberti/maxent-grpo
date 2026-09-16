# Amendment 1: overlapping generation streams

Recorded after source/runtime inspection and before calculating the empirical concentration contrasts. It supersedes the original protocol's assumptions that pooling four K8 groups yields32 distinct sampling streams and that draw halves0/1 versus2/3 are disjoint. The original protocol remains preserved.

## Source finding

The recorded parent seed for draw d is base+d. The training runtime's vLLM0.8.4 V0 parallel sampler assigns child i the seed parent+i. Consequently the four n=8 groups for a prompt nominally use base+[0..7],+[1..8],+[2..9],+[3..10]:11 distinct child-stream identifiers rather than32. The same parent seeds are reused across initial/final checkpoints and treatment conditions. Canonical-action Pantry evaluation uses the same n=8 sampler; masking and response decoding do not remove the fork.

Source anchors: learner/run.py draw-seed construction; actor.py:584–613 n=8SamplingParams and :645–653 neutral request construction; vLLM sequence.py:1428–1434 child fork and llm_engine.py call. A separate source certificate binds exact files and runtime applicability. Recorded options/request seeds take precedence over assuming every request was neutral. Sources without a justified mapping remain explicitly unsupported for stream-based analyses.

Saved responses assigned the same nominal child seed are often identical but not always, consistent with batch/runtime effects. Repeated instances must not be counted as independently seeded observations, and disagreement must not be used to select or exclude outputs.

## Revised concentration views

1. **Primary distinct-stream descriptive estimate.** For each prompt and endpoint, assign a nominal child-stream identifier to every saved position using the certified runtime mapping. Choose exactly one representative per identifier: the earliest `(draw_index, output_index)` occurrence, regardless of its reward or key. Compute collision from these verified keys. Typical neutral records have11representatives. Report their actual count and their own number of correct outputs. Paired common eligibility requires at least two correct representatives in both conditions. Equal-prompt and seed aggregation, full cohort reporting, fixed-across-seed prompt sensitivity, and nominal interval rules remain as in the original protocol. Shared streams across conditions mean this paired selected-population contrast remains descriptive.
2. **Disjoint-stream orientations.** For each paired prompt, take the sorted intersection of available nominal stream identifiers across its conditions. Split this outcome-independent set into lower `floor(n/2)` and upper remaining identifiers; typical records yield5and6. Orientation0 compares A lower with B upper; orientation1 reverses them. Use the same deterministic per-endpoint representative mapping. Certify that the selected identifiers are disjoint across conditions in each orientation. Construct eligibility separately per orientation, and retain both results/coverage. Their complete-case mean is descriptive, not a general unbiasedness theorem after conditioning on both folds. Different5/6 sample budgets change precision and eligibility, not the single-policy collision identity under iid marginal sampling. Any generator-independence statement remains conditional on the certified sampler and its marginal-law assumptions.
3. **Intact K8 sensitivities.** Retain each original eight-sample group, which contains eight distinct child seeds under the audited mapping. Compute paired collision by draw and equal-prompt means over common eligible draws. Treat the four groups as correlated, not independent replicates. Also retain the first original group as an explicitly fixed single-group view. Compare with the primary distinct-stream statistic on the same eligible prompt population when making budget comparisons.
4. **Naive32 diagnostic only.** The collision of the raw concatenation may be calculated solely to quantify the effect of counting repeated streams. Label it `naive_reused_streams_32`, exclude it from concentration claims and principal effect figures, and never invoke its iid-unbiasedness formula for these records.
5. **Four-condition change difference.** Use a common four-condition eligibility set and each endpoint's distinct-stream representatives. This remains supporting descriptive evidence. Do not call it individual-mode survival or a mediation analysis.

Per-policy stream counts, repeated occurrences, and verified-key agreement/disagreement are provenance diagnostics. No disagreement or unfavorable collision value changes the representative rule or endpoint admission.

## What is unchanged

Original pass@8, distinct@8 and mean@8 remain averages over the four intact groups and are reconstructed from every saved failure/success. Group overlap changes their dependence, not each marginal expectation. Report these original metrics separately from selected-stream correct counts and fractions. Do not attach an @8 label to11-,5-,or6-stream sampling.

The single-policy identity `E[C_hat|R]=sum q²`, the shared-RNG selection warning, exact seed cohorts, undefined eligibility handling, partial-block policy, and outcome-independent reporting of every domain remain unchanged. The analysis establishes neither exact support extinction nor iid independence merely by observing distinct seed integers.
