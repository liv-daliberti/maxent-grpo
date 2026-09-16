# Saved-output conditional concentration: retrospective analysis protocol

This protocol is recorded before calculating conditional-collision results for this follow-up. The underlying experiments and their correctness/breadth outcomes have already been inspected. This is a retrospective secondary analysis, not a new preregistered experiment. No training, hosted calls, new sampling, or outcome-based checkpoint selection is authorized by this analysis plan.

## Questions and population

1. Does verifier-only training concentrate verified outcomes? Compare each arm's own initial checkpoint (step0) with its registered terminal checkpoint (step3072), separately for Dr.GRPO, GRPO, and binary MaxRL, at all three Level-1 models and all five domains.
2. Does canonical replay reduce terminal concentration? Compare ReplayDr.GRPO with Dr.GRPO and ReplayMaxRL with MaxRL within model, domain, level and training seed.
3. Supporting checks: before/after changes for replay arms; the change difference `(replay final-replay initial)-(control final-control initial)` on a single four-condition eligible population; within-Level-2 results. No between-level causal contrast is estimated.

Use the exact sources admitted by `paper/results/training_curve_snapshot_20260911.json` and `paper/results/baseline_collapse_precheck.json`, including their frozen source/prefix hashes and endpoint origins. Preserve original terminal cohort membership: 74 Dr replay pairs and67 MaxRL replay pairs at Level1, plus the separate registered baseline population. Sample-level conflicts may make an endpoint unavailable for this new analysis; they do not silently redefine the published cohort. Missing or ambiguous responses are neither imputed nor replaced by newer/better checkpoints. Every such exclusion is reported with its source.

## Verified outcomes and primary measurements

A sample contributes a verified key only when its persisted reward is positive and its canonical key is present. Invalid responses can have non-null raw keys; those keys must not enter correct-mode counts. A positive reward with no key is an integrity error. Preserve failed responses and all fixed draw groups in the denominators for pass@8, distinct@8 and mean@8.

For fixed prompt and policy, write q for the distribution over keys conditional on correctness. Define `C(q)=sum_c q_c^2`, the probability that two independent correct outputs share a key. Larger C means greater order-two concentration; lower C need not establish majorization or preservation of every mode. With R≥2 correct observations and key counts n_c,

`C_hat=sum_c n_c(n_c-1)/(R(R-1))`.

Under iid draws from one fixed law, `E[C_hat | R]=C(q)`. R<2 makes the statistic undefined; zero sightings of alternatives do not establish a one-mode true support. Also report pair disagreement `1-C_hat`; do not invert C_hat to claim effective support.

Reconstruct and verify the original per-draw primary metrics against frozen endpoint summaries. The four fixed K8 groups remain four groups: averaging their pass@8/distinct@8 is permitted; silently changing the primary metrics to pass@32/distinct@32 is not.

## Paired estimands and eligibility

### Full saved-sample estimate (32 responses per condition)

Pool canonical keys over the four recorded eight-draw groups only when the same prompt identity and sampling-law metadata agree. For each training seed and paired contrast, use the common eligible prompt set E_s: every required condition has all four complete groups and at least two correct responses. Average promptwise differences with equal prompt weights, then average seed estimates with equal seed weights. Report E_s, its fraction of the full prompt cohort, A-only/B-only/both/neither eligibility counts, and both conditions' correctness and raw breadth on the same set. Also report their correctness/breadth on the full cohort.

These are paired descriptive concentration estimates for an eligibility-conditioned population. The single-policy unbiasedness identity does not automatically imply unbiasedness after conditioning on BOTH policies' correct counts: shared evaluation randomness can couple a policy's labels to the other policy's counts. Do not claim that shared-seed paired eligibility is independent of labels without proof.

Provide an all-admitted-seed common-prompt intersection E* sensitivity for each block. Keep the source-admitted seed set fixed when constructing it. If an admitted seed or E* is unavailable, mark this sensitivity unavailable rather than silently dropping that seed. This fixes prompt identity across seeds but remains an observed eligible population, not the full benchmark.

### Disjoint-draw sensitivity (16 responses per condition)

To probe shared-stream selection, define two fixed orientations before analysis:

- orientation0: conditionA draws0,1 versus conditionB draws2,3;
- orientation1: conditionA draws2,3 versus conditionB draws0,1.

Verify actual recorded per-prompt sampling streams are disjoint between conditions within each orientation. Distinct draw indices alone do not prove this. The iid interpretation remains conditional on the evaluation generator and fixed decoding law.

Construct eligibility separately within each orientation. Do not require both orientations' count eligibility when estimating either orientation: that would condition again on the coupled streams. Report both orientation means, coverage and sampling-seed audit; their average is a single descriptive training-seed estimate only when both exist. Do not treat orientations as independent training replicates. An across-seed common-prompt sensitivity is likewise defined separately for each orientation. If streams cannot be established as disjoint, label the calculation an index-split sensitivity rather than an independent-stream estimate.

Interpret concentration claims using both the full-sample and disjoint-draw evidence and their populations. Never select the orientation, threshold, domain or statistic with the favorable sign.

### Matched K8 sensitivity

For each prompt, compute collision separately in matched draw indices that are eligible in both conditions. Average those paired draw differences within prompt and then equally over eligible prompts. Report the eligible draw counts and prompt coverage. Also calculate the pooled32 estimate on exactly that eligible prompt set, so sample-budget changes and prompt-selection changes are distinguishable. Draw-level outcomes are not independent training replicates.

### Four-condition change difference

Use a single within-seed prompt intersection with at least two correct outputs in every initial/final × control/replay observation. Evaluate all four C values on that same population. This is supporting descriptive evidence, with the same shared-stream qualification, not an automatic causal mediation or individual-mode survival result.

## Aggregation and uncertainty

Report every model/domain/method block, including undefined and unfavorable results. Domain-level paired training seeds are the inference units. For a complete five-seed block with every seed estimate available, report the mean and a nominal paired Student-t95% interval using df4. For fewer than five estimates, report the exact n and descriptive mean/range, without a complete-block interval. Seed dispersion describes variability of the measured, eligibility-conditioned effects over these runs; it is not a simultaneous confidence guarantee for every prompt or the unseen support. Zero empirical variance and all observed C=1 do not establish equivalence or literal extinction.

Keep the correct-pair-pooled estimator `sum(colliding pairs)/sum(correct pairs)` separate. It weights prompts by successful-pair counts and can move because correctness weights change. It is included to relate to the existing hosted analysis, not substituted for the equal-prompt paired estimator.

Any aggregate across domains must first average a fixed domain vector within the same seed, using every designated domain and no omission of undefined cells. If that is unavailable, omit the aggregate or label an explicitly different descriptive target; do not pool all observed cells and present a benchmark-wide effect. No hypothesis decision based on sign counting across dependent model/domain cells, and no multiple-comparison-adjusted confirmatory claims are planned.

## Theory and empirical interpretation

In the specified ideal categorical mean flow, after the positive time change, `dC/dtau=2[sum(q^3)-(sum(q^2))^2]=2 Var_{c~q}(q_c) >=0`. For K2 the occupancy identity gives `distinct@2=2P-P^2 C(q)`. These link the conditional statistic to existing theory; they do not certify neural dynamics.

Evidence of increased C supports greater conditional concentration in the stated observable population. It does not alone prove support extinction, mode-by-mode survival, majorization, or a universal property of all domains. Negative replay-minus-control C supports reduced measured concentration in that contrast. Positive correctness/raw-breadth gains without lower C remain success-and-breadth gains. Hosted concentration remains descriptive of those deployments and protocols; these training comparisons do not identify hosted training histories.

## Deliverables and acceptance

- Reusable source loaders and analysis code with meaningful mathematical/integrity tests.
- Frozen compact per-prompt measurements and paired seed records sufficient to reproduce summaries, source hashes, original cohort counts, missing/conflict ledger, and all eligibility/stream audits.
- Machine-readable full results, readable summary tables, and standalone publication-quality figures. Do not overwrite earlier numerical artifacts.
- A clear assessment of which parts of “collapse occurs, ModeBench reveals it, canonical replay helps” the new analysis supports, with domains and limitations identified.
- Update the narrative plan around the resulting evidence; any manuscript integration must preserve existing validated primary results and be rebuilt/checked if performed.
