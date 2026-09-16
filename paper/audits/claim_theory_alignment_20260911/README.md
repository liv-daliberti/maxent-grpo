# Claim and theory alignment — September 11, 2026

Both active papers are titled **Mode Collapse in RLVR & ModeBench**. This revision uses the current September 11 local manuscript and its frozen results. The earlier pasted audit concerned an older xGRPO draft; historical remote text was not substituted for the current ReplayMaxRL paper.

## Shared scientific claim

Correctness does not determine the distribution of verified alternatives. ModeBench measures both. The categorical analysis identifies a conditional concentration mechanism and a sufficient replay barrier. Controlled training comparisons measure the effect of adding replay; hosted observations establish the relevance of measuring breadth in the evaluated deployments, without identifying their training history as the cause.

Every central claim now has an explicit basis in the shared appendix section **Claims and Their Evidential Basis**. Motivation about downstream agent recovery or adaptation remains an untested hypothesis.

## Mathematical corrections and preserved results

| Component | Final basis and scope |
| --- | --- |
| Success and sampled breadth | New lemma derives per-prompt occupancy identities, fixed-correctness bounds, strict equality cases, majorization monotonicity, and correctness monotonicity. Independent draws, fixed sampling law, and finite/countable keys are explicit. |
| Categorical abstraction | One independent Euclidean logit per canonical outcome is an assumption. Aggregating response aliases does not establish the same update geometry for a language model. |
| Group updates | Retained the correct Dr.GRPO coefficient and practical MaxRL coefficient/potential of order G−1. Standard GRPO requires common/equal length normalization for the scalar-direction equivalence. Advantages are detached; the result is at the on-policy point. |
| Collapse | Retained the full proof, adding finite-time existence and explicit independent-logit geometry, finite initial logits, no competing regularizer, and a unique initial winner. The theorem is asymptotic, not proof that a finite-sample absence is extinction. |
| Gradient availability | Distinguished the probability of a mixed-reward group from the probability or magnitude of a nonzero gradient. Preserved the bounded-score lemma and its separation from exact entropy gradients. |
| Replay retention | Retained the fixed-bank potential proof and uniform-in-time probability floor under fixed positive relative replay weighting. Added schedule scope: a common nonnegative rate preserves retention, while convergence requires infinite cumulative optimization time. |
| Uniformity and partial banks | Full-coverage uniformity remains conditional. Fixed positive weights lead to their weighted target; partial-bank limits can extinguish unbanked correct modes. The capacity-16 bank cannot cover many Python supports. |
| Actual replay loss | Uniform coefficients over mean-token exemplar scores do not imply uniform key probabilities. Unequal categorical exemplar lengths induce inverse-length target weights; aliases and neural geometry add further differences. |
| Complete-response bridge | Retained the joint-potential lemma, requiring complete-response likelihood under the same prompt and decoding law. Its guarantees concern those training prompts. Held-out transfer and practical eight-draw visibility are separate empirical questions. |
| Discrete optimization | Added the precise conditional extension under summable upward energy drift. The implementation has not established this premise; small step sizes or positive replay coefficients alone do not suffice. |
| Entropy | Retained exact entropy/replay identities and probability-space optima; distinguished finite-logit boundary limits and sampled semantic scores. No claim that entropy is inherently incapable of retention or that a separate balance loss is necessary. |

The new measurement lemma and the entire mathematical appendix are shared verbatim between the two papers. Independent primary-source checks confirmed the GRPO inverse-length distinction and the practical MaxRL G−1 result; see the existing cached primary sources in `../reference_audit_20260905/modern_rl_sources/`.

## Interpretive corrections

- Replaced “correctness-adjusted breadth” with extra verified modes beyond the first success; this decomposition still depends on correctness.
- Corrected the baseline-precheck caption to identify average extra modes at the tested scales, rather than a universal decrease in raw breadth.
- Kept the 74 admissible core replay pairs, partial larger-model cohorts, nominal intervals, and all current exclusions. The current tables already satisfy the metric inequalities; obsolete impossible Falcon entries were not present.
- Identified the first figure as an illustration that changes both the fresh objective and replay; the factorial comparison separates them.
- Corrected the Level-2 scope. Its histograms match native training sets and designated construction reserves, while several native Level-1 terminal test populations differ. Figure 6A is development admission. Within-level replay contrasts remain controlled; cross-level changes do not isolate difficulty at fixed test support.
- Retained fixed-bank score deterioration and the absence of a matched no-replay fixed-bank arm. Teacher-forced scores do not measure exact canonical-key sampling probabilities.
- Classified hosted deployment comparisons and temperature sweeps as exploratory observations of their stated protocols. They do not identify RLVR as the cause of concentration, a model-size effect, a universal temperature optimum, or downstream utility.

The independent [interpretive audit](interpretive_review.md) records the construction-reserve and terminal-population sources. Its temporary-build status describes an intermediate review; final validation is recorded separately.

## Verification and reproducibility

The [independent measurement script](verify_success_breadth_independent.py) uses exact rational enumeration of output tuples, independent of the occupancy expression being checked. It covers 912 distributions, 513 strict-bound/equality checks, 15,376 majorization comparisons, and 684 correctness monotonicity checks. Boundary cases include P=0, P=1, K=1, one valid key, and zero coordinates. The countable-support extension is justified analytically. See its [report](success_breadth_independent_validation.md) and [machine-readable results](success_breadth_independent_validation.json).

The current frozen statistical artifacts were generated with Python 3.11. Python 3.10 reproduces the measurements and rendered tables but changes 25 Student-t endpoints by at most 2.22e-16 because of `statistics.stdev` rounding. Verification uses the existing Python 3.11.7 environment and retains exact comparisons; no measurement or frozen confidence interval was altered to make checks pass.

```bash
env PATH=/usr/local/anaconda3/2024.02/bin:$PATH make -C paper
env PATH=/usr/local/anaconda3/2024.02/bin:$PATH make -C paper/mathai2026 bundle
python paper/audits/claim_theory_alignment_20260911/verify_success_breadth_independent.py
```

`before/` and `before_sha256.json` preserve the pre-revision active sources and PDFs. Workshop synchronization records its prior bindings and replacements separately. Historical experiments, generated numerical inputs, and figure assets remain bound to their existing source records.

Final validation passed: the ICLR scientific main is 8 pages with all 8 main figures (62 pages including references and appendix); the workshop main is 4 pages with 7 main and 11 supplementary figures. Both final builds have resolved references and no overfull boxes. The ICLR prose layout check, frozen-evidence and prompt checks, workshop source-bundle verification, and independent mathematical checks passed. See [final validation](final_validation.json), [ICLR build log](iclr_build.log), [workshop build log](workshop_build.log), and the [source patch](changes.patch).
