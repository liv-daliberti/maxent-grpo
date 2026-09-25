# Extended comparison: final appendix pass

Reviewed Appendix R, from `\section{Diversity-Preserving RLVR: Extended Comparison}` to the next `\clearpage` before Part VIII. `before.tex` is the exact reviewed live block; `replacement.tex` is ready to integrate. No live manuscript, implementation, numerical result, bibliography, or figure was edited.

The replacement preserves the section label, all four original citation keys, the original `+.318` effect and `[+.272,+.358]` interval, the UCPO equation locators, and the measurement macros. It is approximately the same length as the original. This section contains no figures or numbered equations.

`validation.json` records **82 passing checks**, source hashes, resolved references/citations, original-result preservation, and corroborating finite algebra examples. No empirical evaluation, bootstrap, training, or model call was run. Root retains whole-PDF build/layout responsibility.

Replacement SHA-256: `e061510740dc19fe173a4608d1475400b5b4c38408e924d45aac043b9926995d`.

## Material repairs

1. **Removed novelty/history rhetoric.** The opening no longer discusses what “is not our discovery,” which claims prior work covers, or what “we measured rather than cited.” Headings now identify measurement, training signals, targets, evaluated comparators, and scope. The final text presents the scientific comparisons directly.

2. **Corrected DMPO's actual target and loss.** The primary paper's Eq. (5) is a group Boltzmann target proportional to `exp(reward/alpha)`, not raw reward. At positive temperature, binary failures retain positive target mass. Correct rows have equal target weight, so conditioning on correct rows yields canonical-key mass proportional to row multiplicity. Eqs. (6)–(9) use normalized mean-token likelihoods and an MSE distribution-matching penalty added to GRPO. The replacement distinguishes that implementation from exact forward-KL descent. The original “uniform over correct sampled trajectories” statement incorrectly discarded the failure mass.

3. **Removed the false equivalence between DMPO and the bank-frequency ablation.** DMPO's target uses the current sampled group; the replay ablation reweights stored key exemplars with cumulative fresh successful counts. They use different targets, losses, histories, and update procedures. The replacement explicitly says the bank-weighting ablation does not evaluate DMPO.

4. **Corrected UCPO's weights.** `src/oat_drgrpo/learner/grpo.py` sums rollout-policy token log probabilities over response masks. `src/oat_drgrpo/ucpo.py` then normalizes those sequence likelihoods over active successful rows and applies self-normalized inverse weights. The old text called this estimated “mode frequency,” falsely implying canonical-key counting. The mass-conserving formula and bound remain unchanged, with the positive-row and bounded-base-advantage conditions explicit.

5. **Removed the claim that an unsampled mode cannot change.** A stored exemplar absent from a group receives no direct replay-likelihood term from the compared fresh-update methods. Its probability can still move through normalization and parameter sharing. Even with zero logit component for an absent category, `pdot_b=-p_b sum_a p_a g_a` can be nonzero. The finite example in `validate.py` confirms this distinction.

6. **Restricted the bounded-score lemma to its assumptions.** The lemma applies to the specified categorical on-policy score component with a probability-independent finite bound. It cannot automatically certify all token-level mechanisms, reference-KL terms, or AdamW/PPO updates. Exact entropy remains outside its bounded-surprisal premise. The comparison does not establish inevitable collapse or neural replay retention.

7. **Separated response targets, key targets, full coverage, and length normalization.** Uniformity over correct response strings is not generally uniformity over execution keys. A partial bank lacks undiscovered alternatives, and the actual per-token replay loss introduces length dependence. The shared uniform target of exact conditional entropy and full-coverage uniform categorical replay is stated only on the same correct-category set.

8. **Removed an incorrect imported full-output-entropy claim.** UCPO v1 Theorem 5.2 states that binary reward plus full-output entropy has an exactly uniform-correct optimum for sufficiently small positive entropy weight. With incorrect categories present, the actual finite-temperature Gibbs optimum assigns them positive mass. The old paragraph endorsed that claim. The replacement attributes UCPO's binary-reward indifference and forward conditional-KL objective, both supported by its Sections 4.1 and 6.2, without importing the conflicting claim. This is consistent with the manuscript's own distinction between conditional and full-output entropy in Appendix Q. No local theorem was changed.

9. **Corrected the accuracy interpretation of the replay-weighting result.** The stored five-domain Qwen2.5-0.5B result is `Delta B8=0.3178125`, interval `[0.272265625,0.3578125]`, with `Delta pass8=0.00984375`, interval `[-0.119375,0.1390625]`. The latter establishes neither accuracy equivalence nor an accuracy gain. Bank construction and the update rule are held fixed, but actual bank contents can diverge under the two policies. Extra-mode counts remain coupled to correctness. The original assertion that “reuse is not the active ingredient” was not justified and is removed.

10. **Corrected UCPO's reported diversity effect.** On Qwen2.5-0.5B Countdown, the common-prompt `Delta PCMD=0.0024830623` has a paired Student-t interval `[-0.0184258263,0.0233919509]`, so the text no longer claims a demonstrated diversity increase. Its `Delta pass8=0.133203125`, interval `[0.0047355761,0.2616706739]`, supports the stated correctness gain. The replacement specifies five-seed nominal intervals and the common-prompt eligibility rule (at least two verified responses in both arms; at least 30 common eligible prompts per seed).

11. **Corrected what was implemented and compared.** Fixed Semantic MaxEnt is not Clip-Cov or KL-Cov. Neither those token-level methods nor DMPO has a direct run in this evaluation. UCPO and sparse RLEP-Dr are evaluated on the two smaller scales. RLEP-Dr is an adaptation with an offline pool and a mixed fresh/replay advantage, rather than the same online bank procedure minus key balancing. Its MathIR correctness gain remains evidence compatible with useful reuse without uniform key weights. The replacement also points to the separate GAPO/SetPO comparisons, preserving awareness of the latest work.

12. **Clarified the metric/population distinction.** The population PCMD depends on conditional key allocation, not correctness at fixed allocation. Finite-sample eligibility and changes in eligible prompt populations can still alter an aggregate. Token entropy and UCPO's equation-uniqueness score measure different objects; the latter is a fraction of unique formulas, not a count of solution modes. DMPO's quality ratio includes zero-valued invalid outputs and measures solution quality alongside feasibility, not direct conditional key diversity. The original universal statement that suppressing errors necessarily lowers entropy was removed.

## Primary sources reviewed

The full primary sources were already downloaded locally; their contents were read directly. No secondary account was used for the corrected mechanisms.

- **Cui et al., arXiv:2505.22617v1**, `paper/audits/reference_audit_20260905/modern_rl_sources/cui2025entropy_full.txt`: token-entropy definition (Eq. 5), entropy/performance observations and their stated nonuniversality, Clip-Cov gradient detachment on selected positive-covariance tokens, and KL-Cov penalties on high-covariance tokens. [Primary paper](https://arxiv.org/abs/2505.22617v1)
- **Lochab et al., arXiv:2605.00365v1**, `lochab2026ucpo_full.txt`: reward indifference (Section 4.1), conditional-KL target (Section 6.2), practical reweighting (Eqs. 16, 47, 51), and correct-response equation uniqueness (Appendix F). The problematic full-output-entropy optimality statement is in Theorem 5.2; the replacement does not endorse it. [Primary paper](https://arxiv.org/abs/2605.00365v1)
- **Li et al., arXiv:2605.19461v1**, `li2026distribution_full.txt`: Boltzmann target, mean-token group likelihoods, MSE, and combined GRPO objective (Eqs. 5–9); success rate and quality ratio (Section 4.2). [Primary paper](https://arxiv.org/abs/2605.19461v1)
- **Zhang et al., arXiv:2507.07451v1**, `zhang2025rlep_full.txt`: collection of verified trajectories followed by mixed fresh/replayed-response updates. The repository's sparse Dr.GRPO adaptation is identified separately. [Primary paper](https://arxiv.org/abs/2507.07451v1)

All four source paths are rooted in `paper/audits/reference_audit_20260905/modern_rl_sources/`; hashes are in `validation.json`.

## Local evidence

- `src/oat_drgrpo/ucpo.py`, especially `redistribute_ucpo_advantages`.
- `src/oat_drgrpo/learner/grpo.py`, UCPO sequence-log-probability aggregation and caller.
- `src/oat_drgrpo/canonical_replay.py`, `canonical_replay_key_target_weights`.
- `src/oat_drgrpo/rlep.py`, frequency-preserving offline pool and mixed fresh/replay advantages.
- `paper/results/e120_primary_breadth.json`, `rows.five_domain_mean.uniform_minus_frequency`.
- `paper/figures/direct_comparator_endpoint_effects.json`, common-prompt UCPO/RLEP summaries and seed support.
- `paper/results/mode_diversity_coverage.tex`, `MDcells=375` and `MDdistinctcorr=.936`.
- Current Appendix F comparator specification, G supporting-comparator/frequency-weighting discussion, and Appendix Q categorical score-bound/entropy conditions.

The original primary effect and confidence interval are preserved. Additional numbers in the replacement are copied from already existing reported summaries to clarify uncertainty, not new estimates.
