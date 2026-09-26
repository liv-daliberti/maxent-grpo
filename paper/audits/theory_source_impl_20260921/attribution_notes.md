# Attribution and integration notes

The two LaTeX fragments are additions only. `setpo_bridge.tex` belongs after the neural binary subsection; `coupon_coverage.tex` can follow the existing finite-support coverage lemma or the discussion of the full-coverage replay target. Both use existing bibliography keys and state imported results as attributed corollaries. They deliberately do not reproduce the source proofs.

## Checked primary versions

- SetPO: https://arxiv.org/html/2602.01062v1, Assumption 4.1 and Theorem 4.2. Its bounded symmetric equality kernel works on the correct-conditional **response law**; pushforward probabilities are the verified-mode vector q. No tabular update assumption enters this identity.
- Coupon result: https://arxiv.org/html/1504.03878v1, Theorem 3. The source assumes positive coupon masses summing to less than one. Our finite-budget boundary cases follow by continuity. The theorem concerns arbitrary r distinct coupons, not a prespecified subset. It is stronger than merely maximizing expected distinct count. Do not repeat the source's nearby Schur-convex wording for the collection CDF: the tail direction stated in our fragment is the operative result.
- AVSPO: https://arxiv.org/html/2605.21125v2, Lemma B.1 and Theorem B.4, Equations (25)–(26).
- DPH-RL: https://arxiv.org/html/2509.07430v4, Appendix D.1 (2026-03-03 version). Its v1 appendix numbering differs.
- Cui et al.: https://arxiv.org/html/2505.22617v1, Lemma 1, Proposition 1, Theorems 1–2.

## Suggested exact attribution sentences

Before the conditional-score proof:

> The conditional-score identity is given by \citet[Lemma~B.1, v2]{avspo2026}, who apply it to homogeneous reward groups in their Theorem~B.4. Here we condition on every reward vector and also compute the estimator covariance.

At the replay/forward-KL identity:

> Forward-KL minimization as likelihood rehearsal is explicit in \citet[App.~D.1, v4]{li2025divergence}; verified replay replaces the reference target with the bank's distribution over discovered successes.

At the geometry comparison:

> The dependence on update geometry also appears in \citet[Proposition~1 and Theorems~1--2, v1]{cui2025entropy}, whose tabular entropy identities distinguish vanilla from natural policy gradients. Our conditional-mode conclusions concern a different statistic.

Keep the DPH improvement-theorem concern and AVSPO fixed-step-noise concern in the audit memo only; they are not needed in publication prose. Credit Harper for the replicator variance identity rather than duplicating SetPO's functional route to that same tabular result.
