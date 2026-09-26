# Reusing probability and optimizer theory already in the bibliography

Audit date: 2026-09-21. Primary sources were checked online; the September 5 companion note and its optimizer, discovery, and survival fragments were read. No manuscript changes were made.

The best approach is to cite established mathematical tools, state their replay consequences as corollaries, and reserve the main contribution for the interaction between a finite verified bank, the fresh update, and the executed-key metric. The current Appendix P contains deterministic energy retention, a complete-exemplar bridge, fixed-checkpoint visibility formulas, and the newly added local optimizer-response condition. It does not yet use the sharp certificates or stochastic energy extension already developed in the September 5 note. The relevant bibliography entries exist but are not cited in the current main.tex.

## 1. Highest priority: replace the crude floor by standard KL data processing

**Existing entry:** `vanerven2014renyi`.

**Exact source:** van Erven and Harremoës, [Theorem 9, Eq. (22)](https://arxiv.org/html/1206.2459v2#S3.SS1), data processing, and [Theorem 31](https://arxiv.org/html/1206.2459v2#S4.SS1), Pinsker. At order one, binary coarsening gives

\[
\operatorname{kl}(w_b,p_b)\le D_{\rm KL}(\bar w\|p)
=R_w(p)-H(w)\le D.
\]

Thus `p_b >= ell_-(w_b,D)`, where `ell_-` is the lower root of binary KL equal to D. Also `p(B)>=exp(-D)` and `p_b>=max(0,w_b-sqrt(D/2))`. These are cited corollaries, not new information inequalities. The source's variation norm is the full L1 norm; observe the factor of two when using total variation conventions.

**Manuscript mapping:** sharpen `thm:replay-retention` and `lem:shared-exemplar-retention`. For complete responses, normalize coefficients `a_j=alpha_j/L_j` into `w_j=a_j/sum a_j`, then coarsen by executed key. For one prompt, `kl(W_b,p(key b))<=D`, with `W_b` the total target weight mapped to key b. Different prompts require separate normalization or an explicit joint law.

**What remains ours:** this response-to-key application and the energy budget supplying D; no new proof of data processing or Pinsker is needed. Keep only a short sharpness construction if claiming the best possible certificate.

**Ready local material:** `theory_literature_extensions_20260905/survival_body.tex`, first and third subsections. Its illustrative uniform-16 example gives a .0339712 floor at excess cross entropy .01, versus 4.62e-20 from the current termwise bound. This is illustrative, not a measured neural loss.

## 2. Stronger optimizer scope: cite stochastic descent plus a maximal inequality

**Existing entries:** `wang2017sqn`, `robbins1971almost`, `howard2020timeuniform`.

**Exact optimizer source:** Wang et al., [Section 2, AS.1–AS.4, Lemma 2.1 and Eq. (2.13)](https://optimization-online.org/wp-content/uploads/2016/07/5529.pdf), PDF pp. 4–6. They assume a lower-bounded globally smooth objective, independent unbiased finite-variance gradient estimates, bounded positive-definite preconditioners, and—crucially—H_k depending only on previous noise (AS.4). Their conditional descent inequality is

\[
E_k F_{k+1}\le F_k-
(\eta_k\underline\kappa-L\eta_k^2\bar\kappa^2/2)\|\nabla F_k\|^2
+L\eta_k^2\bar\kappa^2\sigma^2/(2m_k).
\]

**Do not reprove:** the smooth variable-metric stochastic-descent machinery, or its standard supermartingale convergence argument. Their Theorem 2.1 gives liminf stationarity; Theorem 2.2 needs the additional second-moment condition (2.3) for gradient convergence. Neither is an AdamW theorem.

**Exact probability source:** [Howard et al. (2020), Lemma 1, Eq. (2.11)](https://arxiv.org/html/1808.03204#S2.SS3), Ville's inequality for any nonnegative supermartingale. If `D_t=F_t+Jmax>=R_t>=0` and `E_t D_{t+1}<=D_t+d_t` for deterministic summable d_t, then

\[
X_t=D_t+\sum_{s=t}^{N-1}d_s
\]

is a nonnegative supermartingale. Directly applying Lemma 1 gives, simultaneously for all t<=N and all exemplars j, with probability at least 1-delta,

\[
\pi_{\theta_t}(e_j|x_j)\ge
\exp\{-L_j[D_0+\sum_{s<N}d_s]/(\alpha_j\delta)\}.
\]

No union factor over bank size is required because the same energy controls all entries. This extends exact pathwise descent to controlled stochastic energy excursions.

**Robbins–Siegmund attribution:** the 1971 chapter is the classical almost-supermartingale convergence source. Its original publisher/reprint full text remained inaccessible in this audit, so I cannot certify an original theorem number from primary text. Do not invent one. A fully checked alternative for the needed special case is [Durrett, fifth edition, Exercise 4.3.3](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf), printed p.233: nonnegative adapted integrable X,Y, conditional drift `E_n X_{n+1}<=X_n+Y_n`, and a.s. summable Y imply finite a.s. convergence. Wang's Proposition 2.1 also explicitly states nonnegative-supermartingale convergence. The finite-confidence result above needs Ville, not just Robbins–Siegmund convergence.

**What remains ours:** deriving the replay-specific energy, accounting for bias/switches, and converting its bound into complete-response/key probabilities. The September 5 `optimizer.tex` already supplies the full conditional result; use it as a corollary with a short assumption-matching paragraph, rather than another general optimization theorem.

**Actual implementation limit:** current-gradient Adam moments violate AS.4; clipping, momentum, cyclic bank selection and changing banks need explicit error budgets. The new local neural proposition is a Taylor inequality with a score-Gram interpretation; it can be presented as a diagnostic consequence of smoothness, rather than an additional foundational theorem. Neither citation proves the needed margins for the released runs.

## 3. Directly reuse confidence sequences for a retention measurement protocol

**Existing entries:** `howard2021confidence`, alongside the already used `clopper1934binomial`.

**Exact source:** [Howard et al. (2021), Proposition 7, Eqs. (56)–(57), final version](https://arxiv.org/html/1810.08240#A1.SS3): the two-sided beta-binomial mixture boundary under the stated two-sided sub-Bernoulli conditions, with rho>gh. Proposition 8 is its one-sided counterpart. Section 4 describes inversion for bounded-mean inference using `g(mu)=mu-a`, `h(mu)=b-mu`, `V_t(mu)=g(mu)h(mu)t`.

**Manuscript mapping:** supplement the existing fixed-N binomial paragraph with an anytime-valid, named-key visibility certificate at a frozen checkpoint. For Bernoulli key indicators, take a=0,b=1; invert the published boundary and allocate failure probabilities across a prespecified roster. This permits choosing the evaluation stopping time after seeing samples.

**Do not reprove:** beta-mixture martingales or Ville. **What remains ours:** the sampling protocol, key identity, multiple-key error allocation, and practical probability threshold chosen for retention.

The sampling law must have a fixed mean (iid checkpoint draws suffice). These intervals do not certify a changing policy's current probabilities by pooling training history. Teacher-forced scores are not Bernoulli trials.

**Version warning:** v3 called the two-sided result Proposition 6 and the one-sided result Proposition 7. The published/current version uses 7 and 8. The September 5 note's citation to Proposition 7 is correct if tied to the final version.

## 4. Reuse coupon collection for fixed-policy coverage; use adaptive hazards for training

**Existing entries:** `anceaume2015coupon`, `durrett2019probability`.

**Exact fixed-law source:** [Anceaume et al., Theorem 2](https://arxiv.org/html/1504.03878v1#S2) shows uniform iid coupon probabilities minimize the time to collect c distinct types in stochastic order. [Theorem 3](https://arxiv.org/html/1504.03878v1#S3) allows a null coupon and fixes total non-null probability: equal non-null probabilities minimize collection time for every c. Map incorrect responses to the null coupon and correctness to the non-null mass.

**Manuscript mapping:** this directly supports the statement that, at a fixed correctness level and finite known mode support, a uniform allocation optimizes the entire distribution of distinct-mode collection time—not just expected distinct@K. Cite it rather than deriving a fresh coupon-collection theorem.

**Limit:** a frozen policy with iid draws is essential. Capacity rejection and changing training policies are outside the coupon theorem. Uniform sampling of *stored exemplars* is not uniform policy sampling of keys.

For adaptive training, [Durrett, Theorem 4.3.4](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf), printed pp.225–226, is conditional Borel–Cantelli. Apply its standard hazard logic to actual admission events, not merely generation. If each unbanked key retains conditional admission hazard >=h_b>0, then the finite-horizon nonadmission probability is <=exp(-Nh_b); finite support plus sufficient capacity yields eventual admission. This is a short corollary/conditioning calculation. The theorem does not establish the hazard floor.

**What remains ours:** defining insertion opportunities correctly, proving an admission floor, and accounting for bank renormalization and replay persistence. `discovery.tex` already makes these conditions explicit. Do not promote generic coupon bounds as a new RL theorem.

## 5. Missing mass is reusable, but addresses a different question

**Existing entries:** `mcallester2003missing`, `berend2013missing`.

For iid samples from one fixed categorical law, missing mass is
`U_N=sum_b p_b 1{b never observed}`. [McAllester–Ortiz, Theorem 16](https://jmlr.org/papers/volume4/mcallester03a/mcallester03a.pdf), printed p.906, supplies the upper-tail exponent `N epsilon^2`; Theorem 10, p.904, gives the lower-tail exponent `e N epsilon^2/2` in their S notation. [Berend–Kontorovich, Theorems 1–2](https://arxiv.org/html/1210.3248v1#S2) give directly

\[
\Pr(U_N>EU_N+\epsilon)\le e^{-N\epsilon^2},\qquad
\Pr(U_N<EU_N-\epsilon)\le e^{-C_0N\epsilon^2/4},
\]

with C0 approximately 7.6821.

**Use:** cite these for aggregate unseen probability at a frozen checkpoint when the catalogue is unknown. **Do not claim:** a bound on the number of unseen modes, survival of every named bank key, or concentration about an observed estimator merely from these formulas. EU_N is unknown; concentration around it alone does not produce a data-dependent certificate. Changing-policy histories also violate the fixed-law premise.

This is lower priority than the first four mappings. It belongs in measurement limitations or a separately designed coverage study, not in the main replay-retention theorem.

## Recommended integration priority

1. Add the sharp KL certificate as a cited corollary immediately after deterministic retention. Keep complete-response normalization explicit.
2. If expanding optimizer scope, adapt the existing stochastic-energy fragment and cite Wang plus Ville directly; leave AdamW applicability conditional.
3. Cite Howard's published confidence sequence for a concrete retention-evaluation protocol; avoid reproducing its proof.
4. Use Anceaume for a concise fixed-correctness coverage consequence. Add adaptive discovery only if the paper states and tests an admission-hazard premise.
5. Keep missing mass separate from named-key retention.

The paper-specific work remains the estimator/replay interaction, exact finite-time concentration threshold, bank/execution-key construction, and evidence about neural optimizer response. More standalone proofs of Pinsker, Ville, coupon collection, or standard Taylor descent would lengthen Appendix P without increasing that contribution.
