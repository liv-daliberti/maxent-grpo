# Reusing existing optimization theory in Appendix P

Date: 2026-09-21. Research only; no manuscript changes. All five papers are already in `paper/example_paper.bib`. The theorem numbering below was checked against primary sources, with the version stated where necessary.

## Main recommendation

**Use Mei et al. (2020), Theorem 5 and Lemmas 11–13, to replace the stationary-only treatment of the Dr.GRPO + exact reverse-KL comparator with an attributed convergence corollary.** The reduction is exact for the independent-logit, fixed-reference, exact-gradient model. It gives convergence, an initialization-dependent linear rate, and a positive trajectory-wide probability floor. This is a substantive strengthening obtained by applying an existing theorem, rather than creating another bespoke proof.

Agarwal et al. supply an exact citation for the natural-gradient counterexample to geometry-independent collapse. Geist et al. supply the standard regularized-policy optimum. Zhan et al. and Vieillard et al. analyze different update rules; their convergence theorems should not be attached to our Euclidean-logit or AdamW trajectories.

## 1. Mei et al. 2020 — directly reusable for Dr.GRPO + reverse KL

Existing key: `mei2020softmaxpg`.

Primary sources: [ICML/PMLR paper](https://proceedings.mlr.press/v119/mei20b/mei20b.pdf), [supplement with detailed proofs](https://proceedings.mlr.press/v119/mei20b/mei20b-supp.pdf).

**Exact anchors.** Section 4.2.1, Update 2, Eq. (21); **Lemma 11** gives centered-logit contraction; **Lemma 13** gives a uniform positive action-probability floor; **Theorem 5**, Eq. (24), gives a linear rate for exact entropy-regularized softmax gradient ascent in a finite bandit. Its step condition is **eta <= 1/tau**, not merely eta = 1/tau. The learning rate is positive and constant. The assumptions include finite action support, finite initial logits, positive entropy coefficient, exact gradients, and rewards normalized to [0,1]. **Theorem 6** is the general-MDP counterpart and is unnecessary for our isolated-prompt bandit.

### Our exact reduction

Let K be the total number of categories, R_a=1[a is correct], alpha=(G-1)/G, beta>0, and let the fixed reference mu have full support. Define

    J(p) = alpha P(p) - beta KL(p || mu),
    r_tilde,a = alpha R_a + beta log mu_a.

Then

    J(p) = <p, r_tilde> + beta H(p).

Thus **the exact expected Dr.GRPO gradient plus exact reverse-KL gradient is precisely entropy-regularized softmax PG for a different fixed reward vector**. No change of optimization geometry is involved.

To meet the paper's [0,1] reward assumption, set

    B = max{1, max_a r_tilde,a - min_a r_tilde,a},
    r'_a = (r_tilde,a - min_b r_tilde,b)/B,
    tau = beta/B,
    eta' = eta B.

Here J = B[<p,r'> + tau H(p)] + min_a r_tilde,a. Gradient ascent with eta on J is exactly gradient ascent with eta' on the normalized objective. Consequently eta' <= 1/tau is equivalent to **eta <= 1/beta**. The extra max with 1 just handles a constant reward vector and keeps B>0.

### Proposed attributed corollary

*Corollary (Dr.GRPO with exact reverse KL; specialization of Mei et al., 2020, Theorem 5).* Let p_n=softmax(z_n) with finite z_1, let mu_a>0 for every category, and take constant 0<eta<=1/beta. Under the exact update

    z_(n+1) = z_n + eta grad_z[alpha P - beta KL(p || mu)],

we have

    p_n -> p*,   p*_a = mu_a exp(alpha R_a/beta) / Z,
    q* = mu(. | correct),

and min_(n,a) p_n,a >= c>0. In particular, for B above one admissible published constant is

    c = K^(-1) exp[-B/beta
                  -4 sqrt(K)(||z_1||_infinity + B/beta)],

and the transformed Theorem 5 rate is

    J(p*)-J(p_n)
      <= [2K(beta ||z_1||_infinity+B)^2/beta]
         exp[-2 beta eta c (n-1)].

*Three-line proof.* Rewrite J as B times the normalized entropy-regularized bandit objective plus a constant using the displayed r', tau, eta'. Apply Mei et al., Theorem 5 and Lemma 13; their step condition becomes eta<=1/beta. Their optimum softmax(r'/tau) is p*, and J(p*)-J(p)=beta KL(p||p*) identifies the limit and its correctness-conditional allocation.

The explicit floor is conservative and can be numerically tiny. Its dependence on initialization is compatible with arbitrarily slow recovery from an increasingly rare initial mode. The result does **not** imply a uniform recovery time across all initializations, uniform correct-mode allocation, or preservation of pretraining diversity beyond the reference conditional.

### Continuous-time qualification

Theorem 5 is a discrete theorem. For the manuscript's continuous flow, a short adaptation of the same contraction argument is available; do not claim the continuous result follows merely by taking a limit of the theorem's conclusion.

Let Pi=I-11^T/K, xi=Pi(beta z-r_tilde), and H_p=diag(p)-pp^T. The flow gives xi_dot=-beta H_p xi and

    d||xi||^2/dt = -2 beta xi^T H_p xi
                <= -2 beta min_a p_a ||xi||^2.

Nonincrease of ||xi|| bounds centered logits, which yields a positive lower bound on min_a p_a and exponential convergence of xi to zero. This is the differential version of Mei's Lemma 11 argument, not a new optimization mechanism. If the manuscript prefers to avoid another proof entirely, state the discrete corollary and retain the mean-flow stationary algebra separately.

### MaxRL caveat

For G>=3, centered MaxRL has a nonconstant c(P), so its potential Psi(P) is nonlinear in P. There is no fixed reward vector r_tilde with <p,r_tilde>=Psi(P)+beta<p,log mu> for every p. Rescaling time makes the KL coefficient beta/c(P) state dependent; it does not convert the flow to Mei's fixed-temperature algorithm. **Do not apply Theorem 5 to MaxRL or standardized GRPO without a further argument.** The special case G=2 has c_MaxRL=1 and therefore does fit the constant-coefficient reduction. AdamW, PPO clipping, sampled gradients, and shared neural parameters remain outside this corollary.

## 2. Agarwal et al. 2021 — geometry citation and log-barrier foundation

Existing key: `agarwal2021policygradient`. Primary source: [JMLR published paper](https://jmlr.org/papers/volume22/19-736/19-736.pdf).

**Exact anchors.** **Theorem 10** gives asymptotic exact softmax-PG value convergence under its step and state-coverage conditions. **Theorem 12** and **Corollary 13**, Section 5.2, concern a log barrier **KL(Unif_A || pi)** on every action; **Remark 14** distinguishes this from Shannon entropy. **Lemma 15**, Section 5.3, gives the exact natural-policy-gradient multiplicative update; **Theorem 16** gives its value-convergence rate.

**Reusable application.** Cite Lemma 15 for our geometry counterexample: in the bandit, p_a^+ is proportional to p_a exp(eta R_a), so two correct actions receive the same multiplier and their ratio is preserved at every finite step. This is stronger and more directly sourced than merely describing a continuous natural-gradient flow.

**Scope.** Theorem 10 can support correctness convergence, but does not identify the winner among equally rewarded correct modes; our arbitrary-positive-step winner theorem is stronger on that axis. The log-barrier results substantiate the established boundary behavior of forward KL. They do not directly prove convergence of selective verified replay: their regularizer covers all actions, including incorrect ones, whereas our bank generally covers a discovered subset. Citing their theorem does not remove that support mismatch or the complete-response bridge.

## 3. Geist, Scherrer, Pietquin 2019 — directly cite the standard optimum

Existing key: `geist2019regmdp`. Primary source: [PMLR paper](https://proceedings.mlr.press/v97/geist19a/geist19a.pdf).

**Exact anchors.** **Proposition 1(i)** identifies the unique maximizer of a linear value minus a strongly convex regularizer via its convex conjugate. **Proposition 2(iii)** establishes regularized Bellman contraction. **Theorem 1** identifies the unique optimal regularized policy. Their algorithms thereafter are regularized dynamic programming / modified policy iteration.

**Reusable application.** For the one-state fixed objective with Omega(p)=beta KL(p||mu), Proposition 1 gives

    Omega*(r)=beta log sum_a mu_a exp(r_a/beta),
    grad Omega*(r)_a proportional to mu_a exp(r_a/beta).

Use r_a=alpha R_a to identify Dr.GRPO+reference-KL's optimum and q*=mu(.|correct) in a short attributed specialization. The explicit conditional ratio follows immediately. This replaces treating the Gibbs optimum as an original result.

**Scope.** Bellman contraction is not convergence of gradient ascent in logits. Cite Mei for that optimizer. A fixed-bank replay loss is a selective cross-entropy barrier rather than their fixed-reference reverse KL; it also permits optima on unbanked faces. Their theorem does not automatically supply our replay convergence proof. MaxRL's nonlinear Psi(P) also needs a separate mapping, rather than substitution of a fixed reward vector.

## 4. Zhan et al. 2023 — useful alternative optimizer, not our update

Existing key: `zhan2023pmd`. Primary sources: [final January 2023 author version, arXiv v4](https://arxiv.org/html/2105.11066v4), [published SIAM record](https://epubs.siam.org/doi/10.1137/21M1456789). The numbering here is explicitly **v4**.

**Exact anchors.** **Theorem 1 (Exact GPMD)** gives linear convergence for Algorithm 1 with any eta>0 and convex regularization satisfying Assumption 1; policy-distance convergence additionally uses strong convexity. The factor is 1-(eta tau/(1+eta tau))(1-gamma). **Theorem 2 (Approximate GPMD)** allows the uniform policy-evaluation and subproblem errors specified in Assumptions 2–3, yielding an error floor.

**Reuse decision.** Theorems apply to their policy-space Bregman proximal step and dual recursion, not to Euclidean softmax SGD, PPO, or AdamW. The regularizer determines the Bregman divergence; replacing that proximal solve with a neural loss step changes the algorithm. An algorithmic comparator using exact policy mirror descent could cite these results directly. They cannot erase our existing optimizer qualifications. Their approximate theorem also requires quantified oracle errors; observed stochastic loss fluctuations do not establish those assumptions.

A detail to avoid misreading: their Delta_zeta is a neighborhood allowing total mass in [1-zeta,1+zeta], not a constraint giving every action probability a positive floor.

## 5. Vieillard et al. 2020 — contextual only for current Appendix P

Existing key: `vieillard2020kl`. Primary source: [NeurIPS published paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/8e2c381d4dd04f1c55093f22c59c3a08-Paper.pdf).

**Exact anchors.** Section 3, Eq. (1), defines KL-regularized modified policy iteration against the **previous policy**. **Theorem 1** bounds error propagation for dual-averaging value iteration without entropy, under a bounded Q-iterate condition. **Theorem 2** treats the additional entropy term.

**Reuse decision.** This establishes the averaging effect of successive-policy KL within approximate dynamic programming. Our comparator uses a frozen reference and direct logit gradients. The reference, optimizer, evaluation step, and error model differ. Consequently these theorems should remain contextual citations, not replace the fixed-reference stationary/convergence or replay-recovery results. The exact greedy subproblem is also a stated premise and is not established for our neural updates.

## Suggested compact citation architecture

1. Attribute the Gibbs/reference-conditional optimum to Geist Proposition 1 and give its one-line substitution.
2. State Dr.GRPO+reverse-KL convergence as the attributed Mei Theorem 5 corollary above, including its step restriction and exact-gradient scope.
3. Attribute natural-gradient ratio preservation to Agarwal Lemma 15 with a one-line equal-reward specialization.
4. Cite Agarwal Section 5.2 when introducing the forward-KL/log-barrier distinction, while keeping the selective-bank and complete-response arguments explicit.
5. Keep GPMD and previous-policy-KL convergence as related optimizer theory, unless the paper introduces those algorithms as actual comparators.

This uses existing foundations to shorten standard background and strengthens the comparator honestly. The neural covariance result, finite sampled PCMD loss, bank-specific threshold/recovery bounds, and finite-inference visibility statements are still the portions requiring our explicit derivations.
