# Reusing existing RLVR theory sources

Read-only manuscript audit, 2026-09-21. Primary arXiv HTML was checked directly; links are pinned to the checked versions where verified. DPH-RL Appendix D.1 is in v4 (2026-03-03), not its original v1. The strongest uses are SetPO for a metric-specific influence identity, DPH-RL for the replay/forward-KL equivalence, AVSPO for conditional scores, and Cui et al. for the distinction between update geometries. None supplies a neural PCMD-retention theorem for the implemented optimizer.

## 1. SetPO: exact specialization to the paper's metric

Source: [SetPO, Theorem 4.2, Assumption 4.1, Equations (2)–(4)](https://arxiv.org/html/2602.01062v1), existing key `li2026setpo`.

The theorem differentiates F(Q)=E_Q[g(E_Q k(Y,Y′))] under Qε=(1−ε)Q+εδy. It assumes a bounded measurable symmetric kernel and continuously differentiable nonincreasing g with bounded derivative.

Use the **correct-conditional complete-response law** Qθ=πθ(·|correct), the fixed execution-key kernel k(y,y′)=1[V(y)=V(y′)], and g(u)=1−u. All assumptions hold. If q_c=Qθ(V(Y)=c), then

- local mass: m_Q(y)=q_{V(y)};
- functional: F(Q)=1−Σ_c q_c²=PCMD;
- influence: I(y;Q)=2(Σ_c q_c²−q_{V(y)}).

For arbitrary differentiable neural θ, fixed verifier/key sets, and justified differentiation under the integral,

    ∇θ PCMD = 2 E_{Y~Qθ}[(Σ_c q_c²−q_{V(Y)}) ∇θ log πθ(Y)].

The conditional normalization cancels because E_Q I=0. No independent-logit assumption is needed for this identity; its sign along training remains unspecified.

**Reuse:** cite this theorem for the metric/influence step, then substitute our actual dynamics. For the tabular drift alone, Harper's replicator variance theorem is shorter; citing both as separate new results would add repetition. SetPO does not establish our replay threshold or optimizer convergence.

## 2. Choice of Divergence / DPH-RL: reuse the valid replay identity

Source: [DPH-RL, Appendix D.1 and Section 4.1](https://arxiv.org/html/2509.07430v4), existing key `li2025divergence`.

Appendix D.1 explicitly identifies forward-KL minimization with maximizing likelihood of stored reference trajectories. Thus our identity

    R_w(p) = H(w) + KL(w || p)

can be presented as an established rehearsal construction, with our contribution being a verified, discovered, mode-balanced target and its dynamics. This supports the forward-/reverse-KL comparison without making the identity itself a theorem contribution. It does not replace our Ψmax sublevel proof, target-specific floor, or recovery calculation.

**Audit-only warning:** do not import Theorem 1's improvement bonus. At γ=0, its displayed bound becomes J(π)−L_old(π)≥δ. In a one-state bandit the left side is identically zero. With Bernoulli reward, π_old=π=(1/2,1/2), and reference probabilities (0.9,0.1) favoring the rewarding action, the reference advantage is δ=0.4, contradicting that bound. This independent check concerns the stated theorem, not the valid Appendix D.1 equivalence. Also, Jensen–Shannon divergence is bounded; its qualitative coverage motivation must not be imported as an infinite boundary barrier.

## 3. AVSPO: an existing conditional-score lemma and careful stochastic reuse

Source: [AVSPO, Lemma B.1; Theorem B.4, Equations (25)–(26); Theorem B.10, Equation (38)](https://arxiv.org/html/2605.21125v2), existing key `avspo2026`.

Lemma B.1 gives E[∇logπ(Y)|Y∈E]=∇logπ(E). Theorem B.4 applies it to homogeneous success/failure groups. Cite that primitive in our neural mean-direction proof; our extension handles every reward-vector-measurable detached advantage and separately computes covariance. Virtual rewards can restore a homogeneous group's update without selecting among correct identities.

Their Equation (38), under smoothness, bounded objective, bounded conditional variance σ², bias b_n, and constant η≤1/(4L), bounds average squared gradient by

    4(Fmax−F0)/(ηT) + 2Lησ² + (3/T)Σ_n E||b_n||².

This could supply a **conditional stationarity** bound after defining the joint replay objective and measuring/bounding update bias; it is not individual-mode retention.

**Audit-only warning:** the stated inference that b_n→0 gives exact stationarity does not follow at fixed η,σ>0. For F(θ)=−θ²/2 with unbiased additive noise, stationary Eθ²=ησ²/(2−η)>0. Reuse the finite bound, retaining its noise floor, or impose a valid diminishing-step regime. No AdamW/PPO assumptions are established by that citation.

## 4. Cui et al.: cite geometry-dependent entropy identities, not mode retention

Source: [The Entropy Mechanism, Lemma 1, Proposition 1, Theorems 1–2, Appendix E.2–E.4](https://arxiv.org/html/2505.22617v1), existing key `cui2025entropy`.

For independent tabular softmax logits, Lemma 1 gives first-order entropy change −Cov_π(logπ,Δz). Proposition 1 supplies vanilla expected-gradient Δz=ηπA. Theorem 1 therefore has −ηCov_π(logπ,πA), while Theorem 2 replaces πA with A for natural policy gradients.

**Reuse:** credit these results for the geometry distinction and shorten background softmax/entropy derivations. Our binary conditional natural-gradient invariance is a direct specialization: correct outcomes receive equal advantages, so their logit differences do not change. Our Euclidean PCMD drift and stochastic second-order loss still require their own metric-specific calculation.

The source statements are first-order local approximations under specified tabular updates. They do not prove monotone entropy under arbitrary advantages, mode support loss, or neural optimizer convergence. Trajectory/token entropy also does not determine PCMD: conditional trajectory entropy can increase entirely within one verified mode. Do not replace the verified-mode argument with this entropy result.

## 5. AGRAE: reuse the coordinate identity, retain the probability caveat

Source: [AGRAE, Theorem 1, Appendix B.3 Equations (23)–(29); Theorem 2, Equation (10)](https://arxiv.org/html/2602.05548v3), existing key `agrae2026`.

The independent trajectory-logit calculation gives Δz_b=η[A_b^aggregate−p_bΣ_i A_i]. With mean-zero group advantages, an absent category receives zero direct logit increment. Without centering it can receive the normalization contribution. Theorem 2 gives Σ_i|A_i|=2G√[p̂(1−p̂)] for binary reward standardization without a stabilizer.

**Reuse:** cite the first identity as a primitive for our sampled-coordinate bound and stochastic PCMD expansion. Our contribution includes duplicate category counts, exact expectation/covariance, and the resulting PCMD change. The second identity is useful context for difficulty weighting, not a replacement for c_G(P) or covariance: p̂ is empirical group correctness, and a sum of advantage magnitudes is not a neural gradient norm.

The cited update is a **logit** identity despite loose probability wording in its presentation. An unsampled logit staying fixed does not imply its normalized probability stays fixed. Neither result certifies neural mode retention.

## 6. DMPO: useful comparison, limited transferable theorem

Source: [Distribution Matching, Proposition 3.1 and Equations (5)–(8)](https://arxiv.org/html/2605.19461v1), existing key `li2026distribution`.

Proposition 3.1 is the coordinate consequence of group MSE: L_DM<δ implies q_i≥target_i−√(Gδ). Cite it when describing what fresh-group distribution matching actually protects. Its q_i is a softmax of mean-token log scores **within the sampled group**, not unconditional trajectory or execution-key probability. It cannot replace the bank-retention floor.

Two independent checks limit reuse. First, equal-reward Boltzmann maximizers do not become a unique Dirac mass as temperature tends to zero; on a finite support the limit is uniform over tied maximizers. Second, MSE on group scores is unchanged when every group log score receives the same offset, so perfect matching supplies no lower bound on total group probability. Proposition 3.2's proportional-sampling interpretation also requires equal lengths or explicit length factors because Equation (6) normalizes mean-token scores.

**Recommendation:** retain this as a precise local-comparator citation. Do not borrow its global collapse, mode-coverage, or optimizer-convergence rhetoric to justify Appendix P.

## Suggested proof economy

1. Cite AVSPO's conditional-score lemma and AGRAE's softmax-coordinate identity, then state our genuinely broader mean/covariance specialization.
2. Cite standard replicator theory/Harper for tabular variance dynamics; use SetPO only for the separate verified-key functional and arbitrary-neural-gradient bridge.
3. Cite DPH-RL Appendix D.1 for likelihood replay as forward KL, while retaining our target-specific retention and recovery proofs.
4. Keep the finite-step PCMD, sampled symmetry-breaking coefficient, replay competition threshold, and finite-budget visibility results. The reviewed sources do not already supply those statements under the paper's assumptions.
