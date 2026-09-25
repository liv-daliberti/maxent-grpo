# Finite-time and stochastic additions to Appendix P

These are derived directly from the appendix's categorical assumptions. They do not make a convergence claim for neural AdamW/PPO. The highest-value genuinely additional result is the exact finite sampled-step symmetry-breaking theorem below: it proves a phenomenon that disappears when sampling is replaced by its expectation.

## 1. The theory can predict the exact population metric

Let D(q)=PCMD(q)=1-S2(q), S2=∑_c q_c², S3=∑_c q_c³, and let effective time satisfy dτ=c(P)P(1-P)dt. Under the existing mean flow,

  dD/dτ = -2[S3-S2²] = -2 Var_{C~q}(q_C).

Proof: differentiate S2 using q'_c=q_c(q_c-S2). This is strictly negative at every nonuniform interior q. Equality holds at an interior point exactly when q is uniform. This gives a direct, quantitative theorem for the paper's headline metric, rather than only a limiting single-winner statement.

There is a useful matched-correctness formulation. Write r_a=p_a/(1-P) for incorrect conditional probabilities and T2=∑_a r_a². Then

  d(logit P)/dτ = S2+T2,
  dD/d(logit P) = -2 Var_q(q_C)/(S2+T2).

The coefficient c(P), advantage choice, and group size all cancel. Thus all included positive-coefficient outcome-binary mean flows initialized at the same logits predict the same PCMD-versus-correctness curve, not merely the same winning mode. This is directly testable with matched-checkpoint accuracies. It applies only to these mean flows: the stochastic theorem below explains one reason empirical curves need not coincide.

## 2. Exact finite-step stochastic symmetry breaking

**Proposition (uniform correct modes lose expected PCMD under one sampled update).** Fix finite logits with 0<P<1 and m≥2 equally likely correct modes, q_c=1/m. Draw one G≥2 group and update z⁺=z+η ĝ_G using the paper's centered estimator, with η>0, w(r)>0 for 1≤r≤G-1, and zero update on all-equal groups. Then

  E[D(q⁺)] < D(q)=1-1/m

for every η>0. Yet E[q⁺_c]=1/m for every c by permutation symmetry. As η→0,

  E[D(q⁺)] = 1-1/m
     - η² (m-1)/m³ E[w(R)² R(G-R)²/G⁴] + O(η³).

All expectations are conditional on the fixed initial policy. This is a one-step statement about SGD in independent logits, not an eventual stochastic-collapse theorem.

**Proof.** Let N_c count correct category c in the group and R=∑_c N_c. Because the group advantages sum to zero, the softmax-score baseline cancels exactly, giving

  ĝ_c = k_R N_c,   k_R=w(R)(G-R)/G²

on mixed groups, and zero on all-equal groups. Conditional on R, N~Multinomial(R;1/m,…,1/m). The correct conditional distribution after the update is exactly

  q⁺_c = exp(η k_R N_c) / ∑_d exp(η k_R N_d).

PCMD has its unique maximum 1-1/m at uniform q. On the positive-probability event R=1, one correct logit increases while all other correct logits stay fixed; therefore q⁺ is nonuniform and PCMD strictly decreases. Every other outcome has PCMD no larger than its initial maximum. This proves strict expectation inequality without a small-step approximation.

For the expansion, for any fixed vector v, softmax(ηv)_c=1/m+η(v_c-v̄)/m+O(η²). Because the coordinates always sum to one,

  ∑_c softmax(ηv)_c² = 1/m + η²/m² ∑_c(v_c-v̄)² + O(η³).

Conditional multinomial variance gives

  E[∑_c(N_c-R/m)² |R]=R(1-1/m).

Substitute v=k_R N and average over R. The finite set of possible groups and bounded w makes the Taylor remainder uniformly O(η³). Permutation symmetry gives the expected-coordinate statement.

**Dr.GRPO specialization (w=1):**

  E[w(R)² R(G-R)²/G⁴]
    = (G-1)/G³ · P(1-P) · [1+(G-2)(1-P)].

Proof uses E[R(G-R)(G-R-1)]=G(G-1)(G-2)P(1-P)² and E[R(G-R)]=G(G-1)P(1-P). At fixed P away from endpoints and large G this is P(1-P)²/G + O(G⁻²).

**Reward-standardized specialization without numerical stabilizer**, w(r)=G/sqrt[r(G-r)] on mixed groups:

  E[w(R)² R(G-R)²/G⁴] = [(1-P)-(1-P)^G]/G.

**Centered binary MaxRL specialization:** k_r=(G-r)/(Gr) for r>0, hence the coefficient inside the variance formula is

  E[(G-R)²/(G² R) · 1{R>0}].

At very small P with fixed G this is ((G-1)²/G)P+O(P²), whereas Dr.GRPO gives ((G-1)²/G³)P+O(P²): the raw one-step variance effect is G² larger at the same learning rate. Since MaxRL's mean update is also about G times larger in that regime, this is not a fair algorithm-quality comparison without matching step sizes.

**Predictions and qualifications:**

- Even perfectly balanced correct logits do not protect expected diversity under finite sampling. Averaging logits or q before calculating diversity hides this effect.
- At fixed P and common SGD learning rate, the initial expected PCMD decrement is quadratic in learning rate and approximately inverse in G for Dr.GRPO at large G.
- This coefficient is a per-update prediction. Comparisons at fixed total rollout budget or matched correctness need a different accounting.
- Larger groups therefore change stochastic symmetry breaking. The paper should not export 'group size only rescales time' outside its explicitly deterministic theorem.
- The proof does NOT show expected PCMD decreases at every nonuniform policy, identify which mode wins under noise, or prove eventual collapse for constant-step SGD. Those are harder statements and may require qualifications or counterexamples.

## 3. Finite-budget invisibility rather than finite-time support loss

Let * denote the initially unique largest correct coordinate, a=q_*(0), and define

  A = ∑_{c≠*} q_c(0)/(a-q_c(0)).

For every finite effective time τ≥0,

  1-q_*(τ) ≤ A exp(-aτ),
  PCMD(q(τ)) ≤ 2A exp(-aτ).

Consequently, among K independent full-policy draws at training time τ,

  Pr(at least one correct non-* mode is observed) ≤ K A exp(-aτ).

In particular, if τ≥a⁻¹ log(KA/δ), with the threshold replaced by zero if its right side is negative, all K draws fail to reveal any competing correct mode with probability at least 1-δ. This is an operational finite-time collapse statement even though all mode probabilities remain strictly positive. It also applies to K draws conditioned on correctness.

**Proof.** Let ξ_c=q_c/q_*. Its dynamics are ξ'_c=-q_* ξ_c(1-ξ_c), and q_* is nondecreasing because q_*≥∑q². Hence q_*≥a and

  ξ_c(τ)/(1-ξ_c(τ))
      ≤ [ξ_c(0)/(1-ξ_c(0))] exp(-aτ).

Thus ξ_c(τ)≤[q_c(0)/(a-q_c(0))]exp(-aτ). Since 1-q_*=∑ξ/(1+∑ξ)≤∑ξ, the first inequality follows. Also PCMD=1-∑q²≤1-q_*²≤2(1-q_*). A union bound over K draws gives the visibility result because each draw hits a competing correct mode with probability P(1-q_*)≤1-q_*.

A corresponding expected observed-support bound is

  E[number of distinct correct modes in K draws] ≤ 1+KAexp(-aτ).

**Correctness-indexed version.** Since S2+T2≤2,

  τ ≥ [logit P(τ)-logit P(0)]/2.

Therefore

  1-q_*(τ) ≤ A [odds(P(τ))/odds(P(0))]^(-a/2).

This gives an explicit bound at matched correctness, without estimating c(P). It is conservative; report it as a sufficient visibility bound, not a sharp empirical collapse-time predictor.

**Optional physical-time bound for coefficients with c_min=min_[0,1] c(P)>0.**

  τ(t) ≥ (1/2) log[1+2c_min P(0)(1-P(0))t],

so 1-q_*(t)≤A[1+2c_min P(0)(1-P(0))t]^(-a/2). This applies to the three named estimators under their existing finite-weight assumptions. It does not automatically apply to every coefficient in the expanded outcome-binary corollary, whose endpoint value can be zero.

Proof: logit P(τ)≤logit P(0)+2τ implies 1-P(τ)≥(1-P(0))exp(-2τ), while P(τ)≥P(0). Thus τ'≥c_min P(0)(1-P(0))exp(-2τ), which integrates to the stated expression.

## 4. Optional asymptotic rates expose a limitation of the current narrative

With n incorrect categories, all initial probabilities positive, unique correct leader, and c(1)>0, the existing ODE also yields

  1-P(t) ~ 1/[c(1)(1+1/n)t],
  q_c(t) = Θ(t^(-n/(n+1))) for each c≠*,
  PCMD(q(t)) = Θ(t^(-n/(n+1))).

Sketch: incorrect conditional dynamics are r'_a=-r_a(r_a-T2), so r→Uniform(n); correct q→e_*; deviations decay exponentially in τ. Consequently d logitP/dτ→1+1/n, with an integrable error, and 1-P∼Cexp[-(1+1/n)τ]. Since τ'=c(P)P(1-P), e^((1+1/n)τ)∼(1+1/n)c(1)Ct. Also each losing q_c∼C_c exp(-τ), since dlog(q_c/q_*)/dτ=q_c-q_*=-1+integrable error.

This rate result is mathematically clean but less useful for the main paper than the stochastic and visibility propositions. It cautions that eventual collapse need not mean rapid collapse within the experimental budget. The exponent depends on the independently parameterized incorrect-category abstraction and should not be advertised as a neural-training prediction.

## Recommended theory package

Main text: outcome-binary mean-direction theorem + exact PCMD drift + stochastic symmetry breaking + the replay drift/retention result from the other audit. Present the finite-budget visibility corollary as the practical meaning of collapse, rather than claiming exact support disappears at finite time. The algebra is short enough to prove fully, and the stochastic theorem adds a claim the current mean-flow theorem fundamentally cannot capture.

## 5. Remove the infinitesimal-step assumption for expected-gradient SGD

There is also an inexpensive discrete deterministic strengthening, which uses no small-learning-rate restriction.

**Proposition (finite expected-gradient steps).** Let

  z_{k+1}=z_k+η_k c(P_k)∇_z P_k,

where every η_k>0 is finite, ∑_k η_k=∞, and c is continuous and strictly positive on (0,1). Start at finite logits with a unique largest correct coordinate. Then P_k→1 and q_{*,k}→1. Moreover PCMD(q_{k+1})<PCMD(q_k) whenever q_k is nonuniform, for every finite η_k>0.

**Proof.** Set h_k=η_k c(P_k)P_k(1-P_k)>0. The exact coordinate updates are

  z_c⁺=z_c+h_k q_c,   z_a⁺=z_a-h_k r_a,
  q_c⁺ = q_c exp(h_k q_c)/∑_d q_d exp(h_k q_d).

After sorting q in descending order, the ratios q_c⁺/q_c are decreasing in c. Since both vectors sum to one, their difference crosses zero at most once from positive to negative; every sorted prefix of q⁺ therefore has at least the mass of the same prefix of q. Thus q⁺ majorizes q. Since ∑q² is strictly Schur-convex, PCMD strictly decreases unless q was uniform. This also gives monotonic decrease for any strictly Schur-concave diversity statistic, including Shannon entropy.

Each correct logit increases and each incorrect logit decreases, so P increases. The exact correctness-odds increment is

  logit P⁺ - logit P
    = log∑_c q_c exp(h_k q_c) - log∑_a r_a exp(-h_k r_a)
    ≥ h_k∑_c q_c² ≥ h_k/m.

If P converged to an interior limit, continuity/positivity of c and ∑η_k=∞ would imply ∑h_k=∞, forcing logit P to diverge by this inequality, a contradiction. Therefore P→1. Also ∑h_k must diverge: otherwise all correct/incorrect logit increments would have finite total magnitude, the logits would converge to finite values, and P could not tend to one.

The initial correct leader remains the leader and its conditional probability is nondecreasing by the prefix inequality. Let a=q_*(0) and ξ_{c,k}=q_{c,k}/q_{*,k}. Then

  ξ_{c,k+1}=ξ_{c,k} exp[-h_k(q_{*,k}-q_{c,k})],
  q_{*,k}-q_{c,k} ≥ a(1-ξ_{c,0})>0.

Since ∑h_k=∞, every ξ_{c,k}→0. Hence q_*→1.

**Scope:** this removes the infinitesimal approximation for SGD on the exact expected gradient. It does not make sampled SGD monotone step by step, restore exact estimator-independent trajectories at finite step sizes, or establish the claim for AdamW/PPO. It gives a useful middle layer between the current mean flow and the separate stochastic one-step theorem.
