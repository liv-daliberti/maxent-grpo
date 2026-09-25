# Theory source reuse for Appendix P

Implemented after approval: see the [implementation and validation record](../audits/theory_source_impl_20260921/README.md). The text below preserves the original research audit.

Research audit, 21 September 2026. Scope: the existing entries in
`paper/example_paper.bib`, the current Appendix P, and the earlier local
September 5 theory notes. Primary sources were checked for theorem statements
and assumptions. This audit adds research notes only; it does not change
`main.tex`, `main.pdf`, or Section 3.1. Theorem numbers below refer to the
current compiled appendix and are accompanied by stable source labels.

Yes: several sources already in our set supply useful results directly.
The best revision would import the established machinery, state short
corollaries with the substitutions made explicit, and concentrate our proofs
on finite-group estimators and the competition between fresh updates and
verified replay. Two imports strengthen existing guarantees substantially:
Mei et al. for Dr.GRPO with exact KL, and KL data processing for retention.

| Priority | Existing source | Concrete use in Appendix P | Proof treatment |
|---|---|---|---|
| 1 | Mei et al. (2020), Theorem 5 and Lemma 13 | Upgrade the Dr.GRPO branch of P.28 from a stationary-policy characterization to discrete convergence and a trajectory probability floor | Attributed corollary; reward transformation and assumption matching only |
| 2 | van Erven–Harremoës (2014), Theorem 9 | Sharpen P.20's replay floor and the complete-response-to-key certificate | Cite data processing; state binary-KL inversion |
| 3 | Tajwar et al., MaxRL v3, Theorem 5 | P.3 is already the practical-estimator theorem in our notation | Replace its repeated derivation with the exact substitution |
| 4 | Harper (2009), Theorems 1–2 | The first identity in P.13 is the replicator fitness-variance identity | Attribute it; retain the matched-correctness calculation |
| 5 | SetPO, Theorem 4.2 | Identify PCMD as an existing kernel diversity functional and obtain its neural parameter gradient | Short specialization using the verified-key equality kernel |
| 6 | Anceaume et al. (2015), Theorem 3 | Uniform correct-mode probabilities optimize finite-budget coverage at fixed correctness | Cite the stochastic-order theorem and map incorrect draws to null coupons |
| 7 | Wang et al. (2017), Lemma 2.1; Howard et al. (2020), Lemma 1 | Conditional stochastic retention beyond exact pathwise energy descent | Optional corollary; retain explicit noise/bias/optimizer premises |

**1. A published convergence result we can actually use.**

For Dr.GRPO, put `a=(G−1)/G` and consider exact Euclidean-logit ascent of

\[
J(p)=aP-\beta\mathrm{KL}(p\|\mu)
    =\sum_i p_i\underbrace{(a\mathbf1_{i\in C}+\beta\log\mu_i)}_{\widetilde r_i}
       +\beta H(p).
\]

This is an entropy-regularized bandit with a fixed transformed reward.
[Mei et al., Theorem 5](https://proceedings.mlr.press/v119/mei20b/mei20b.pdf)
therefore applies after normalizing the finite reward vector. With a
full-support fixed reference, finite initial logits, exact gradients, and a
constant step `0<η≤1/β`, it yields geometric decay of the objective gap and
convergence to

\[
p_i^*=\frac{\mu_i\exp(a\mathbf1_{i\in C}/\beta)}
 {\sum_j\mu_j\exp(a\mathbf1_{j\in C}/\beta)},
\qquad q_c^*=\frac{\mu_c}{\sum_{d\in C}\mu_d}.
\]

Their Lemma 13 also supplies a positive all-iteration probability floor.
This strengthens `prop:kl-stationary` for Dr.GRPO. It does not extend directly
to MaxRL's nonlinear potential, sampled KL estimates, or AdamW. The theorem
is discrete; applying its lemmas to continuous flow is an additional
adaptation, not the theorem's literal statement.

For exact constants and a proposed attributed corollary, see the
[optimization audit](../audits/theory_source_reuse_20260921_optimization.md).
The practical interpretation should acknowledge that KL can retain modes
under these assumptions, while inheriting the reference's conditional
imbalance. Our contribution then concerns target choice and finite recovery.

**2. A much stronger retention certificate with almost no new proof.**

For normalized bank weights `w`, suppose replay cross entropy satisfies
`R_w(p)≤C`. Put `ε=C−H(w)`. By
[van Erven–Harremoës, Theorem 9](https://arxiv.org/html/1206.2459v2), coarsening
to a banked category and its complement gives

\[
\operatorname{kl}(w_b,p_b)\le\mathrm{KL}(w\|p)\le\varepsilon,
\qquad p_b\ge\ell_-(w_b,\varepsilon),
\]

where `ℓ₋` is the lower solution of binary KL equal to `ε`. Coarsening to
the entire bank also gives `p(B)≥exp(−ε)`. This improves the termwise floor
in `thm:replay-retention` without reproving log-sum or Pinsker.

Illustration, independently recomputed: a uniform bank of 16 with
`C=log(16)+0.01` gives a per-mode floor **0.0339712**, compared with
**4.62×10⁻²⁰** from `exp(−16C)`. These are hypothetical mathematical inputs,
not measured training losses.

For neural exemplars, first normalize complete-response coefficients
`α_j/L_j`, then apply the same theorem to the deterministic execution-key
map. Use one prompt at a time, or define a joint prompt law. Our energy
control and response-to-key mapping remain necessary. The old
[survival fragment](../audits/theory_literature_extensions_20260905/survival_body.tex)
already contains this extension; import the specialization, not its general
information-inequality proofs.

**3. Remove the duplicate MaxRL estimator proof.**

[MaxRL v3, Appendix D, Theorem 5](https://arxiv.org/html/2602.02710v3)
already analyzes the centered estimator that drops the whole update on
all-failure groups. Set their `N=G` and pass rate to our `P`:

\[
\mathbb E\widehat g=
\left[\sum_{j=0}^{G-2}(1-P)^j\right]\nabla P
=\frac{1-(1-P)^{G-1}}{P}\nabla P.
\]

P.3 (`lem:maxrl-mean`) can be explicitly labeled a specialization and its
proof shortened to this substitution. Keep the `G−1` versus `G` distinction
because it identifies the implemented estimator. Their Theorem 1 also
supplies the conditional-score identity used in our neural argument;
our general advantage coefficient and covariance still need their own
short calculation.

**4. Attribute the classical part of PCMD drift.**

[Harper, Theorems 1–2](https://arxiv.org/html/0911.1383v1#S2.SS4)
give the gradient geometry and fitness-variance identity for replicator
dynamics. Set the potential `V(q)=½Σq_c²` and fitness `f_c(q)=q_c`.
Since `PCMD=1−2V`, the first formula in P.13 follows immediately:

\[
\frac{d\,\mathrm{PCMD}}{d\tau}=-2\operatorname{Var}_{C\sim q}(q_C).
\]

The second formula, indexed by log-odds of correctness, is our short
specialization. Keep it. The reduction from the actual group estimator to
replicator dynamics, and the proof that effective time is unbounded, are
also necessary. A general replicator citation alone does not check those
premises or prove the finite-step results.

**5. SetPO provides a direct verified-key metric bridge.**

[SetPO, Assumption 4.1 and Theorem 4.2](https://arxiv.org/html/2602.01062v1)
apply to a bounded symmetric kernel and a smooth decreasing shaping
function. On the correct-conditional response law `Qθ`, choose
`k(y,y′)=1{V(y)=V(y′)}` and `g(u)=1−u`. Their diversity functional becomes
exactly PCMD, and its influence is `2(Σq_c²−q_{V(y)})`.

For a fixed verifier and differentiable neural policy, the score identity
then yields

\[
\nabla_\theta\mathrm{PCMD}
=2\mathbb E_{Y\sim Q_\theta}
 \left[(\sum_cq_c^2-q_{V(Y)})\nabla_\theta\log\pi_\theta(Y)\right].
\]

This is a useful attributed corollary without independent logits. It gives
a metric gradient, not a sign for its change during training. Use Harper
for the brief tabular drift proof and SetPO for this distinct neural metric
connection; avoid presenting two proofs of the same drift identity.

**6. Borrow a stronger reason for a uniform target.**

[Anceaume et al., Theorem 3](https://arxiv.org/html/1504.03878v1)
states stochastic optimality of equal non-null coupon probabilities at
fixed null probability. Map incorrect responses to the null coupon and
the `m` correct execution modes to non-null coupons. For a frozen policy
with fixed correctness `P`, `p_c=P/m` minimizes the time to collect every
specified number of distinct modes in stochastic order. Consequently it
maximizes each probability `Pr(distinct@K≥j)`.

This directly justifies the ideal uniform-correct target for finite-budget
coverage. It is stronger than a statement about expected distinct count.
It does not assert that uniform bank selection makes the trained neural
policy uniform, nor that an incomplete bank discovers omitted modes.
State the implication as a cited corollary, with no coupon-collector proof.

**7. If we extend optimizer scope, import the probability machinery.**

[Wang et al., Lemma 2.1](https://optimization-online.org/wp-content/uploads/2016/07/5529.pdf)
provides a stochastic descent inequality under smoothness, unbiased
finite-variance gradients, and a bounded preconditioner determined before
the current gradient noise. Apply it to the fixed replay potential.
If the resulting nonnegative energy satisfies
`E_t D_{t+1}≤D_t+d_t` with a deterministic summable error budget, adding the
remaining error budget makes a nonnegative supermartingale.
[Howard et al. (2020), Lemma 1](https://arxiv.org/html/1808.03204)
then supplies a simultaneous trajectory bound by Ville's inequality.

Only the replay energy and its conversion into exemplar probabilities
need explanation here. The
[existing optimizer fragment](../audits/theory_literature_extensions_20260905/optimizer.tex)
already contains the conditional extension. It does not certify the
implemented AdamW process: current-gradient moments, clipping, changing
banks, and scheduling require additional control. This optional result
improves mathematical scope only when those assumptions are explicit.

**Other useful imports and limits.**

- [Choice of Divergence v4, Appendix D.1](https://arxiv.org/html/2509.07430v4)
  already connects stored-trajectory likelihood with forward KL. Cite that
  identity when positioning replay; our distinct work concerns verified
  target construction and its retention/recovery behavior. Pin v4: v1's
  Appendix D.1 is a different section.
- [AVSPO, Lemma B.1](https://arxiv.org/html/2605.21125v2) supplies a general
  conditional-score identity. Its finite stochastic stationarity bound
  retains a noise floor; it does not establish our per-key retention claim.
- [Cui et al.](https://arxiv.org/html/2505.22617v1) provides tabular entropy
  change formulas under Euclidean and natural-gradient updates. Use these
  for geometry attribution, not as a substitute for an executed-key PCMD
  statement.
- [Howard et al. (2021), published Proposition 7](https://arxiv.org/html/1810.08240)
  supplies a beta-binomial confidence-sequence construction for repeated
  evaluation of a frozen policy. Cite it if adding a retention-measurement
  protocol; do not rederive mixture martingales. Changing checkpoints
  cannot be pooled as one fixed Bernoulli law.
- [Agarwal et al., Lemma 15](https://jmlr.org/papers/volume22/19-736/19-736.pdf)
  gives the natural-policy-gradient multiplicative update. Equal rewards
  preserve correct-mode ratios at every finite step, giving a direct
  citation for the geometry qualification.
- Geist, Zhan, and Vieillard analyze useful regularized-control or
  mirror-descent settings, but their optimizer assumptions do not match
  our Euclidean sampled neural updates. Mei is the direct import here.
- The audits record specific limitations in stronger claims from DPH-RL,
  DMPO, and AVSPO. Borrow their checked identities rather than transferring
  broad collapse or convergence claims without matching premises.

**Recommended appendix revision.**

1. Compress P.3 and the classical part of P.4/P.13 into attributed
   specializations. Preserve the estimator normalization and time-change
   checks.
2. Add the sharp data-processing certificate after P.20 and apply it to
   complete-response execution keys.
3. Add the Mei corollary next to P.28, scoped to exact Dr.GRPO + KL updates.
4. Add a short cited finite-budget coverage consequence of Anceaume.
5. Use SetPO for the neural metric-gradient bridge if it helps connect the
   new neural estimator results to PCMD. Treat stochastic-energy retention
   as optional; do not add several pages of standard probability proofs.

Retain full proofs for the parts not supplied by the reviewed sources:
the exact general finite-group coefficient and covariance; finite expected
steps and sampled PCMD loss; replay's competition threshold and recovery
bound; and the actual link from protected exemplars to verified modes.
This makes the division between established theory and our applications
explicit, while strengthening the guarantees readers can use.

Detailed source audits:
[optimization](../audits/theory_source_reuse_20260921_optimization.md),
[probability and optimizer scope](../audits/theory_source_reuse_20260921_probability.md),
[recent RLVR](../audits/theory_source_reuse_20260921_rlvr.md).
