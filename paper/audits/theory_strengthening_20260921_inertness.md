# Theory audit: binary inertness, finite updates, and shared parameters

Prepared from `paper/main.tex` lines 3661–4106, 4107–4435, 5097 onward and the corresponding compiled-PDF text. No manuscript changes made.

## Executive recommendation

Promote the reward-conditioning identity to a general neural-policy theorem, then state its stochastic covariance and variable-length residual. Keep categorical collapse as a consequence of a specified geometry. A discrete full-batch theorem removes the infinitesimal assumption cheaply. A separate G=2 reinforced-urn theorem can remove sampling-noise removal too, but it yields a **random** surviving mode and remains a specialized categorical result.

Do not extend objective inertness to multi-prompt shared training, Adam, PPO epochs, or finite-step conditional-probability trajectories. The exact two-prompt counterexample below shows why. These limitations are useful scientific content, not merely disclaimers.

## 1. Neural-policy reward-conditioning theorem

Fix one prompt, a smooth policy pi_theta(y), and a fixed binary verifier event C. Let P(theta)=Pr(Y in C), 0<P<1, and s(Y)=grad_theta log pi_theta(Y). Assume differentiation under the expectation is valid. Draw G independent responses and let R_i be their binary outcomes, R=sum R_i. Use any detached, finite outcome-only advantages a(1,R),a(0,R):

    ghat_a = (1/G) sum_i a(R_i,R) s(Y_i).

Write v=grad_theta P. Then, without independent logits or any tabular assumption,

    E[ghat_a] = c_a(P) v,
    c_a(P) = ( E[R a(1,R)]/P - E[(G-R)a(0,R)]/(1-P) )/G.

Proof: conditional on the entire reward vector, response scores are independent, with means v/P for successful rows and -v/(1-P) for failed rows. Taking the conditional expectation proves the formula.

The formula also extends to indexed advantages measurable with respect to the full reward vector, replacing the count-times-advantage terms by their corresponding row sums. Exchangeability is needed for the compact a(R_i,R) notation, not for collinearity itself.

For a common autonomous preconditioner M(theta), positive c_a, and one prompt, the deterministic flows theta_dot=c_a(P) M(theta)v have the same parameter-space orbits after a scalar time change. M may be state dependent; it need not be constant or identity. This does not cover optimizer memory, objective-dependent adaptive preconditioners, clock-time learning-rate schedules, or prompt sums with unequal coefficients.

### Exact covariance: where stochastic objectives can differ

Let Sigma_1=Cov(s(Y)|R_i=1), Sigma_0=Cov(s(Y)|R_i=0), and

    beta_r = [r a(1,r)/P - (G-r)a(0,r)/(1-P)]/G.

Then

    Cov(ghat_a)
      = Var(beta_R) v v^T
        + G^(-2) E[ R a(1,R)^2 Sigma_1
                      +(G-R)a(0,R)^2 Sigma_0 ].

Proof: conditional covariance is the sum of independent conditional-row score covariances. Conditional mean is beta_R v. Apply total covariance.

This is an exact, readily interpretable extension: outcome-only objectives cannot change the one-prompt mean direction but can change both the magnitude and anisotropy of stochastic fluctuations. Consequently, identical mean-flow trajectories do not imply identical stochastic winner probabilities or finite-time diversity.

### Finite-step neural consequence

For a twice-differentiable observable h(theta), assume ||Hessian h||_op <= L_h on each update segment. For c_a>0 and the signal-normalized SGD step theta_plus=theta+eta ghat_a/c_a,

    | E[h(theta_plus)-h(theta)] - eta grad h dot v |
      <= (L_h eta^2 / (2 c_a^2)) E||ghat_a||^2.

Proof: Taylor's theorem with bounded Hessian and the exact expected-gradient identity. Comparing two objectives gives a difference bounded by the sum of their two second-order remainder bounds. If a third-order remainder is controlled, the leading difference is eta^2/2 times the Hessian contraction with the difference of their signal-normalized covariance matrices.

The statement applies to mode probabilities, conditional collision probability, or a log-probability statistic wherever the relevant derivative bounds hold. It is a one-step conditional theorem; a global stochastic approximation result needs further stability and step-size assumptions.

## 2. Discrete full-batch categorical collapse, with proof

Let p=softmax(z) over finitely many correct and incorrect categories and assume finite initial logits, 0<P_0<1, and a unique largest correct probability. Let c(P)>0 be continuous on (0,1). Consider exact expected-gradient updates

    z_(t+1) = z_t + eta_t c(P_t) grad_z P_t,

where each eta_t is finite and positive and sum_t eta_t=infinity. Then

    P_t -> 1,  q_star,t -> 1,  q_c,t -> 0 for c != star,

where star is the unique initial conditional maximum. No small-step requirement is necessary.

Proof. Define Delta_tau_t=eta_t c(P_t) P_t(1-P_t), q_c=p_c/P, and r_a=p_a/(1-P) for incorrect categories. The exact updates are

    Delta z_c = Delta_tau_t q_c,
    Delta z_a = -Delta_tau_t r_a.

Thus every correct logit increases, every incorrect logit decreases, and P_t increases. The initially largest correct logit stays largest because its increment exceeds every smaller correct increment; q_star>=1/m.

Suppose P_t converges to P_infty<1. On the compact interval [P_0,P_infty], c(P)P(1-P) has a strictly positive minimum. Hence sum Delta_tau=infinity, and z_star grows by at least (1/m)sum Delta_tau while all incorrect logits are bounded above by their initial values. This forces P_t->1, a contradiction.

Now sum Delta_tau must still be infinite when P_t->1: otherwise every logit has finite total variation because |Delta z_a|<=Delta_tau, so all logits converge to finite limits, forcing P_infty<1.

Finally let xi_c,t=q_c,t/q_star,t<1. Exact log-odds updates give

    log xi_c,t+1 - log xi_c,t
      = -Delta_tau_t q_star,t (1-xi_c,t)
      <= -Delta_tau_t (1-xi_c,0)/m.

Thus xi_c,t->0 and the result follows.

This proves identical limiting winner for positive-coefficient full-batch objectives, not identical discrete trajectories. The step-size sequence and objective can change the path before the limit.

## 3. Variable-length normalization: exact neural residual

Let h(Y)=1/L(Y), or any response-dependent multiplier, and define

    ghat_h = (1/G) sum_i a(R_i,R) h(Y_i) s(Y_i),
    A_1=E[R a(1,R)]/G,
    A_0=E[(G-R)a(0,R)]/G,
    mu_r=E[h(Y)|R_i=r],
    K_r=Cov(h(Y),s(Y)|R_i=r),

where K_r is a vector covariance. Conditional independence of responses given their outcome vector yields

    E[ghat_h] = c_tilde grad P + A_1 K_1 + A_0 K_0,
    c_tilde = A_1 mu_1/P - A_0 mu_0/(1-P).

Therefore the departure from reward-only collinearity is exactly a within-outcome covariance between response weighting and response score. Cauchy–Schwarz gives

    ||A_1 K_1 + A_0 K_0||
      <= sum_(r=0,1) |A_r| sd(h|r) sqrt(tr Sigma_r).

Equal length makes both covariances zero. Equal expected lengths alone does not. This gives a quantitative bridge to implemented GRPO and a direct measurement target rather than an all-or-nothing exclusion. PPO clipping during later epochs likewise introduces response-dependent weights and requires its own residual; the exact theorem holds at the fresh on-policy point.

## 4. Concrete shared-multi-prompt counterexample

Let theta=(u,v), sigma be logistic, and assign equal training weight to prompts A and B. Set

    P_A=sigma(u+v), P_B=sigma(u-v),
    q_A=(sigma(v),1-sigma(v)).

A valid response distribution for A is

    p_A(c1)=P_A sigma(v),
    p_A(c2)=P_A[1-sigma(v)],
    p_A(f)=1-P_A.

For B, split its correct mass equally between two correct modes; its conditional distribution can be constant. Every response has a binary verifier reward.

At u=0,v=log 3,

    P_A=3/4, P_B=1/4, q_A=(3/4,1/4),
    grad P_A=(3/16,3/16),
    grad P_B=(3/16,-3/16).

For group size G=3, Dr.GRPO has c_D=2/3, while centered MaxRL has c_M(P)=2-P. The average parameter flows are therefore

    theta_dot_D=(1/8,0),
    theta_dot_M=(9/32,-3/64).

They are not collinear. The conditional disagreement/PCMD of A is D_A=2 sigma(v)[1-sigma(v)]. At this point,

    D_A_dot_D=0,
    D_A_dot_M=9/1024 >0.

Both P_A and P_B increase under both flows. Thus changing only the binary-outcome objective can increase conditional diversity in shared multi-prompt training, without decreasing correctness or using mode-aware advantages. It does so by changing relative prompt gradient weights.

The correct general multi-prompt formula is

    E[g_a] = sum_x mu_x c_a(P_x) grad P_x.

It is generally not a global scalar multiple of the corresponding gradient for another objective. The main-text phrase that a fresh objective cannot address diversity must be restricted to a single-prompt local mean direction under common geometry.

## 5. Optional stronger stochastic theorem: G=2 exponential reinforcement

This theorem is more substantial than merely extending the full-batch flow, but is specialized and should be presented as such.

Assume independent categorical logits, constant learning rate eta>0, G=2, and centered binary advantages that give zero updates on all-equal groups and a fixed positive advantage +a to the correct row and -a to the incorrect row on a mixed group. This covers common-length Dr.GRPO, standardized binary GRPO, and centered binary MaxRL, with different constants a. Then stochastic training satisfies

    P_t -> 1 and q_t -> e_Cstar almost surely,

where Cstar is random. Every correct mode with positive initial probability has strictly positive probability of becoming Cstar, including a mode that is not initially largest.

Proof. On a mixed group, score-baseline terms cancel and the update is exactly

    Delta z = delta (e_c-e_n), delta=eta a/2>0.

On an all-equal group the policy is unchanged. At any finite logit state, the probability of a mixed group is 2P(1-P)>0; therefore the waiting time to the next mixed group is finite almost surely. Countably many such arguments imply infinitely many mixed events over an infinite run.

At mixed events, conditional on the current policy, the correct identity is sampled with probability q_c. Consequently the embedded chain over correct identities has weights

    w_c(N_c)=exp(z_c(0)+delta N_c),

where N_c counts how many mixed events selected mode c. This is an exponentially reinforced urn.

For a self-contained exponential-clock embedding, assign independent variables E_(c,n) with exponential rates w_c(n), and define T_c=sum_(n>=0) E_(c,n). Each T_c is finite almost surely since its expectation is exp(-z_c(0))/(1-exp(-delta)). The T_c are independent and have continuous distributions, so their minimum is almost surely unique. Order all per-color clock events before that minimum. Exponential memorylessness makes each next color have probability proportional to its current w_c, so this ordered process has exactly the urn law. The minimizing color has infinitely many events before its explosion time; every other color has only finitely many because its own explosion time is larger. Hence, from some finite embedded-event index on, only the winning correct mode is selected. Its logit tends to infinity, other correct logits eventually stop changing, and incorrect logits never increase. Thus P->1 and q->e_Cstar.

Each T_c has support (0,infinity), so Pr(T_c<min_d!=c T_d)>0 for every c. This also proves why the original deterministic claim about the initial winner cannot extend unchanged to noisy updates.

Care: the proof concerns correct identities appearing in mixed (nonzero-update) groups. Other correct modes can still appear in all-correct groups; it does not claim literal finite-time zero policy support. It does not cover G>2, decreasing eta, shared neural parameters, PPO epochs, or Adam. Exponential reinforcement is classical probability; the paper should cite an authoritative reinforced-urn/exponential-embedding source if using this specialization.

## 6. Existing manuscript claims needing repair before strengthening

1. `thm:replay-retention` now claims any bounded reward-measurable advantage, yet its proof uses Psi increasing and Psi(1) as an upper bound. This requires c_a>=0. For unrestricted signed c_a, replace Psi(1) by sup_(P in [0,1]) Psi(P). Retention still follows from boundedness of the potential, but the existing numerical bound is not generally justified.

2. Full- and partial-bank convergence to P=1 uses rho+c_a(P)P>0. That is not true for arbitrary signed c_a. Example: a_i=-R_i gives c_a=-1. Under full-bank replay with 0<rho<1, the probability-space objective is rho log P-P plus the conditional target term, and its optimum has P=rho, not P=1. Keep the convergence result within the positive-coefficient class or state the needed condition explicitly.

3. Having a negative Bernstein control point d_j does not imply that c_a(P) is negative anywhere. Nonnegative control points are a sufficient, not necessary, positivity certificate. The text saying the sign is settled by control-point signs “and by nothing else” is too strong when the signs are mixed. Some estimators outside the stated corollary still improve correctness everywhere and still collapse in the tabular mean model.

4. For an interior zero of c_a(P), a positive-correctness trajectory can stop at that root rather than traverse its orbit backwards. The prose that every excluded objective merely reverses correctness is too categorical. The formula remains true; the dynamical consequences need to distinguish positive, zero, and negative coefficients.

5. Claims in the geometry subsection and elsewhere that no fresh objective can raise conditional diversity or that geometry does not limit this conclusion require the single-prompt qualification. The multi-prompt counterexample above directly refutes a broader interpretation.

## 7. Highest-yield validation

- At a frozen neural checkpoint, compute prompt-specific average gradients for several binary estimators using enough groups to estimate the mean. Compare cosine alignment and signal-normalized covariance.
- Repeat with common-length weights and actual per-response length weights; measure the residual predicted by Section 3.
- Aggregate across prompts and quantify how unequal c_a(P_x) changes the aggregate direction.
- For the G=2 theorem, run categorical Monte Carlo and report the random winner probabilities as a function of initial imbalance and eta. This illustrates precisely why deterministic initial-winner predictions are not stochastic predictions.
- Keep these mechanism checks distinct from claims about production AdamW trajectories.
