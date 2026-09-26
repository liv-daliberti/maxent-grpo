# Replay theory audit and concrete strengthening

## Existing material (do not sell as new)

`paper/main.tex` already proves: categorical positive replay gradient at a disappearing banked coordinate; fixed-bank, fixed-dose energy barrier; full- and partial-bank convergence to replay weights (partial bank loses unbanked modes); length-normalized target weights proportional to inverse exemplar length; shared-parameter retention conditional on descent of a joint complete-exemplar potential; finite-draw visibility from probability floors; reference-KL stationary target and boundary/recovery comparisons. Merely adding a generic shared-parameter potential lemma, partial-bank caveat, or union-bound visibility statement would duplicate existing work.

## Priority 1: finite-time replay threshold and discrete contraction

This is a substantive new strengthening, directly derivable from existing assumptions, and better aligned with finite training than the exponentially tiny energy floor.

Let B be a fixed uniform bank of k>=2 correct categorical modes. Full correct-support coverage is NOT needed. Let P be total correctness, M=sum_{b in B} p_b, r_b=p_b/M, C=sum_B r_b^2. In the same mean-logit model,

    zdot_b = c(P) p_b (1-P) + rho(1/k-p_b),  b in B.

Define a(t)=rho-c(P)(1-P). Exact identities:

    d/dt log(p_i/p_j) = -a(t) M(t) (r_i-r_j)
    rdot_b = -a(t) M(t) r_b (r_b-C)
    Cdot = -2 a(t) M(t) [sum_B r_b^3 - C^2]
         = -2 a(t) M(t) Var_{b~r}(r_b).

Thus uniform replay reverses fresh concentration precisely when rho>c(P)(1-P). Below this threshold a nonuniform bank continues concentrating despite a positive replay term; equality freezes its conditional allocation. Every positive rho can have the same asymptotic target while doses differ sharply at a finite horizon. This supplies a much more defensible coefficient statement than “the coefficient only needs a sign.” For MaxRL, c(P) is larger than for Dr.GRPO at a given P, so finite-time within-bank restoration requires a larger dose even though faster discovery may enlarge the bank.

If a>=0 throughout [T,t], define exposure

    S(T,t)=integral_T^t a(u) M(u) du.

Let D(t)=log(max_B r_b/min_B r_b). Ordering is preserved. Since

    max r - min r >= (1-exp(-D))/k,

we have

    d/dt (exp(D)-1) <= -(a M/k) (exp(D)-1),

and consequently

    exp(D(t))-1 <= [exp(D(T))-1] exp(-S(T,t)/k).
    min_B r_b(t) >= 1/[1+(k-1)exp(D(t))].

For any delta in (0,1/k), set D_delta=log[(1/delta-1)/(k-1)]. A sufficient recovery exposure is

    S(T,t) >= k log[(exp(D(T))-1)/(exp(D_delta)-1)]

when the right side is positive. If additionally a>=a0>0 and M>=M0>0, divide by a0 M0 to obtain a wall-clock bound. This is logarithmic in an initially tiny bank-coordinate probability; unlike the energy floor it expresses useful recovery time. Bank mass M is explicit: the theorem does not silently assume that the bank already has all mass. It bounds conditional-on-bank probabilities; unconditional p_b>=M0 delta requires the mass premise.

Discrete exact-mean update:

    z_{n+1}=z_n+eta_n [c(P_n) grad_z P_n+rho(w-p_n)],
    gamma_n=eta_n a_n M_n in [0,2].

For i,j in B, d_{ij,next}=d_{ij}-gamma_n(r_i-r_j). Because r_i-r_j <= d_{ij}/2 for d_{ij}>=0, bank ordering is preserved. With E_n=exp(D_n)-1,

    E_{n+1} <= [1-(1-exp(-gamma_n))/k] E_n.

Proof: D_next=D-gamma(max r-min r); use
1-exp(-gamma x)>=(1-exp(-gamma))x for x in [0,1], followed by exp(D)(max r-min r)>=E/k. For gamma<=1,

    E_N <= E_0 exp[-(1-exp(-1))/k * sum_n gamma_n].

This is a genuine finite-step theorem (ordinary gradient descent on exact mean gradients), not merely changing the language around gradient flow. It still does not establish stochastic AdamW behavior. Equal uniform categorical weights are essential: the implemented unequal-length loss has inverse-length weights, so present the uniform theorem as the clean mechanism and state/measure the weighted discrepancy.

For nonuniform w the exact corresponding equation is

    rdot_b=r_b{rho[w_b-<r,w>]-a M[r_b-C]},

which should be displayed rather than pretending the transient target is uniform. On the full bank face P=M=1 this is the weighted cross-entropy dynamics with target w. This equation can guide the weighted extension and controlled equal-length ablation.

## Priority 2: a neural optimizer response theorem that can be measured

The existing joint-potential lemma permits shared parameters only by assuming descent. It does not explain whether replay's force survives neural gradient interference or the actual optimizer. A useful next theorem is local and conditional, with conditions directly logged.

For a complete bank exemplar, let ell_b(theta)=log pi_theta(e_b|x_b)/L_b, h_b=grad ell_b, r=(1/k)sum_j h_j. Consider a preconditioned step Delta=eta H(g_fresh+lambda r)+u, where H is positive semidefinite and u contains momentum/weight-decay/other residual contributions as appropriate. If ell_b has Hessian norm at most B_b along the step, Taylor's theorem gives

    ell_b(theta+Delta)-ell_b(theta)
       >= eta h_b^T H g_fresh
          + eta lambda/k sum_j h_b^T H h_j
          + h_b^T u - (B_b/2)||Delta||^2.

The replay response coefficient is kappa_b=(1/k)sum_j K_bj with K_bj=h_b^T H h_j. Positive diagonal K_bb alone is insufficient; row sums can be negative. If lambda*kappa_b exceeds adverse fresh/momentum drift plus curvature, the stored response improves on this update even when absent from fresh rollouts. This is the precise bridge between signal availability and effect.

For AdamW, do not claim H fixed when replay changes the second moment. Either use the actual realized Delta directly in the Taylor bound, derive a frozen-preconditioner conditional claim, or compare replay-on/off cloned optimizer updates at the same checkpoint and record h_b^T(Delta_on-Delta_off). A bounded Hessian assumption can be checked approximately by a step-size sweep, but empirical agreement is not a uniform proof.

A simple necessary limitation: take one-parameter logits with slopes (1,-2,0), two banked modes each currently probability .001, and an unbanked category with .998. Their scores are approximately 1.001 and -1.999. Uniform replay has mean score about -.499, so it initially *decreases* the first rare bank exemplar's likelihood. Thus no theorem can say every rare banked mode is reinforced under arbitrary shared neural geometry. A kernel/interference condition is scientifically necessary.

Suggested low-cost check: save checkpoints; clone model and optimizer state; obtain fresh-only and replay-on one-step updates at several replay doses; measure actual complete-exemplar score changes, row-sum Gram responses, optimizer residual, and whether each exemplar was in the fresh group. This directly tests the missing implementation link, whereas an additional asymptotic tabular plot does not. Exemplar score increases must remain distinguished from whole-key probability changes.

## Priority 3: changing banks and scheduling, with limited ambitions

The implemented bank grows and its round-robin return time changes. Finite capacity/no eviction means finitely many bank additions, so eventual stabilization is plausible but does not give a useful finite training guarantee. A useful finite-window analysis would explicitly include maximum return gap and accumulated adverse drift between visits.

For arbitrary categorical update Delta z, the elementary bound

    log p_b(next)-log p_b >= -osc(Delta z)

implies that across a gap with |Delta z_a|<=eta A, p_b cannot fall by more than exp[-2A sum_gap eta]. Combine this with a replay-visit restoration bound to state a dose-versus-revisit-frequency requirement. For shared parameters use the preceding local score response with accumulated residual terms. This is more directly related to the scheduler than simply assuming a constant effective rho.

An exact changing-bank energy extension also exists but probably is not worth a main result: with E=R_sum-(k/rho)Psi(P), E descends between admissions and an admission j jumps it by -log p_j(entry)-Psi(P_entry)/rho. Thus an old bank coordinate has a floor controlled by the cumulative admission surprisals and objective range. This is typically as numerically vacuous as the existing floor; prioritize the finite-rate theorem and optimizer-response test instead.

## Repairs required before adding claims

1. “Any bounded reward-measurable objective” in the replay retention theorem is broader than its proof. The proof uses Psi increasing. Use sup_{s in [0,1]}Psi(s) in the barrier constant for arbitrary bounded c_a; require c_a>=0 (or rho+P c_a(P)>0) for convergence to P=1. Counterexample: c=-kappa, kappa>rho, full uniform correct bank plus one failure has stationary P=rho/kappa and p_b=P/k. Current C_T=-log(P/k)-kappa(1-P)/rho can even be negative, making the stated floor exceed 1.

2. KL recovery has a real factor error and an order-label error. On the two-correct-mode face,

    sdot=2 beta s^2(1-s)^2 log[((1-s)mu_b)/(s mu_c)],

so T_KL(s)~1/[2 beta s log(1/s)], not the displayed coefficient 1. The proof's O(s log(1/s)) remainder is the same order as its supposed leading term and cannot justify a 1+o(1) constant. The title Theta(1/s) is also false in ordinary Theta notation; use Theta(1/[s log(1/s)]). General multi-mode constants/path assumptions require a separate proof. Replay on this face has sdot=rho s(1-s)(1-2s) and T~log(1/s)/rho.

3. “A bounded penalty supplies no probability floor” is too broad. For a reference mu, inf_{p:p_b=0} KL(p||mu)=-log(1-mu_b)>0. A sufficiently low KL sublevel excludes that face, hence DOES supply a floor. The correct assertion is that an arbitrary finite reverse-KL sublevel need not exclude the boundary, whereas every finite forward-cross-entropy sublevel does. Example: if all categories are correct, P=1 and initial D0<-log(1-mu_b), KL descent excludes p_b=0. Do not infer impossibility of convergence/floor from this particular loose energy bound.

4. The paper's categorical inverse-length target qualification must accompany main-text language about uniform learned modes and attainable ceilings. Replaying equal mean-token scores does not in general target uniform key probabilities, and a single stored string is only a subset of its key event.

## Suggested paper emphasis

Main theory narrative: binary-only objectives share a conditional mean direction; replay adds a force with an explicit finite-time balance against that direction; whether that force improves neural exemplar likelihood is governed by measured gradient interference and optimizer response. Keep standard replicator specialization as context. Promote a finite-time/discrete replay theorem, and validate its operative quantities. Do not turn the existing asymptotic floor into a practical guarantee by wording alone.
