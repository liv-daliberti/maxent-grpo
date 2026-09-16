# Independent review: finite-step categorical replay

**Verdict: the proposed deterministic finite-step extension is correct under the assumptions below.** It extends the retention and asymptotic probability conclusions beyond continuous gradient flow, including a proper subset of correct actions in the bank. It does not cover Adam, stochastic group updates, clipping, changing replay banks, or neural parameter sharing. This is an independent derivation; numerical checks are supplementary evidence, not its proof.

Let the action set be finite, let the nonempty bank satisfy \(B\subseteq C\), and let \(w_b>0\) with \(\sum_{b\in B}w_b=1\); extend \(w\) by zero outside the bank. Start from finite independent logits \(z_0\), define \(p=\operatorname{softmax}(z)\), \(P=\sum_{i\in C}p_i\), and use the ordinary Euclidean gradient in these logits. Assume fixed \(\rho>0\), \(\Psi\in C^2([0,1])\), and \(\Psi'\geq0\). Set

\[
R_w(z)=-\sum_{b\in B}w_b\log p_b(z),\qquad
F(z)=\rho R_w(z)-\Psi(P(z)),
\]
\[
A=\sup_{[0,1]}|\Psi'|,\qquad
B_2=\sup_{[0,1]}|\Psi''|,\qquad
L=\rho/2+A/2+B_2/8.
\]

The derivative bound is called \(B_2\) here to avoid confusing it with the replay bank \(B\).

**Global smoothness.** Since the bank weights sum to one,
\(R_w=\log\sum_i e^{z_i}-\sum_iw_iz_i\), so its Hessian is the categorical covariance matrix \(J=\operatorname{diag}(p)-pp^\top\). For any vector \(u\),

\[
u^\top Ju=\operatorname{Var}_p(u_I)
\leq\frac{(\max_i u_i-\min_i u_i)^2}{4}
\leq\frac{\|u\|_2^2}{2}.
\]

The first inequality follows by bounding the variance of a variable in an interval; the second follows from Cauchy–Schwarz applied to the difference of two coordinates. Thus \(\|\nabla^2R_w\|_{\rm op}\leq1/2\). Differentiating the tilted categorical expectation \(P(z+tu)\) twice gives the exact identity

\[
u^\top\nabla^2P\,u
=\mathbb E_p[(\mathbf1_{\{I\in C\}}-P)(u_I-\mathbb E_pu_I)^2].
\]

Its absolute value is at most \(\operatorname{Var}_p(u_I)\), hence \(\|\nabla^2P\|_{\rm op}\leq1/2\). Also,

\[
\|\nabla P\|_2^2
=(1-P)^2\sum_{i\in C}p_i^2+P^2\sum_{i\notin C}p_i^2
\leq2P^2(1-P)^2\leq1/8.
\]

The chain rule
\(\nabla^2F=\rho J-\Psi'(P)\nabla^2P-\Psi''(P)\nabla P\nabla P^\top\)
therefore yields the stated global operator-norm bound \(L\). No bounded-logit assumption is needed.

**Descent and the retention floor.** For a fixed step size \(0<\eta<2/L\), exact gradient descent obeys

\[
z_{n+1}=z_n-\eta\nabla F(z_n),\qquad
F(z_{n+1})\leq F(z_n)-\eta(1-\eta L/2)\|\nabla F(z_n)\|_2^2.
\]

All finite iterates have finite logits and positive probabilities. Monotonicity of \(\Psi\) and energy descent imply

\[
R_w(z_n)\leq C_0:=R_w(z_0)+
\frac{\Psi(1)-\Psi(P(z_0))}{\rho},\qquad
p_b(z_n)\geq\exp(-C_0/w_b)>0.
\]

Each term \(-w_b\log p_b\) is nonnegative, which justifies extracting the individual floor from the total loss. It holds uniformly for every iterate, for any nonempty fixed bank; full coverage of \(C\) is unnecessary for retaining banked actions. The floor can nevertheless be extremely small.

Since \(F\geq-\Psi(1)\), telescoping descent gives
\(\sum_{n\geq0}\|\nabla F(z_n)\|_2^2<\infty\), and therefore \(\nabla F(z_n)\to0\). One explicit finite-iteration consequence is

\[
\min_{0\leq n<N}\|\nabla F(z_n)\|_2^2
\leq\frac{\rho C_0}{\eta(1-\eta L/2)N}.
\]

This is a bound on the best gradient norm, not a finite-iteration convergence rate for probabilities or semantic breadth.

**The probability limit, including a partial bank.** Extend the gradient continuously to the closed simplex by

\[
g_i(p)=\rho(p_i-w_i)-\Psi'(P)p_i(\mathbf1_{\{i\in C\}}-P).
\]

Compactness gives a convergent subsequence of any probability sequence; every such subsequential limit \(p_*\) satisfies \(g(p_*)=0\). For an incorrect action,
\(g_i=(\rho+\Psi'(P)P)p_i\). Its coefficient is strictly positive, so every incorrect coordinate of \(p_*\) is zero and \(P_*=1\). For a correct action, the remaining equation is \(g_i=\rho(p_i-w_i)=0\). Hence the only possible probability limit is the vector \(w\), extended by zero off the bank. Uniqueness of the subsequential limit in a compact space proves

\[
p_b(z_n)\longrightarrow w_b\quad(b\in B),\qquad
p_i(z_n)\longrightarrow0\quad(i\notin B).
\]

This does not assert convergence to finite logits: logits may diverge as off-bank probabilities vanish. A finite fixed bank thus preserves its own identities but asymptotically excludes unbanked correct actions in this particular global categorical model. That exclusion is not a universal neural or changing-bank claim.

**Constants for the two specified mean fields.** For Dr.GRPO with group size \(G\geq2\), \(\Psi(P)=(G-1)P/G\), so \(A=(G-1)/G\), \(B_2=0\), and \(L=\rho/2+(G-1)/(2G)\). For the implemented centered MaxRL estimator, which drops the all-incorrect group,

\[
\Psi'(P)=\sum_{j=0}^{G-2}(1-P)^j,
\quad A=G-1,
\quad B_2=\sum_{j=1}^{G-2}j=(G-1)(G-2)/2,
\]
\[
L=\rho/2+(G-1)(G+6)/16.
\]

The bounds include \(G=2\), where \(\Psi'=1\) and \(\Psi''=0\). They correspond to the implemented \(G-1\) mean potential, not the distinct order-\(G\) success-only estimator. A common fixed loss normalization must be absorbed into the actual coefficients before applying these constants. For inverse-length weighted replay, normalize the weights and use its effective \(\rho\); a changing bank or update frequency is outside this autonomous objective.

The strict step-size range is a sufficient condition, not a necessary or sharp threshold. Neither the continuous-time result nor this exact deterministic gradient-descent extension establishes a guarantee for the stochastic, Adam-updated, clipped, token-level neural experiments.

Supplementary numerical evidence is in the independently written [verifier](verify_discrete_replay_20260905.py) and [results](verify_discrete_replay_20260905.json). Its 36 trajectories cover Dr.GRPO/implemented MaxRL, \(G=2,3,16\), unequal weights, full/partial banks, and step sizes \(0.1,0.5,0.95\) times \(2/L\). All 20,000-step trajectories satisfy the checked descent inequality and retention floor. Maximum gradient and Hessian-vector finite-difference errors are approximately \(3.42\times10^{-10}\) and \(1.80\times10^{-10}\). These checks support implementation consistency; the argument above supplies the mathematical guarantee.
