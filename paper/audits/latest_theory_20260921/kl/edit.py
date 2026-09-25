from pathlib import Path
import difflib
out=Path(__file__).parent
whole=(out.parent/'before.tex').read_text()
a=whole.index(r'\subsection{Reference KL retains the reference, not the discovery}')
b=whole.index(r'\subsection{Two ceilings, and which one a domain sits under}',a)
s=whole[a:b]
(out/'before.tex').write_text(s)
changes=[]
def replace(old,new):
 global s
 assert s.count(old)==1, (s.count(old),old[:100])
 s=s.replace(old,new)
 changes.append({'old':old,'new':new})
replace(r'\subsection{Reference KL retains the reference, not the discovery}',r'\subsection{Reference-KL stationary targets and recovery}')
replace(r'''Assumption~\ref{ass:categorical-mean-flow}(A4) sets the reference-KL
coefficient to zero, which is what our runs train at
(Table~\ref{tab:run-contract}, $\beta_{\mathrm{KL}}=0$). The collapse theorem
therefore says nothing about the regularizer most RLVR recipes carry, and a
reader is entitled to ask whether that regularizer already does the work the
bank is introduced for. This subsection separates the categorical stationary target, exact-gradient
convergence, and recovery rates. For Dr.GRPO and MaxRL, the full-support
stationary conditional is the reference's own.
Corollary~\ref{cor:kl-dr-mei-convergence} applies entropy-regularized softmax
theory to prove convergence for exact Dr.GRPO gradient steps. The recovery
comparison below concerns a specified two-mode face: reference KL's
restoring logit force vanishes as a mode vanishes, whereas replay supplies
a nonvanishing banked-coordinate force.
App.~\ref{app:theory-kl-measured} compares these mechanisms with measured runs.

Replace (A4) by''',r'''Assumption~\ref{ass:categorical-mean-flow}(A4) sets the reference-KL
coefficient to zero, as in the default training setting of
Table~\ref{tab:run-contract}. A positive fixed-reference coefficient changes
the categorical target. For Dr.GRPO and centered MaxRL, the unique
full-support stationary policy allocates its correct mass in the same
proportions as the reference. Exact Dr.GRPO gradient steps converge to this
policy under Corollary~\ref{cor:kl-dr-mei-convergence}. On a two-correct-mode
face, reverse KL and replay also differ in recovery time: the reverse-KL
restoring logit force vanishes with the depleted probability, while replay's
banked-coordinate force has a positive limit. The measured KL comparison is
in App.~\ref{app:theory-kl-measured}.

For this analysis, replace (A4) by''')
replace(r'''The low-variance estimator standard in GRPO implementations is unbiased for
$\mathrm{KL}(p\,\|\,\mu)$ under samples drawn from $p$
\citep{shao2024deepseekmath}, so the reverse direction is the one to analyze;
the forward direction is treated at the end of this subsection and behaves
differently in exactly the way that matters. Write
$D(z)=\mathrm{KL}(p(z)\,\|\,\mu)$. The prompt-isolated categorical mean flow is''',r'''The KL estimator in GRPO has expectation $\mathrm{KL}(p\,\|\,\mu)$
under samples drawn from $p$ \citep{shao2024deepseekmath}. Unbiasedness of
this value does not imply that differentiating it at fixed samples gives an
unbiased gradient of the distributional KL. With the KL differentiated
exactly, write $D(z)=\mathrm{KL}(p(z)\,\|\,\mu)$. The prompt-isolated
categorical mean flow is''')
replace(r'''which is exact gradient flow of the potential
$F_\beta(z)=-\Psi(P(z))+\beta D(z)$, built exactly as
Equation~\eqref{eq:replay-potential} was built for verified replay.''',r'''which is the negative gradient flow of
$F_\beta(z)=-\Psi(P(z))+\beta D(z)$, with $\Psi'=c_G$ as in
Equation~\eqref{eq:replay-potential}.''')
replace(r'''This yields a floor when its right side is below the displayed face
threshold, but need not do so for an arbitrary initial energy. These are statements about loss sublevels. A particular optimizer can still
admit a trajectory-dependent probability floor, as
Corollary~\ref{cor:kl-dr-mei-convergence} establishes for exact Dr.GRPO
gradient steps with a fixed reference.''',r'''This energy bound excludes the face $p_b=0$ when its right side is below
$-\log(1-\mu_b)$; an arbitrary initial energy need not satisfy that condition.
A trajectory can have a positive probability floor even when this sublevel
bound does not establish one, as for the exact Dr.GRPO gradient steps in
Corollary~\ref{cor:kl-dr-mei-convergence}.''')
replace(r'''For Dr.GRPO, the constant fresh coefficient permits a stronger conclusion
by direct application of established optimization theory. The following
specializes \citet[Theorem~5 and Lemma~13]{mei2020softmaxpg}; its Gibbs
target is also the standard regularized maximizer of
\citet[Proposition~1]{geist2019regmdp}.''',r'''Dr.GRPO with exact reference KL is an entropy-regularized softmax bandit,
so \citet[Theorem~5 and Lemma~13]{mei2020softmaxpg} give convergence and a
positive trajectory-wide probability floor. Its Gibbs target is the
regularized maximizer of \citet[Proposition~1]{geist2019regmdp}.''')
replace(r'''Its score contains
an unbounded log ratio, so it is not an instance of the uniformly
bounded-score premise in Lemma~\ref{lem:sampled-score-bound}; both
comparisons nevertheless have a vanishing coordinate force.''',r'''The corresponding score multiplier contains
an unbounded log ratio, so Lemma~\ref{lem:sampled-score-bound}'s uniformly
bounded-multiplier premise does not apply. The exact KL force nevertheless
vanishes at this boundary.''')
replace(r'''Table~\ref{tab:kl-recovery} is a numerical illustration in a larger
categorical system, with $\MDklModes{}$ correct and three incorrect categories.
Its reference is uniform, its other depleted logits start at $-40$, and its
target is $p_b=\MDklTarget{}$. The KL times grow by a factor
\MDklDecadeKL{} per decade of initial depth; replay adds approximately
$\log(10)/\rho$ per decade. These finite-depth integrations illustrate the
recovery contrast but do not prove the two-mode asymptotic constants for
the larger system.''',r'''Table~\ref{tab:kl-recovery} integrates the exact Dr.GRPO mean field in a
system with $\MDklModes{}$ correct and three incorrect categories, with
$G=16$. The initial logits are $z_c=0$, $z_b=\log d$, and $-40$ for every
other category; thus $d=p_b(0)/p_c(0)$ is the initial odds ratio, rather than
$p_b(0)$ itself. The KL reference is uniform over all seven categories, and
the replay target is uniform over the four correct categories. Recovery ends
at $p_b=\MDklTarget{}$. Each tenfold decrease in $d$ multiplies the KL time
by \MDklDecadeKL{} and adds approximately $\log(10)/\rho$ to the replay
time. These finite-depth integrations illustrate the recovery contrast;
the asymptotic constants in Corollary~\ref{cor:kl-recovery-time} apply to
its specified two-mode face.''')
replace(r'''  \caption{\textbf{Time for a depleted correct mode to return to
  $p_b=\MDklTarget{}$, by the depth it starts from.} Numerical integration of
  the categorical mean flow with $m=\MDklModes{}$ correct modes, $G=16$, and a
  uniform reference, in the flow's own time units. Reference KL multiplies its
  recovery time by \MDklDecadeKL{} per decade of depth; verified replay adds
  approximately a constant per decade. Corollary~\ref{cor:kl-recovery-time}
  proves the contrasting asymptotic orders on a separate two-mode face.}''',r'''  \caption{\textbf{Replay recovery time grows approximately linearly with
  log-odds depletion.} Time to reach $p_b=\MDklTarget{}$ under the exact
  Dr.GRPO categorical mean flow, with $m=\MDklModes{}$ correct and three
  incorrect categories and $G=16$. Columns give initial odds
  $d=p_b(0)/p_c(0)$, with $z_c=0$, $z_b=\log d$, and all other logits $-40$.
  KL uses a uniform reference over all categories; replay uses a uniform
  target over correct categories. Each tenfold decrease in $d$ multiplies
  KL recovery time by \MDklDecadeKL{}, while replay adds approximately
  $\log(10)/\rho$. Times are numerical integrations in the flow's own units;
  the two-mode asymptotics are in Corollary~\ref{cor:kl-recovery-time}.}''')
replace(r'\textbf{Regularizer} & $s=10^{-3}$ & $10^{-4}$ & $10^{-5}$ & $10^{-6}$',r'\textbf{Regularizer} & $d=10^{-3}$ & $10^{-4}$ & $10^{-5}$ & $10^{-6}$')
replace(r'''\paragraph{The direction of the divergence decides whether it is a barrier.}
Forward KL to the same reference is a different object:
$\mathrm{KL}(\mu\,\|\,p)=-\sum_a\mu_a\log p_a-\mathcal H(\mu)$ is exactly the
weighted replay loss $R_w$ of
App.~\ref{app:theory-replay} with $w=\mu$, so it is unbounded at the
boundary and does supply the floor $p_a\ge\exp(-C_{T,\mu}/\mu_a)$. Three
differences remain, and they are the design of the bank. Its weights are the
reference's own probabilities, so a mode the base model rarely emits receives
a correspondingly weak floor, where uniform replay weights every admitted key
alike. Its support is all of $\mathcal A$, so it protects incorrect categories
on the same footing as correct ones, where the bank is verifier-filtered.
And it cannot be estimated from on-policy samples without importance weights or
draws from $\mu$, where the bank is a stored sample of the policy's own
verified history. Verified replay is the barrier construction with the target
replaced by a policy-generated, validator-accepted, uniformly weighted one.''',r'''\paragraph{Forward KL and cross-entropy barriers.}
Forward KL to the same reference satisfies
$\mathrm{KL}(\mu\,\|\,p)=-\sum_a\mu_a\log p_a-\mathcal H(\mu)$.
It differs from the weighted cross-entropy $R_\mu$ in
App.~\ref{app:theory-replay} by the constant $-\mathcal H(\mu)$ and diverges
when any coordinate vanishes. For coefficient $\beta>0$, descent of
$-\Psi(P)+\beta R_\mu$ therefore gives
$p_a(t)\ge\exp(-C_{T,\mu}/\mu_a)$, where
$C_{T,\mu}=R_\mu(z(T))+[\Psi_{\max}-\Psi(P(T))]/\beta$.
This is the weighted replay bound with target $w=\mu$.

The target distinguishes forward reference KL from verified replay. A small
reference mass $\mu_a$ weakens the coordinate floor through its exponent
$1/\mu_a$, while a uniform bank gives every stored solution mode equal weight.
A full-support reference protects incorrect categories as well as correct
ones; a verified bank assigns weight only to correct categories. Estimating
the forward expectation with draws from $p$ uses importance weights
$\mu_a/p_a$, while sampling from $\mu$ or a stored target gives the corresponding
cross-entropy expectation directly. Verified replay uses the latter
construction with a target supported on discovered verified modes.''')
replace(r'''Verified-only cross-entropy instead has the
proved limit $P\to1$ when $c_G\ge0$.''',r'''Cross-entropy over a fixed nonempty verified bank has the
limit $P\to1$ under the replay convergence assumptions with $c_G\ge0$.''')
replace(r'''Both coefficients therefore matter at a finite horizon. We have not swept
the replay dose empirically, so the finite-time comparison is a prediction of
the specified dynamics rather than a measured dose response.''',r'''Both coefficients therefore matter at a finite horizon. The replay-dose
dependence here is a prediction of the specified categorical dynamics; the
reported replay experiments do not estimate a dose-response curve.''')
replace(r'''Schedules, changing references, momentum, and shared
neural parameters require separate analysis.''',r'''Their assumptions exclude schedules, changing references, momentum, and
shared neural parameters.''')
replace(r'''The diversity axis this paper reports is
$\pmd{}=1-C(q)=1-\sum_{c\in\mathcal C}q_c^2$.''',r'''For the conditional solution mode distribution,
$\pmd{}=1-C(q)=1-\sum_{c\in\mathcal C}q_c^2$.''')
replace(r'''the frozen reference's own success-conditional diversity''',r'''the fixed reference's success-conditional diversity''')
replace(r'''The distinction between the reference allocation
and the discovered-bank target remains useful without that stronger claim.''',r'''These targets specify asymptotic or stationary allocations under the
respective categorical assumptions.''')
replace(r'''\paragraph{Two boundaries, and which instrument reaches each.}
The results in this appendix separate along the two boundaries of the simplex
that training actually visits, and they are reached by different means.

At $P\to0$ no verified success is available anywhere, and
Lemma~\ref{lem:replay-gradient-availability} bounds the probability of a
mixed group --- the only kind that carries fresh signal --- by $GP$. This is a
\emph{discovery} boundary, and a fresh objective is the instrument for it:
Corollary~\ref{cor:maxrl-starvation} gives MaxRL a factor approaching $G$
there, and Equation~\eqref{eq:maxrl-potential-finite-sum} identifies what it
is maximizing as a harmonic mixture of pass@$k$ for $k=1,\ldots,G-1$.''',r'''\paragraph{Scarce fresh successes and depleted discovered modes.}
As $P\to0$, fresh verified responses become rare. For the centered
estimators in Lemma~\ref{lem:replay-gradient-availability}, only mixed
reward groups supply a fresh task update, and their probability is at most
$GP$. Corollary~\ref{cor:maxrl-starvation} gives MaxRL a mean-gradient
magnitude ratio approaching $G$ relative to Dr.GRPO in this limit.
Its potential is a harmonic mixture of pass@$k$ for $k=1,\ldots,G-1$
(Equation~\eqref{eq:maxrl-potential-finite-sum}). This coefficient increase
does not change the probability of sampling a verified response at a fixed
policy. A previously populated bank can still supply verified targets when
current correctness is low.''')
replace(r'''Changing fresh advantages adds arithmetic to the existing update; replay adds
stored targets and gradient computation. Equation~\eqref{eq:replay-flop-ratio}
and App.~\ref{app:replay-cost} quantify that cost separately from the
mathematical comparison.''',r'''Replay also requires stored targets and additional gradient computation,
quantified in Equation~\eqref{eq:replay-flop-ratio} and
App.~\ref{app:replay-cost}.''')
(out/'replacement.tex').write_text(s)
(out/'changes.patch').write_text(''.join(difflib.unified_diff((out/'before.tex').read_text().splitlines(True),s.splitlines(True),fromfile='latest-KL-before.tex',tofile='KL-replacement.tex')))
print(f'Wrote {len(changes)} exact replacements in {len(s.splitlines())} lines.')
