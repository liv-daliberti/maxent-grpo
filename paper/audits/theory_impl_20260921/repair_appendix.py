"""Apply the reviewed corrections within Appendix P only."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[3]
path = root / "paper/main.tex"
original = path.read_text()
start = original.index(r"\section{Mode Collapse and Verified Replay}")
end = original.index(r"\section{What Verified Replay Costs}", start)
prefix, text, suffix = original[:start], original[start:end], original[end:]


def replace(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new, 1)


def span(first, last, new):
    global text
    a = text.index(first)
    b = text.index(last, a)
    text = text[:a] + new + "\n\n" + text[b:]


replace(
    "Every result in this appendix is stated under one model, fixed here once.",
    "The categorical results use the following model. The neural score identities,\n"
    "finite-step updates, and sampled-update results below state separately which\n"
    "assumptions they replace; none identifies a neural mode with an independently\n"
    "optimized logit.")
replace(
    r"\subsection{No outcome-binary objective can change which mode wins}",
    r"\subsection{Conditional trajectories of outcome-binary mean flows}")
replace(
    "Under\nAssumption~\\ref{ass:categorical-mean-flow} the entire class of group-centered\n"
    "objectives built from binary verifier outcomes induces the same conditional\n"
    "trajectory, and differs only in how fast that trajectory is traversed.",
    "Under\nAssumption~\\ref{ass:categorical-mean-flow}, outcome-binary estimators with\n"
    "positive correctness coefficients induce the same conditional trajectory,\n"
    "differing only in how fast it is traversed.")
replace(
    "$\\min_jd_j$ and $\\max_jd_j$ at every $P$, so the sign of the coefficient is\n"
    "settled by the signs of these $G$ numbers and by nothing else.",
    "$\\min_jd_j$ and $\\max_jd_j$ at every $P$. Nonnegative control points,\n"
    "with at least one positive, therefore certify positivity in the interior.\n"
    "This is a sufficient condition: mixed signs alone do not determine whether\n"
    "$c_a$ is positive throughout the interval.")
replace(
    r"\begin{corollary}[Every outcome-binary objective collapses]",
    r"\begin{corollary}[Collapse for correctness-improving outcome-binary mean flows]")
span(
    "This closes the escape route rather than leaving it open.",
    r"\begin{remark}[Group size is not a remedy]",
    r"""The nonnegative-gap condition is sufficient rather than necessary. Some
estimators with mixed-sign gaps still have $c_a(P)>0$ throughout the interior
and satisfy the same conclusion. If $c_a$ has an interior zero, the mean flow
can stop there; if it is negative, correctness decreases locally.
Theorem~\ref{thm:objective-inertness} identifies the direction in all cases,
but the collapse conclusion requires the stated positivity and initial
imbalance. It does not describe shared-prompt or sampled neural updates.""")
replace(
    r"\begin{remark}[Group size is not a remedy]",
    r"\begin{remark}[Group size in the deterministic conditional flow]")
replace(
    "Drawing more samples per prompt is\ntherefore an answer to the first of those problems and not to the second one.",
    "These are mean-flow statements. Group size also changes gradient covariance\n"
    "and finite-sample symmetry breaking, as the sampled-update results below\n"
    "make explicit; the deterministic equivalence is not a stochastic invariance.")

# Retention remains valid for arbitrary bounded coefficients, with the actual
# supremum of the correctness potential. Convergence needs a sign condition.
replace(
    "Theorem~\\ref{thm:objective-inertness}. Under~\\eqref{eq:replay-mean-flow}, define\n\\[\n  C_T",
    "Theorem~\\ref{thm:objective-inertness}. Set\n"
    "$\\Psi_{\\max}=\\max_{s\\in[0,1]}\\Psi(s)$, which is finite, and under\n"
    "\\eqref{eq:replay-mean-flow} define\n\\[\n  C_T")
text = text.replace(r"\Psi(1)-\Psi(P(T))", r"\Psi_{\max}-\Psi(P(T))")
replace(
    "Since $P(t)\\le1$ and $\\Psi$ is increasing,",
    "Since $\\Psi(P(t))\\le\\Psi_{\\max}$,")
replace(
    "The floor is qualitative and can be numerically vacuous.",
    "For the nonnegative coefficients of the named fresh estimators,\n"
    "$\\Psi_{\\max}=\\Psi(1)$, recovering the simpler constant. Signed coefficients\n"
    "also satisfy the retention bound, but need not yield $P\\to1$.\n"
    "The floor is qualitative and can be numerically vacuous.")
replace(
    "$T$ the bank covers the full correct support, $\\mathcal B=\\mathcal C$. Let",
    "$T$ the bank covers the full correct support, $\\mathcal B=\\mathcal C$.\n"
    "Assume additionally $c_G(P)\\ge0$ on $[0,1]$, as for the three named\n"
    "estimators. Let")
replace(
    "The potential is bounded below by $-\\Psi(1)$",
    "The potential is bounded below by $-\\Psi_{\\max}$")
replace(
    "If $\\mathcal B=\\mathcal C$, then\n$R_w=H(w)-\\log P+\\mathrm{KL}(w\\|q)$",
    "For the following convergence statements assume $c_G\\ge0$.\n"
    "If $\\mathcal B=\\mathcal C$, then\n$R_w=H(w)-\\log P+\\mathrm{KL}(w\\|q)$")
replace(
    "$J_{\\max}=\\sum_x\\nu_x\\Psi_x(1)$",
    "$J_{\\max}=\\sum_x\\nu_x\\max_{s\\in[0,1]}\\Psi_x(s)$")

span(
    "The\nanswer is not that reference KL fails to prevent winner-take-all",
    r"\begin{itemize}",
    r"""The
comparison separates stationary targets from recovery rates. For the
Dr.GRPO and MaxRL coefficients, the full-support stationary conditional is
the reference's own. Its restoring logit force vanishes as a mode vanishes,
whereas replay supplies a nonvanishing banked-coordinate force.
The recovery comparison below is proved on a specified two-mode face; it is
not a convergence theorem for all full-support trajectories.
App.~\ref{app:theory-kl-measured} compares these mechanisms with measured runs.

Replace (A4) by""")
span(
    r"\begin{lemma}[A bounded penalty supplies no probability floor]",
    r"\begin{proposition}[The retained conditional is the reference's]",
    r"""\begin{lemma}[Reverse-KL sublevels and replay barriers]
\label{lem:kl-no-barrier}
For a full-support reference,
$D=\mathrm{KL}(p\|\mu)\le\log(1/\min_a\mu_a)$ on the simplex, while
$R_{\mathcal B}\to\infty$ if a banked coordinate vanishes. More precisely,
\[
 \inf_{p:p_b=0}D(p)=\log\frac1{1-\mu_b}.
\]
Thus every finite replay-loss sublevel bounds each protected probability
away from zero. A reverse-KL sublevel $\{D\le B\}$ does so for coordinate $b$
if $B<-\log(1-\mu_b)$, but includes a point with $p_b=0$ if
$B\ge-\log(1-\mu_b)$.
\end{lemma}
\begin{proof}
The global bound follows from $\sum_ap_a\log p_a\le0$ and
$-\sum_ap_a\log\mu_a\le\log(1/\min_a\mu_a)$.
On the face $p_b=0$, write $\widetilde\mu_a=\mu_a/(1-\mu_b)$ for $a\ne b$.
Then $D(p)=\mathrm{KL}(p\|\widetilde\mu)-\log(1-\mu_b)$, whose minimum is
attained at $\widetilde\mu$. Continuity and compactness exclude that face
from any strictly lower sublevel and give a positive coordinate minimum.
For replay, $-\log p_b\le kR_{\mathcal B}$ gives the bound directly.
\end{proof}

Descent of $F_\beta$ gives
$D(p(t))\le [F_\beta(z(T))+\Psi_{\max}]/\beta$.
This yields a floor when its right side is below the displayed face
threshold, but need not do so for an arbitrary initial energy. The distinction
is which sublevels exclude the boundary, not an impossibility of retention
under a bounded penalty.""")
replace(
    "Under Assumption~\\ref{ass:categorical-mean-flow} with (A4$'$), the flow\n"
    "\\eqref{eq:kl-mean-flow} has exactly one stationary point of full support,",
    "Under Assumption~\\ref{ass:categorical-mean-flow} with (A4$'$), using the\n"
    "Dr.GRPO or centered MaxRL coefficient, the flow\n"
    "\\eqref{eq:kl-mean-flow} has exactly one stationary point of full support,")
span(
    "Read against Corollary~\\ref{cor:replay-no-collapse}, this is the substantive",
    r"\begin{lemma}[Reference KL has no probability-independent boundary force]",
    r"""The stationary target differs from full-coverage uniform replay:
Corollary~\ref{cor:replay-no-collapse} gives $q\to u$, whereas the KL
stationary allocation is $\mu|_{\mathcal C}$. A broader reference changes that
allocation; replay instead constructs its target from verified discoveries.
The proposition does not bound transient diversity by the reference's
diversity or establish convergence of every trajectory to this stationary
point, and it does not apply directly to neural optimizer trajectories.

\begin{remark}
The correctness prediction also depends on reference correctness
$s_\mu=\mu(\mathcal C)$:
$1-P^\star=(1-s_\mu)/(1-s_\mu+s_\mu e^\lambda)$.
For Dr.GRPO with $G=16$ and $\beta\le.1$, $\lambda\ge9.375$;
this amplifies the reference correctness odds by at least $e^{9.375}$ but
does not give a reference-independent bound on the correctness deficit.
\end{remark}""")
replace(
    "Reference KL therefore belongs to the class described by\n"
    "Lemma~\\ref{lem:sampled-score-bound}, with the difference that this is a\n"
    "property of the exactly evaluated penalty rather than of a sampled estimator.",
    "This is a property of the exactly evaluated penalty. Its score contains\n"
    "an unbounded log ratio, so it is not an instance of the uniformly\n"
    "bounded-score premise in Lemma~\\ref{lem:sampled-score-bound}; both\n"
    "comparisons nevertheless have a vanishing coordinate force.")
span(
    r"\begin{corollary}[Recovery from depth $s$:",
    r"\begin{table}[t]",
    r"""\begin{corollary}[Recovery on a two-correct-mode face]
\label{cor:kl-recovery-time}
Consider the continuous extension of the categorical probability flow to
the face supported on two correct modes $b,c$, with $p_b=s$, $p_c=1-s$.
This boundary reduction replaces the full-support start in (A5); here $P=1$
and the fresh correctness gradient is zero. For reference KL take fixed
$\beta>0$ and $\mu_b,\mu_c>0$. For replay take the uniform bank $\{b,c\}$
and fixed $\rho>0$. Fix
$0<\delta<\min\{1/2,\mu_b/(\mu_b+\mu_c)\}$.
The times to reach $p_b=\delta$ from $s\downarrow0$ satisfy
\[
 T_{\mathrm{KL}}(s)
 =\frac{1+o(1)}{2\beta s\log(1/s)},\qquad
 T_{\mathrm{replay}}(s)
 =\frac{\log(\delta/s)}{\rho}+O(1).
\]
In particular their ratio diverges. The KL order is
$\Theta(1/[s\log(1/s)])$, and the replay order is
$\Theta(\log(1/s))$.
\end{corollary}
\begin{proof}
Write $u=p_b$ along this face. The softmax identity
$\dot u=u(1-u)(\dot z_b-\dot z_c)$ gives exactly
\[
 \dot u_{\mathrm{KL}}
 =2\beta u^2(1-u)^2\log\frac{(1-u)\mu_b}{u\mu_c},
 \qquad
 \dot u_{\mathrm{replay}}
 =\rho u(1-u)(1-2u).
\]
Both velocities are positive up to $\delta$ by its stated restriction.
Separating variables expresses each hitting time as the integral from
$s$ to $\delta$ of the reciprocal velocity. For KL the integrand is
$(1+o(1))/(2\beta u^2\log(1/u))$ as $u\downarrow0$, and its integral is
asymptotic to $1/(2\beta s\log(1/s))$ (equivalently, apply
l'Hopital's rule to the two divergent functions). For replay the integrand
is $1/(\rho u)+O(1)$, giving the second expression.
\end{proof}

Table~\ref{tab:kl-recovery} is a numerical illustration in a larger
categorical system, with $\MDklModes{}$ correct and three incorrect categories.
Its reference is uniform, its other depleted logits start at $-40$, and its
target is $p_b=\MDklTarget{}$. The KL times grow by a factor
\MDklDecadeKL{} per decade of initial depth; replay adds approximately
$\log(10)/\rho$ per decade. These finite-depth integrations illustrate the
recovery contrast but do not prove the two-mode asymptotic constants for
the larger system.""")
replace(
    "recovery time by \\MDklDecadeKL{} per decade of depth, matching the\n"
    "  $\\Theta(1/s)$ of Corollary~\\ref{cor:kl-recovery-time}; verified replay adds\n"
    "  a constant per decade, matching $\\Theta(\\log(1/s))$. The coefficients scale\n"
    "  the constants and not the exponents.",
    "recovery time by \\MDklDecadeKL{} per decade of depth; verified replay adds\n"
    "  approximately a constant per decade. Corollary~\\ref{cor:kl-recovery-time}\n"
    "  proves the contrasting asymptotic orders on a separate two-mode face.")
span(
    r"\paragraph{One coefficient has a window; the other only has a sign.}",
    r"\begin{corollary}[The three limits, in the reported diversity metric]",
    r"""\paragraph{Stationary targets and finite training budgets.}
Full-coverage uniform replay has the same categorical limit for every
$\rho>0$, but this does not make dose selection irrelevant.
The finite-time analysis above compares replay with the instantaneous
fresh concentration pressure $c_G(P)(1-P)$. Small doses can allow diversity
to decline before the asymptotic target is approached.

For reference KL, increasing $\beta$ changes stationary correctness through
$\operatorname{logit}P^\star=\operatorname{logit}\mu(\mathcal C)+c_G(P^\star)/\beta$.
Its boundary recovery time also depends on $\beta$; on the two-mode face
the leading scale is $1/[2\beta s\log(1/s)]$.
Both coefficients therefore matter at a finite horizon. We have not swept
the replay dose empirically, so the finite-time comparison is a prediction of
the specified dynamics rather than a measured dose response.

\paragraph{Scope of the comparison.}
The stationary and rate statements concern independently optimized
categorical logits. Schedules, changing references, momentum, and shared
neural parameters require separate analysis. The implemented sampled KL
term also differs from an exactly differentiated distributional KL:
an absent response supplies no direct sampled score, although normalization
and shared parameters can still change its probability.
Reference KL has guarantees for regularized control and proximity to a
reference \citep{geist2019regmdp,vieillard2020kl}. The results here distinguish
its stationary target and boundary force from verified replay; they do not
exclude useful finite-time diversity preservation by KL.""")
replace(
    r"\begin{corollary}[The three limits, in the reported diversity metric]",
    r"\begin{corollary}[Two limits and a KL stationary value in the diversity metric]")
replace(
    "bounded outcome-binary objective alone sends $\\pmd{}\\to0$\n"
    "(Corollary~\\ref{cor:class-collapse}),",
    "bounded outcome-binary objective satisfying the positivity and unique-maximum\n"
    "hypotheses of Corollary~\\ref{cor:class-collapse} sends $\\pmd{}\\to0$,")
replace(
    "Adding a reference KL instead leaves\nthe flow with a single interior stationary point,",
    "With the Dr.GRPO or centered MaxRL coefficient, adding a reference KL\n"
    "instead gives a single interior stationary point,")
span(
    "The difference in what the three statements assert is itself a result",
    r"\paragraph{Two boundaries, and which instrument reaches each.}",
    r"""The first two statements are convergence results under their stated
hypotheses; the third identifies an interior stationary point.
Lemma~\ref{lem:kl-no-barrier} explains when a KL sublevel excludes a face,
but a general convergence proof is not supplied here.
Neither the stationary value nor the two-mode recovery rate is a bound on
transient neural diversity. The distinction between the reference allocation
and the discovered-bank target remains useful without that stronger claim.""")
span(
    "At $p_b\\to0$ for an already-discovered correct mode,",
    r"\subsection{Two ceilings, and which one a domain sits under}",
    r"""At $p_b\to0$ for a discovered correct mode, the bounded sampled-coordinate
force vanishes, while verified replay contributes $\rho(w_b-p_b)$.
The reference-KL force also vanishes, at a different rate.
These are coordinate-level comparisons in the stated geometry: they do not
exclude retention through parameter sharing or other update rules.

The fresh and replay terms compose into
$F=\rho R_{\mathcal B}-\Psi_{\mathrm{MaxRL}}(P)$.
Its descent gives the fixed-bank retention guarantee, and the finite-time
analysis quantifies when replay overcomes concentration. Their gradients can
oppose one another, so composition does not imply an absence of interference.
Changing fresh advantages adds arithmetic to the existing update; replay adds
stored targets and gradient computation. Equation~\eqref{eq:replay-flop-ratio}
and App.~\ref{app:replay-cost} quantify that cost separately from the
mathematical comparison.""")

# Downstream interpretation must not turn model-specific statements into
# empirical causal or transient guarantees.
replace(
    "Corollary~\\ref{cor:class-collapse} is confirmed without qualification.",
    "The observed concentration is consistent with the categorical mechanism\n"
    "in Corollary~\\ref{cor:class-collapse}, although neural training does not\n"
    "satisfy that corollary's update assumptions.")
span(
    "Both misses have one cause:",
    r"\paragraph{Two ceilings, not one comparison.}",
    r"""These discrepancies show that the stationary categorical predictions do
not quantitatively describe the measured checkpoints. Finite training,
shared parameters, and the sampled adaptive optimizer are all possible
sources of the difference; this comparison does not identify their separate
effects. The boundary-force identities and the two-mode
$\Theta(1/[s\log(1/s)])$ versus $\Theta(\log(1/s))$ recovery comparison
motivate mechanism checks, rather than certify the measured neural rates.""")
replace(
    "An anchor can carry \\pmd{} to the reference's own conditional diversity and no\n"
    "further, by Proposition~\\ref{prop:kl-stationary}.",
    "The interior KL stationary value equals the reference's conditional\n"
    "diversity by Proposition~\\ref{prop:kl-stationary}; it is not a transient\n"
    "upper bound for the trained neural policy.")
span(
    "This is the composition argument of App.~\\ref{app:theory-objective-inertness}",
    r"\paragraph{Consequences for the comparison.}",
    r"""These measurements are consistent with separating fresh learning and
memory. The per-prompt mean-direction identity does not itself predict the
shared neural trajectory: MaxRL alone reaches $.148$ PCMD in Graph and at
or below $.022$ elsewhere, no better than Dr.GRPO alone in four domains.
The larger Python bank motivates the discovery interpretation, but does not
isolate it from prompt weighting, stochasticity, or optimizer effects.""")
replace(
    "range has no counterpart in a replay dose.",
    "range concerns a stationary target tradeoff, whereas the replay dose also\n"
    "controls finite-time restoration. A matched replay-dose sweep would be\n"
    "needed to compare those sensitivities empirically.")
replace(
    "It should be put plainly. A reference KL prevents winner-take-all for every\n"
    "$\\beta>0$ in the flow, needs no bank, no admission rule and no capacity, stores\n"
    "nothing between updates, and adds one term to a loss most implementations\n"
    "already carry.",
    "For every $\\beta>0$, the analyzed reference-KL flow has a full-support\n"
    "stationary point. It needs no replay bank, admission rule, or bank capacity,\n"
    "although it retains a reference policy, and adds a familiar loss term.")
replace(
    "Replay asks for a positive\nnumber, by Corollary~\\ref{cor:replay-no-collapse}.",
    "Replay has the same categorical limit for every positive dose, but its\n"
    "finite-time restoration still depends on the dose relative to fresh\n"
    "concentration pressure.")
replace(
    "coefficient supplies there is the finite-horizon brake of\n"
    "Corollary~\\ref{cor:kl-recovery-time} and not the fixed point of\n"
    "Proposition~\\ref{prop:kl-stationary}.",
    "coefficient supplies there cannot be inferred from the stationary\n"
    "conditional formula. Zero observed successes also does not establish\n"
    "zero population correctness.")
span(
    "This caveat is specific to the collapse conclusion,",
    "A bounded advantage based on fresh samples has a different boundary behavior.",
    r"""The mean-gradient identity has broader scope than the collapse conclusion.
For one prompt and a common autonomous preconditioner $M(\theta)$, positive
coefficients give flows $c_a(P)M(\theta)\nabla_\theta P$ that follow the same
orbit after a change of time. Outcome-dependent covariance, response-length
weights, shared prompts, and optimizer memory can break the corresponding
training-trajectory equivalence, as the preceding neural analysis shows.
Parameter count alone does not identify this geometry. The empirical scale
comparisons change pretrained models and schedules; they do not establish
the cause of cross-scale differences or the training history of a hosted
model.""")

assert path.read_text() == original, "Concurrent edit detected; rerun against current source."
(Path(__file__).parent / "main.before_repairs.tex").write_text(original)
path.write_text(prefix + text + suffix)
assert path.read_text().startswith(prefix) and path.read_text().endswith(suffix)
print("Applied appendix-only corrections; prefix and suffix preserved.")
print("Before SHA256", hashlib.sha256(original.encode()).hexdigest())
print("After SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
