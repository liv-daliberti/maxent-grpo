# Latest reference-KL theory cleanup

Scope: the current subsection beginning `Reference KL retains the reference, not the discovery` and ending before `Two ceilings, and which one a domain sits under`, extracted from `/tmp/paper-latest-theory-20260921/before.tex`. The replacement is `replacement.tex`; no live `main.tex`, experiment, or generator file was edited.

The subsection heading becomes **Reference-KL stationary targets and recovery**, retaining `app:theory-kl`. The manual appendix TOC entry must be synchronized by the integrator.

## Preserved latest work

Every latest theorem, lemma, proposition, corollary, remark, label, citation, and displayed equation is preserved. In particular, the exact Dr.GRPO/Mei convergence and rate result, initialization-dependent trajectory floor, reverse-KL face threshold, distinction between MaxRL stationary targets and proven convergence, two-mode recovery asymptotics, forward-barrier interpretation, and finite-time limitations remain. All sixteen stored numerical recovery times and their measured summary macros remain unchanged.

## Editorial cleanup

- Removed reader-directed motivation, discussion of what a subsection separates, proof-construction history, and the claim that the appendix's boundary cases are necessarily boundaries actually reached by neural training.
- Replaced the introductory claim about all runs using zero KL by the default training setting; the positive-KL comparison is explicitly linked separately.
- Reframed the Mei attribution directly as a specialization of entropy-regularized bandit optimization. Scientific attribution and its step-size/gradient assumptions remain.
- Replaced `admitted`, `verified history`, and `frozen reference` wording with stored/discovered verified modes and a fixed reference.
- Rephrased the absent empirical replay-dose sweep as an identifiable scientific limitation: these experiments do not estimate a dose-response curve. Preserved the substantive caveat.
- Kept all optimizer, full-support, fixed-reference, fixed-bank, transient-diversity, initialization, and neural-scope limitations.
- Recaptioned the recovery table with a bold empirical finding followed by complete flow, initialization, target, and time-unit definitions.

## Mathematical precision repairs

1. **Expected KL value versus its sampled derivative.** GRPO's nonnegative sampled value has reverse-KL expectation only under the stated on-policy sampling condition. A derivative taken with sampled categories held fixed need not be an unbiased derivative of distributional reverse KL. The section now explicitly studies the exactly differentiated KL, retaining the separate implemented-gradient limitation. Independently, for categorical samples, the estimator `mu_a/p_a - log(mu_a/p_a) - 1` has expectation `KL(p||mu)`, while the expected fixed-sample logit derivative is `p-mu`; this demonstrates why value unbiasedness alone does not transfer the theorem to the sampled implementation.

2. **Potential sign.** The displayed flow is negative gradient flow of `F_beta=-Psi(P)+beta D`, with `Psi'=c_G`; the prose now states the sign directly. No equation changed.

3. **MaxRL's G=2 edge case.** The coefficient is nonincreasing from G-1 to 1, with strict decrease for G>2. It is constant at G=2. The stationary-policy proof now says this explicitly.

4. **Forward KL constant and floor.** `KL(mu||p)=R_mu-H(mu)`, rather than literally `R_mu`. The cross-entropy differs by a constant, so its gradient/barrier conclusions remain valid. The inline coordinate-floor constant is now explicitly `C_{T,mu}=R_mu(z(T))+[Psi_max-Psi(P(T))]/beta`. This is the earlier weighted bound with `w=mu` and replay coefficient beta; it adds no new assumption or theorem. Forward-target sampling is described constructively using reference/stored draws or on-policy importance weights, replacing a broad impossibility statement.

5. **Sampling scarcity does not erase stored successes.** P→0 makes fresh verified responses rare; it does not imply no verified success is available anywhere. A populated bank can still supply targets. The mixed-group signal statement is restricted to the centered estimators, and MaxRL's factor approaching G is explicitly a mean-gradient magnitude ratio at the same policy, not a higher chance of drawing a success.

6. **Recovery-table depth is odds.** The existing generator initializes `z_c=0`, `z_b=log(depth)`, and the remaining five logits to -40. Thus depth equals `d=p_b(0)/p_c(0)`, while the initial probability equals `d/(1+d+5 exp(-40))`. The table header now says d and the caption specifies these exact initial conditions. The two-mode theorem continues to use s as an initial probability. No stored time changed.

## Independent validation

`validate.py` passes **592 checks** without training, model calls, bootstrap, or recovery integrations:

- exact finite-distribution identities for reverse-KL gradient, value estimator, fixed-sample derivative, global KL bound, coordinate-force bound, and face minimum;
- the Mei reward/temperature/step rescaling and exact KL objective-gap identity;
- 54 deterministic Dr.GRPO/MaxRL stationary-policy cases across group size, reference, and coefficient, verifying stationary residuals and the reference-conditional ratio;
- the exact KL and replay velocities on a two-correct-mode face, including references with mass outside that face;
- the replay reciprocal-velocity partial fraction identity and finite asymptotic remainder, plus the KL l'Hopital derivative ratio;
- initial table odds, all 16 published values, the 7.3–8.2 KL decade-factor macro, and the replay decade increments from the existing JSON. The largest relative deviation of the stored replay increment from log(10)/rho is 0.000541;
- byte-identical displayed equations and preservation of labels, result environments, and citations.

The numerical cases supplement direct inspection of the proofs; they do not replace proofs or establish neural convergence. The reverse-KL lower-sublevel argument still uses continuity and compactness. Its boundedness is not presented as proof that all trajectories lack a floor. The two-mode recovery constants remain restricted to that face and are not attributed to the seven-category table.

## Sources checked

[DeepSeekMath, Section 4.1.1, Equation (4)](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1) gives the sampled KL value used for the expectation-versus-gradient distinction. The latter distinction was checked independently by direct differentiation.

[Mei et al., Theorem 5 and Lemma 13](https://proceedings.mlr.press/v119/mei20b/mei20b.pdf) and the [supplement](https://proceedings.mlr.press/v119/mei20b/mei20b-supp.pdf) confirm the exact-update setting, positive probability floor, and rate for constant eta≤1/tau. The reward/temperature rescaling preserves that condition as eta≤1/beta.

Local attribution/context audit: `paper/audits/theory_source_reuse_20260921_optimization.md`; existing implementation: `ops/verify_kl_replay_recovery_flow.py`; stored measurements: `paper/results/kl_replay_recovery_flow.json`, associated macros, and table body. The generator was read, not executed.

Root handles source integration, latest-input build checks, PDF layout, and Overleaf packaging.
