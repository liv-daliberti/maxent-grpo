# Surrounding wording for the attributed Mei corollary

Research draft only. `main.tex` has not been edited. Insert `mei_corollary.tex` immediately after the proof of `prop:kl-stationary`, before the paragraph beginning “The stationary target differs from full-coverage uniform replay.” The MaxRL proposition and proof remain intact.

## Opening paragraph of the reference-KL subsection

Replace the sentences from “This subsection answers in the idealized flow” through “The recovery comparison below is proved on a specified two-mode face; it is not a convergence theorem for all full-support trajectories” with:

```tex
This subsection separates the categorical stationary target, exact-gradient
convergence, and recovery rates. For the Dr.GRPO and MaxRL coefficients,
the full-support stationary conditional is the reference's own.
Corollary~\ref{cor:kl-dr-mei-convergence} applies established
entropy-regularized softmax theory to prove convergence for exact Dr.GRPO
gradient steps. The two-mode recovery comparison below addresses a different
question: the time required to restore a rare correct mode. Reference KL's
restoring logit force vanishes at that boundary, whereas replay supplies a
nonvanishing banked-coordinate force.
```

## Paragraph immediately after the reverse-KL sublevel lemma

Replace its final sentence, “The distinction is which sublevels exclude the boundary, not an impossibility of retention under a bounded penalty,” with:

```tex
These are statements about loss sublevels. A particular optimizer can still
admit a trajectory-dependent probability floor, as
Corollary~\ref{cor:kl-dr-mei-convergence} establishes for exact Dr.GRPO
gradient steps with a fixed reference.
```

## Paragraph immediately following the inserted corollary

Replace the existing paragraph beginning “The stationary target differs from full-coverage uniform replay” with:

```tex
The reference-conditional target differs from full-coverage uniform replay:
Corollary~\ref{cor:replay-no-collapse} gives $q\to u$, whereas
Corollary~\ref{cor:kl-dr-mei-convergence} gives $q_n\to\mu|_{\mathcal C}$
for exact Dr.GRPO steps. A broader reference changes that allocation;
replay constructs its target from verified discoveries. For MaxRL,
Proposition~\ref{prop:kl-stationary} identifies the stationary conditional
without a convergence claim. Neither result bounds transient diversity by
the reference's diversity or certifies a neural optimizer trajectory.
```

## Technical choices and references

- The source is Mei et al., ICML 2020, Section 4.2.1, **Theorem 5 and Lemma 13**, not the general-MDP Theorem 6. The exact published step condition is `eta <= 1/tau`; after normalizing the shifted reward, it becomes `eta <= 1/beta`.
- The fragment states existence and parameter dependence of the published constants instead of reproducing their conservative exponential formula. The full mapping and explicit constants are recorded in `paper/audits/theory_source_reuse_20260921_optimization.md`.
- The proof is a direct reduction to the published discrete theorem. It deliberately does not add a continuous-flow convergence claim by an unjustified limit argument.
- `mu` is understood, as in the preceding subsection, to be a normalized reference distribution. All categories receive positive reference mass. The step must evaluate both the fresh mean and the KL gradient exactly.
- The MaxRL exception is specifically its nonconstant coefficient for `G >= 3`; at `G=2` its coefficient is constant and the same reduction would apply with `alpha=1`, but no extra corollary is needed.
- Primary sources: [Mei main paper](https://proceedings.mlr.press/v119/mei20b/mei20b.pdf), [Mei supplement](https://proceedings.mlr.press/v119/mei20b/mei20b-supp.pdf), [Geist paper](https://proceedings.mlr.press/v97/geist19a/geist19a.pdf).
