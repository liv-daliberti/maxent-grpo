# Per-mode survival: sharper certificates and direct measurements

The useful extension is a **normalized, sharp binary-KL certificate**, coupled to measurements that distinguish exemplar likelihood from execution-key probability. The results and full proofs are in [survival.tex](survival.tex), a standalone note with its own bibliography. These are applications of established information-theoretic and statistical tools, not new claims of foundational inequalities or neural-optimizer convergence. No manuscript, scheduler, runtime, or cohort was changed.

## Proven certificates

For normalized positive bank weights `w`, extend `w` by zero off the bank. If replay cross entropy `R_w(p)=-sum w_b log p_b <= C`, then `D=C-H(w)>=0` and `KL(w_extended || p)<=D`. Binary data processing gives

`kl(w_b || p_b) <= D`, hence `p_b >= lower_inverse_kl(w_b,D)`.

For `0<w_b<1`, the inverse is the unique root in `(0,w_b]`; it is positive for every finite D. The bound is sharp: put that mass on b and distribute all remaining mass proportionally to the other target weights. Strictly positive off-bank coordinates can approach the boundary construction. Total bank mass is at least `exp(-D)`. Pinsker supplies the simpler, often weaker floor `max(0,w_b-sqrt(D/2))`. A positive closed-form intermediate bound is `exp(-(D+h2(w_b))/w_b)`.

This improves the current termwise `exp(-C/w_b)` bound substantially near the target. For a uniform 16-mode bank:

| Excess cross entropy D | Sharp per-mode lower bound | Current termwise lower bound |
|---:|---:|---:|
| 0 | 0.0625 | 5.42e-20 |
| .001 | .0522550 | 5.34e-20 |
| .01 | .0339712 | 4.62e-20 |
| .1 | .00518104 | 1.09e-20 |
| 1 | 2.67e-9 | 6.10e-27 |

These are illustrative inputs, not observed training losses. The D=0 case describes the exact target on the closed simplex. At D=160 the sharp **log** floor is still −2563.74: normalization improves the certificate but cannot make a huge energy budget informative.

For entropy alone, `H(q)>=h0` on m possible modes certifies a strictly positive floor for every coordinate **if and only if** `h0>log(m-1)`. The sharp floor solves

`h2(r)+(1-r)log(m-1)=h0`, for `0<r<=1/m`.

Below or at that threshold, one mode can have zero mass while the others are uniform. Equivalently a forward-KL budget `KL(q||u)<=epsilon` must be below `log(m/(m-1))` to exclude a missing coordinate; finite reverse KL excludes one at every finite budget. This is a distinction between level sets, **not** a claim that exact entropy dynamics necessarily collapse. Conditional entropy also says nothing by itself about total correct mass.

[van Erven and Harremoës, Theorems 9 and 31](https://arxiv.org/abs/1206.2459v2) provide the general data-processing and Pinsker statements. [Garivier and Cappé](https://proceedings.mlr.press/v19/garivier11a.html) provide an established binary-KL inversion application; this note does not import their bandit regret theorem as a survival theorem.

## When a likelihood score certifies a verified key

At a single prompt and checkpoint, use distinct **complete-response events**, all scored under the same decoding law. Give the exemplars normalized target weights w; multiple exemplars for one key combine to target mass `W_b`. The normalized response cross entropy bounds response KL, and execution followed by the key indicator is data processing. Thus `p(key b)>=lower_inverse_kl(W_b,C-H(w))`. The elementary individual bound is `p(key b)>=pi(complete exemplar)`.

The necessary conditions are precise: include EOS or a specified terminal event, or use a fixed deterministic horizon; match prompt formatting, temperature, vocabulary masks, top-p/top-k rules, and verifier semantics; use a fixed snapshot for all rows entering one CE calculation. Prefix probabilities do not generally certify their parsed key because extensions may change the answer or be invalid. Different prompts require separate conditional distributions or an explicitly normalized joint prompt mixture.

Mean-token scores are not probabilities. At fixed length L, sequence probability is `exp(L*mean_logprob)` and the ratio across checkpoints is `exp(delta_sequence_logprob)`. For an objective `sum alpha_j*(-log pi_j)/L_j`, first put `a_j=alpha_j/L_j`, `A=sum a_j`, `w_j=a_j/A`; its loss is `A*R_w`. Uniform averaging of mean-token scores therefore induces inverse-length **sequence** target weights. A change in one exemplar's likelihood need not equal the change in the total probability of its execution key. Numerically approximate log scores also need error control before being called rigorous numerical lower bounds.

## Direct finite-count interpretation

At a frozen checkpoint, prespecify a key and count its occurrences among N independent draws. One-sided exact binomial bounds directly measure its probability. For zero sightings the 95% upper bound is `1-.05^(1/N)`: .31234 for N=8 and .08937 for N=32. Simultaneously protecting 16 prespecified modes by Bonferroni gives .16495 at N=32. Zero count supplies no positive lower bound and is not evidence of exact extinction.

For a single prespecified mode, two sightings in 32 draws give a one-sided 95% lower bound .01122. It takes 299 zero-count draws to certify p<.01 at one-sided 95%; 574 suffice simultaneously for 16 modes. The lower and upper examples are **separate one-sided 95% limits**; treating them jointly as one 95% interval requires allocating both tails.

An exact binomial construction is supported by [Clopper and Pearson's original paper](https://www.barestatistics.nl/uploads/1/1/7/9/11797954/clopper__pearson_1934.pdf). It applies per prompt/key/checkpoint, not to a pooled count across heterogeneous prompts, seeds, or changing policies. Repeatedly inspecting samples and choosing when to stop requires a prespecified sequential method or a [confidence sequence](https://arxiv.org/abs/1810.08240). The TEX gives a self-contained beta-mixture martingale construction for a fixed Bernoulli probability. That does not justify pooling changing checkpoints to infer the latest probability.

Missing mass concerns the **total probability** of unobserved categories under a fixed iid law. It is relevant to a future discovery/coverage audit with unknown catalogue, but it neither estimates the number of extinct modes nor certifies each named key. [McAllester and Ortiz](https://jmlr.org/papers/volume4/mcallester03a/mcallester03a.pdf) and [Berend and Kontorovich](https://arxiv.org/abs/1210.3248) analyze this problem. In particular, a bound around the unknown expected missing mass is not automatically a computable confidence interval, and a singleton-frequency point estimate is not a guarantee. For E121's already named frozen keys, direct counts and likelihood trajectories are the more immediate measurements.

## E121: available measurements and remaining analysis gaps

Read-only file inspection found **no run directories or telemetry for any of the five registered seeds** (43–47). This is file-availability evidence, not a new scheduler audit. The existing readiness receipt already describes incomplete E120 prerequisites and obsolete dependency identities; no scheduling work was performed here.

The [preregistration](../../preregistration/e121_fixed_bank_survival_telemetry_20260903.md) freezes Graph bank membership and fresh counts at learner step384. It records identity fingerprints and mean/sequence log scores on subsequent replay visits. These are training-prompt exemplar trajectories, not categorical draw frequencies on the held-out evaluation prompts. The protocol appropriately calls them score surrogates and does not claim to verify the theorem's numerical floor.

The current and frozen [auditor](../../../ops/exp_scaling/audit_e121_fixed_bank_survival.py) are byte-identical (SHA-256 `2b48f90cdc6a339d8eb0b144f6a269e99d5f22baeefcdab3c735448c7330d77d`). Source review identifies substantive unfinished protocol checks:

1. It creates its identity population only from observed rows; a frozen key never logged at all cannot be detected without a checkpoint/roster comparison. Fewer-than-two visits are counted only for identities seen at least once.
2. `finite_every_observation_fraction` is hardcoded to1.0 after parsed rows pass finite checks. There is no denominator of all scheduled visits and no test for omitted frozen flags/whole records. This is not the registered finite-at-every-scheduled-observation estimand.
3. Membership groups are checked only on the intersection of prompt and membership indices. Missing membership fingerprints can pass; alignment between row identities and the frozen complete roster is not established.
4. The registered 10th percentile, sequence-score deltas, worst intermediate drop, complete per-identity visit distributions, and hierarchical 10,000-draw bootstrap are absent. The implemented median selects the upper middle order statistic for an even count, which should be made explicit or replaced by the conventional midpoint when completing the summary.
5. `resolve_metrics` chooses one direct/terminal-attempt file. It does not reconcile multiple authorized checkpoint-resume segments, remove restart duplicates by scientific step, or validate full horizon completion. Infrastructure histories must be reconciled before a first-to-final claim.

Telemetry itself computes sequence scores as `mean_score * response_mask_token_count` (`src/oat_drgrpo/learner/grpo.py:1638–1690`). Materialization scores every stored response token (`src/oat_drgrpo/canonical_replay.py:181`), and admission stores masked rollout tokens (`grpo.py:2848`). These paths alone do not prove that every stored sequence includes the applicable terminal event; the frozen bank and generation contract must establish it. There is no explicit per-row token-count field in this telemetry block, although token IDs/lengths are recoverable from the bank checkpoint and sequence/mean values normally imply the length.

Completing these checks is necessary before using the registered mechanism endpoint. It need not change the fixed intervention, seeds, roster, or estimands. New categorical sampling on frozen training prompts would be a separately specified measurement; no such sampling or amendment was performed. In reporting, distinguish **remembered identity**, **scored exemplar**, **observed key in a finite draw**, **probability above a chosen threshold**, and **exact extinction**.

## Validation and verified source metadata

Run `python3 paper/audits/theory_literature_extensions_20260905/verify_survival.py`.
The [JSON](verify_survival.json) records PASS for126 sharp reverse-KL equality constructions (largest residual5.33e-15),200 random CE/decomposition checks,24 entropy-threshold constructions, and21 binomial-coverage checks across N=8,32,128. Roots use log space, so very small mathematical floors are not misreported as zero through numerical underflow.

Primary full texts were checked for all six sources:

- **Tim van Erven; Peter Harremoës.** *Rényi Divergence and Kullback–Leibler Divergence.* 2014 author version, arXiv1206.2459v2; IEEE Transactions on Information Theory, DOI10.1109/TIT.2014.2320500. The arXiv record verifies authors, version date, journal acceptance, DOI; full text verifies Theorems9/31 and the entropy–uniform-KL identity. No publication page numbers are inferred from the preprint's template header.
- **Aurélien Garivier; Olivier Cappé.** *The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond.* COLT2011, PMLR19:359–376. Official proceedings metadata and full text verified; Section2 defines Bernoulli KL and its inversion.
- **C. J. Clopper; E. S. Pearson.** *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial.* Biometrika26(4):404–413,1934; DOI10.1093/biomet/26.4.404. Original article scan verifies metadata and binomial-tail construction. Publisher endpoint did not load; a readable copy of the original article was used, not a secondary exposition.
- **Steven R. Howard; Aaditya Ramdas; Jon McAuliffe; Jasjeet Sekhon.** *Time-uniform, nonparametric, nonasymptotic confidence sequences.* Annals of Statistics49(2):1055–1080,2021; DOI10.1214/20-AOS1991; arXiv1810.08240. Author record and full text checked, especially Proposition7 and the distinction between pointwise and time-uniform intervals. Latest arXiv revision is2022; that does not change the2021 journal year.
- **David McAllester; Luis Ortiz.** *Concentration Inequalities for the Missing Mass and for Histogram Rule Error.* JMLR4:895–911,2003. Official full text verifies metadata and fixed-distribution missing-mass scope.
- **Daniel Berend; Aryeh Kontorovich.** *On the concentration of the missing mass.* Electronic Communications in Probability18,paper3:1–7,2013; DOI10.1214/ECP.v18-2359; arXiv1210.3248. Primary article and preprint verify definition and concentration statements. The initial arXiv year2012 is distinct from journal publication2013.
