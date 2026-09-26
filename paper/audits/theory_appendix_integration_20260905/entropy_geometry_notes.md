# Entropy and information-geometry appendix integration

The two fragments adapt the already reviewed supplement into the paper's notation and theorem environments. They retain complete derivations and separate existing theory from the finite verified-mode specialization. No main manuscript, manuscript bibliography, or original supplement was edited by this agent.

## Placement and crosslinks

- Insert `natural_gradient_appendix.tex` immediately before the existing exact-entropy subsection. It defines its fixed bank and normalized weights locally, so it does not depend on the later replay subsection. Label: `app:theory-natural-gradient`; theorem: `ext:thm-natural-replay`.
- Insert `entropy_appendix.tex` immediately after the existing exact-entropy/sampled-score subsection. Label: `app:theory-entropy-comparison`; lemma: `ext:lem-entropy-objectives`.
- The entropy fragment refers to existing `lem:entropy-scope` for the full-output Gibbs optimum, avoiding a duplicate proof or formula. It points to `app:theory-survival-certificates` for the sharp entropy threshold and replay certificates.
- The natural-gradient fragment refers to `thm:grpo-collapse`, `lem:grpo-mean`, and `lem:maxrl-mean`, all present in the original paper. Its new equation labels use the `ext:` namespace.

## Citation attribution and verified metadata

Four new entries are in `entropy_geometry_refs.bib`; the existing `geist2019regmdp` entry is reused.

| Key | Verified primary record | What the appendix uses |
| --- | --- | --- |
| `pereyra2017confidencepenalty` | Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, Geoffrey Hinton. *Regularizing Neural Networks by Penalizing Confident Output Distributions*. arXiv:1701.06548 [cs.NE], 2017. [Primary record](https://arxiv.org/abs/1701.06548) | Section 3.2's established distinction between label smoothing and a confidence penalty through the direction of KL. The verified-subset identities are derived in the new lemma. |
| `agarwal2021policygradient` | Alekh Agarwal, Sham M. Kakade, Jason D. Lee, Gaurav Mahajan. *On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift*. JMLR 22(98):1–76, 2021. [Journal record](https://jmlr.org/papers/v22/19-736.html) | Section 5.2's log-barrier policy-gradient precedent. No full state/action-coverage guarantee is imported into a partial replay bank. |
| `mei2020softmaxpg` | Jincheng Mei, Chenjun Xiao, Csaba Szepesvári, Dale Schuurmans. *On the Global Convergence Rates of Softmax Policy Gradient Methods*. ICML, PMLR 119:6820–6829, 2020. [Proceedings record](https://proceedings.mlr.press/v119/mei20b.html) | Lemma 13 and Theorem 5 give probability floors and geometric convergence for exact entropy-regularized tabular policy gradient under their finite-bandit and step-size assumptions. The primary PDF directly states the positive floor and the theorem for Update 2 with step size at most inverse entropy coefficient. The Dr.GRPO linear binary reward satisfies Assumption 1, whereas nonlinear MaxRL is outside that direct specialization. |
| `harper2009informationgeometry` | Marc Harper. *Information Geometry and Evolutionary Game Theory*. arXiv:0911.1383 [cs.IT], 2009. [Primary record](https://arxiv.org/abs/0911.1383) | Sections 2.3 and 3.1 identify the categorical Fisher/Shahshahani metric and its replicator gradient. The new proof rederives the metric characterization before solving the correctness-plus-replay specialization. The canonical arXiv submission year is 2009; the subject is cs.IT, not q-bio.PE. |
| `geist2019regmdp` (existing) | Matthieu Geist, Bruno Scherrer, Olivier Pietquin. *A Theory of Regularized Markov Decision Processes*. ICML, PMLR 97:2160–2169, 2019. [Proceedings record](https://proceedings.mlr.press/v97/geist19a.html) | Section 2.2's entropy/conjugacy calculation, linked to the paper's existing exact-entropy lemma. |

The primary metadata for all four new entries were checked again during integration. Mei's published PDF (page 7, Lemma 13 and Theorem 5) and Harper's primary PDF (Sections 2.3 and 3.1) were reopened to check the precise attribution. The remaining source locators match the earlier verified `entropy_sources.json` in the theoretical supplement.

## Mathematical checks and scope

The entropy lemma compares global maximizing categorical distributions on the simplex closure with positive total correct mass. It makes no new convergence assertion for conditional-entropy dynamics. Full-output entropy, conditional correct-mode entropy, and the historical bounded semantic estimator remain explicitly different.

The natural-gradient theorem assumes the exact categorical Fisher metric, a fixed finite bank, a positive smooth correctness derivative, and an interior starting distribution. The proof includes the gradient's defining metric identity, the correctness and conditional equations, the explicit conditional solution, global interior existence at all finite times, and the resulting probability floors. An incomplete bank still sends unbanked correct mass to zero when replay is positive. Neither theorem is described as an AdamW/PPO guarantee or a new general result in information geometry.

Root independently reviewed both fragments: PASS for mathematics, scope, and crosslinks. The isolated LaTeX check built a four-page temporary document using `pdflatex`, `bibtex`, and two further `pdflatex` passes. The final log contained no warnings, undefined references, or overfull boxes. The temporary check is at `/tmp/theory_appendix_entropy_geometry_20260905`; final integrated manuscript layout remains the root's check.

An independent review of the companion optimizer fragment's new bounded-noise corollary also passed its algebra and maximal-probability argument. The review requested that its statement explicitly retain conditional noise centering, deterministic initialization, and the confidence-level range; its constants and formulas required no correction.

Root integration note: the final manuscript wording received subsequent prose reflow and notation alignment for the existing layout checks. Handoff hashes above describe the reviewed draft; final integrated hashes and checks are in `validation.json`. No result assumptions were removed.
