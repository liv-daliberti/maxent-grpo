# E83: fixed semantic MaxEnt without verified replay, Qwen2.5-0.5B

**Frozen before submission on 2026-08-07.**

## Question

Does fixed, validator-gated semantic MaxEnt do anything on its own, without
verified replay underneath it?

E78 established two arms on this cohort: compute-matched Dr.GRPO (`control`)
and uniform verified-likelihood replay (`replay`). E81 added replay plus
semantic MaxEnt. E83 adds the fourth cell: semantic MaxEnt with the replay
derivative off.

Together the four arms are a complete 2x2 over the two applied derivatives:

|                   | no semantic term  | semantic term |
|-------------------|-------------------|---------------|
| no replay gradient| E78 `control`     | **E83**       |
| replay gradient   | E78 `replay`      | E81           |

E83 does not re-run E78 or E81. All three existing arms are inherited
unchanged as frozen comparators.

E83 is registered before E81 has a reportable endpoint. Its design is fixed by
the factorial, not by E81's outcome, and nothing in this protocol may be
revised once E81's results are known.

## Design

- Model: the pinned Qwen2.5-0.5B-Instruct revision used by E78 and E81,
  trained from the base model, not from any existing checkpoint.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split, identical files to E78 and E81.
- Seeds: 43, 44, 45, 46, and 47, paired within domain against all three
  existing arms.
- Training: exactly eight passes, hence 3,072 optimizer updates per run;
  group size 16, one PPO epoch, learning rate 2e-7, rollout temperature 1,
  top-p 1, and beta_KL = 0.
- Evaluation and resumable model checkpoints: every 192 updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint.
- Placement: every E83 cell is pinned to the same physical node as the E78 and
  E81 cells of the same domain and seed, so every paired difference is taken
  within one GPU model.

The new cohort has 5 domains x 1 arm x 5 seeds = 25 runs. With the three
inherited arms the analysis cohort is 100 runs.

PointMaze is excluded, as in E81 and E82.

## Arms

### Inherited comparators, not re-run

`control` and `replay` are the E78 runs recorded in
`var/artifacts/e78_verified_replay_only_05b_jobs.json`; `semantic` with replay
is the E81 run set in
`var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json`.

### Semantic MaxEnt without replay (`semantic_only`)

Identical to the E78 `control` arm — same passive verified bank, same capacity
of 16 observed modes per prompt, same one scheduled bank per optimizer update
in deterministic global round-robin order, same teacher-forced score traversal,
same backward-compute envelope, and the same **exactly zero** applied replay
derivative — plus one detached advantage term.

"Without replay" means what it means in the published control: the replay
bookkeeping and traversal are performed so the compute envelope matches, and
the replay score derivative is identically zero. E83 is therefore Dr.GRPO plus
semantic MaxEnt in its applied objective.

The semantic term is bit-identical in definition to E81's and E82's. For row
`i` of a candidate group, let `a_i` be its canonical outcome key. A row is
**eligible** if and only if it is active, parseable, and validator-positive.
Ineligible rows receive exact zero and never enter the predictor's support, so
an all-wrong group is an exact no-op.

For an eligible row, the prompt-local open-set predictor is built from the
persistent per-prompt history of previously verified outcomes plus the
leave-one-out verified peers of the current group, with pseudocount 1 and one
structural unseen bucket. With `s_i` the surprisal of `a_i` under that
predictor, clipped at `C = 5`, the semantic advantage is

    A_sem_i = eta * (min(s_i, C) - E[min(s, C)]) / C,

where the expectation is taken under the same detached predictor. The centered
factor lies in [-1, 1], so `|A_sem_i| <= eta`. It is added to the Dr.GRPO task
advantage after that advantage's own centering, with no second centering step
and no outer clamp.

Because the predictor is detached, ascending this centered surprisal is the
score-function estimator of the gradient of the entropy of the prompt-local
verified-outcome distribution; at `p_hat = p_theta` the detached cross-entropy
surrogate has exactly the entropy gradient. E83 therefore isolates that
entropy ascent, with no replay gradient present.

The coefficient is fixed at **eta = 0.10**, the same reference dose E81 and
E82 use, carried over unchanged so the factorial varies one thing at a time.
E83 does not tune the coefficient and does not identify an optimal value.

## Exact exclusions

As E81. The arm hard-disables the applied replay derivative, the one-time
canonical novelty bonus, the conditioned-bank balance KL, canonical-bank
entropy shaping, the balance actuator, adaptive or dual semantic coefficients,
entropy controllers of every kind, token entropy bonuses, sequence entropy,
SEED, counterfactual proposals, singleton escape, support-escape and proposal
mechanisms, signed-surrogate xDr, and reference KL.
There is no reward for discovering an outcome for the first
time. No exhaustive support, evaluation
outcome, desired mode count, or desired entropy is available to training,
scheduling, stopping, or checkpoint choice.

## Compute and code matching

All four cells of the 2x2 carry the identical compute envelope: the same
sampling budget, the same optimizer-update count, the same bank traversal, and
the same backward envelope. They differ only in which of the two derivatives
is applied. The arms are not matched on the CPU-side cost of maintaining the
semantic predictor, which is not a gradient-carrying resource.

E83 runs from the E78 runtime snapshot with exactly two files replaced, the
same two E81 and E82 replace: `src/oat_drgrpo/args.py`, which widens one
validation predicate, and `ops/run_experiment.sh`, which adds the
`compute_matched_semantic_maxent` variant branch alongside the branches the
other arms use. Both changes are additive and unreachable from any E78 arm's
configuration. The launcher walks both trees and fails closed unless the
divergence is exactly that patch set.

E83 is a paired comparison against runs already executing when it was
registered, not a simultaneously randomized one. Model, data, splits, seeds,
schedule, decoding, placement, and training code are pinned identical; the
residual difference is the wall-clock window.

## Outcomes and estimands

At every registered half-pass checkpoint report, separately by domain:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct modes@8; and
- excess multiplicity, `distinct@8 - pass@8`.

The primary comparison is the paired seed difference `semantic_only - control`
at pass 8 for `distinct@8` and `pass@8`: the main effect of semantic MaxEnt
with no replay present. The secondary comparisons are the interaction
contrast

    (E81 - replay) - (E83 - control),

which asks whether semantic MaxEnt does something different in the presence of
replay than in its absence, and `semantic_only - replay`, which ranks the two
single-component arms against each other. The trajectory summary is
trapezoidal AUC over the complete pass-0 through pass-8 half-pass grid.

Show all five paired seed differences and their mean and range.
Do not pool domains into one effect,
do not report a universal semantic-MaxEnt claim, and
do not select a best checkpoint. A breadth gain that arrives with a `pass@8` or
`mean@8` loss is reported as a trade, not as a win. The interaction contrast
is reported descriptively, per domain, and is not tested for significance.

## Registered interpretation

Fixed now, before any of the four cells reports:

1. Semantic MaxEnt helps alone and with replay, with no interaction: it is
   reported as an independent component that composes with verified replay.
2. It helps only with replay present: reported as requiring a populated
   verified bank, with the mechanism stated — the predictor has no support to
   score against until replay has kept modes alive.
3. It helps only without replay: reported as redundant with, or in competition
   with, verified replay.
4. It does not help in either condition: verified replay alone remains the
   method, and E81 and E83 are reported together as the registered negative
   result that retired the component.

## Mechanism telemetry

As E81, with one addition: the applied replay gradient must be exactly zero on
every update where a replay opportunity exists, and this is checked rather
than assumed. Every run also logs, passively: eligible fraction; the effective
semantic advantage's mean, min, max, and RMS; its positive, negative, and zero
fractions; the fraction with `|A_sem| > 0.05`; the predictive probability
quantiles; normalized predictive entropy; tracked prompt and outcome counts;
and the ratio of semantic advantage RMS to task advantage RMS.

Because E83 has no replay gradient, its verified bank is populated only by
what the policy rediscovers. Banked-mode survival and verified bank size are
therefore reported as outcomes of this arm, not as controls.

## Integrity and failure policy

All 25 cells are submitted held from one hash-bound runtime snapshot and are
released only after their scheduler environments pass an exact audit. A
malformed environment, a snapshot diverging from the E78 snapshot outside the
two declared files, a missing or mismatched E78 pair, a nonzero applied replay
gradient, a duplicate run directory, a non-finite loss, proposal or novelty
leakage, a traceback, or a missing pass-8 endpoint fails closed. Infrastructure
interruption may resume only from the same run's hash-bound checkpoint and
exact bank, semantic predictor, optimizer, data-cursor, and request-stream
state. No failed scientific run is silently replaced or excluded.
