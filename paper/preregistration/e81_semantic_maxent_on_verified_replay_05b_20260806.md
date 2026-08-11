# E81: fixed semantic MaxEnt added to verified replay, Qwen2.5-0.5B

**Frozen before submission on 2026-08-06.**

## Question

Given verified replay, does a fixed, validator-gated semantic MaxEnt term
increase the breadth of distinct correct solutions without reducing
correctness?

E78 established the two reference arms on this cohort: compute-matched Dr.GRPO
(`control`) and uniform verified-likelihood replay (`replay`). E81 adds a third
arm that is the E78 `replay` arm plus one additional term and nothing else.
E81 does not re-run either E78 arm. Both are inherited unchanged as the frozen
comparators.

The E72 component study could not answer this. Its semantic contrast moved
several components at once (semantic MaxEnt, uniform mode balance, novelty,
and the replay objective itself), and E77, which was designed to isolate the
fixed components, was cancelled before producing endpoints. The
semantic-on-top-of-replay cell has never been run.

## Design

- Model: the pinned Qwen2.5-0.5B-Instruct revision used by the E78 cohort,
  trained from the base model, not from any E78 checkpoint.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split, identical files to E78.
- Seeds: 43, 44, 45, 46, and 47, paired within domain against the E78 arms.
- Training: exactly eight passes, hence 3,072 optimizer updates per run;
  group size 16, one PPO epoch, learning rate 2e-7, rollout temperature 1,
  top-p 1, and beta_KL = 0.
- Evaluation and resumable model checkpoints: every 192 updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint.
- Placement: every E81 cell is pinned to the same physical node as the E78
  cells of the same domain and seed, so each paired difference is taken within
  one GPU model.

The new cohort has 5 domains x 1 arm x 5 seeds = 25 runs. Together with the
inherited E78 arms the analysis cohort is 75 runs.

PointMaze is excluded. Its interactive trainer is a separate driver with no
semantic-advantage path, and distributing an episode-level semantic advantage
across decisions without creating a length incentive is a design question, not
a configuration change. E81 makes no claim about PointMaze.

## Arms

### Inherited comparators, not re-run

`control` and `replay` are exactly the E78 runs recorded in
`var/artifacts/e78_verified_replay_only_05b_jobs.json`.

### Verified replay plus fixed semantic MaxEnt (`semantic`)

Identical to the E78 `replay` arm — same passive verified bank, same capacity
of 16 observed modes per prompt, same one scheduled bank per optimizer update
in deterministic global round-robin order, same uniform verified-likelihood
replay objective at the same fixed weight 0.10, same live replay derivative —
plus one detached advantage term.

For row `i` of a candidate group, let `a_i` be its canonical outcome key. A row
is **eligible** if and only if it is active, parseable, and validator-positive.
Ineligible rows receive exact zero and never enter the predictor's support, so
an all-wrong group is an exact no-op and no incorrect, unparseable, or failed
output is ever rewarded for being unusual.

For an eligible row, let the prompt-local open-set predictor be built from the
persistent per-prompt history of previously verified outcomes plus the
leave-one-out verified peers of the current group, with pseudocount 1 and one
structural unseen bucket. With `s_i` the surprisal of `a_i` under that
predictor, clipped at `C = 5`, the semantic advantage is

    A_sem_i = eta * (min(s_i, C) - E[min(s, C)]) / C,

where the expectation is taken under the same detached predictor. The centered
factor lies in [-1, 1], so `|A_sem_i| <= eta`. It is added to the Dr.GRPO task
advantage after that advantage's own centering, with no second centering step
and no outer clamp.

The coefficient is fixed at **eta = 0.10** for every domain, seed, and update.
This is the reference dose inherited from the E43/E56/E72 implementations,
fixed before execution and not selected against any E78 outcome. E81 evaluates
this fixed intervention. The experiment
does not identify an optimal coefficient, and it does not claim eta = 0.10 is
the best available dose. Because every task in this cohort uses a unit
correct/incorrect reward gap, the bound `|A_sem| <= 0.10` guarantees a verified
response keeps at least a 0.90 advantage margin over an incorrect one, so the
semantic term can reorder verified outcomes among themselves but can never
invert the correctness ordering.

## Exact exclusions

The `semantic` arm hard-disables the one-time canonical novelty bonus, the
conditioned-bank balance KL, canonical-bank entropy shaping, the balance
actuator, adaptive or dual semantic coefficients, entropy controllers of every
kind, token entropy bonuses, sequence entropy, SEED, counterfactual proposals,
singleton escape, support-escape and proposal mechanisms, signed-surrogate xDr,
and reference KL. There is no reward for discovering an outcome for the first
time; the term is a stationary function of the predictor, not of discovery
order. No exhaustive support, evaluation outcome, desired mode count, or
desired entropy is available to training, scheduling, stopping, or checkpoint
choice.

## Compute and code matching

The semantic term adds no rollouts, no additional generated tokens, no extra
forward or backward passes, and no additional loss term. It is arithmetic over
canonical outcome keys already materialized by the replay bank, producing a
detached tensor added to an existing advantage tensor. Sampling budget,
optimizer-update count, replay traversal, and backward envelope are therefore
identical to the E78 `replay` arm. The arms are not matched on the CPU-side
cost of maintaining the semantic predictor, which is not a gradient-carrying
resource.

E81 runs from the E78 runtime snapshot with exactly two files replaced:

- `src/oat_drgrpo/args.py`, which widens one validation predicate so that
  open-set semantic MaxEnt may compose with the uniform verified-likelihood
  replay objective as well as the split mass/balance objective it already
  allowed; and
- `ops/run_experiment.sh`, which adds one new variant branch,
  `verified_replay_semantic_maxent`, and names it in one usage string.

Both changes are strictly additive. The widened predicate is reached only when
`semantic_shannon_coef > 0`, which no E78 arm sets, and a new `case` branch is
unreachable for any other variant string. The launcher fails closed unless the
materialized E81 snapshot differs from the E78 snapshot in exactly those two
paths and in no other file. E81 therefore inherits E78's training code
byte-for-byte on every path either E78 arm executes.

E81 is a paired comparison against completed runs, not a simultaneously
randomized one. Model, data, splits, seeds, schedule, decoding, placement, and
training code are pinned identical; the residual difference is the wall-clock
window in which the runs execute. Training on GPU is not bitwise reproducible,
so no exact-replication cell is claimed or attempted.

## Outcomes and estimands

At every registered half-pass checkpoint report, separately by domain:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct modes@8; and
- excess multiplicity, `distinct@8 - pass@8`.

The primary comparison is the paired seed difference `semantic - replay` at
pass 8 for `distinct@8` and `pass@8`. This is the single-component contrast and
it is the one E81 exists to make. The secondary comparison is the paired
difference `semantic - control` at pass 8, which reports the combined
replay-plus-semantic package against compute-matched Dr.GRPO. The trajectory
summary is trapezoidal AUC over the complete pass-0 through pass-8 half-pass
grid.

Show all five paired seed differences and their mean and range.
Do not pool domains into one effect,
do not report a universal semantic-MaxEnt claim, and
do not select a best checkpoint. A breadth gain that arrives with a `pass@8` or
`mean@8` loss is reported as a trade, not as a win.

## Registered interpretation

Three outcomes are pre-declared, and the manuscript language for each is fixed
now:

1. `distinct@8` improves in most domains without correctness loss: semantic
   MaxEnt is reported as an optional extension to verified replay.
2. It improves only in some domains: reported as a domain-dependent refinement,
   naming the domains, with the null domains shown at equal prominence.
3. Neutral or harmful: verified replay alone remains the method, and E81 is
   reported as the registered negative result that retired the component.

## Mechanism telemetry

Every run logs, passively and without any effect on training: eligible
fraction; the effective semantic advantage's mean, min, max, and RMS; its
positive, negative, and zero fractions; the fraction with `|A_sem| > 0.05`; the
predictive probability quantiles; normalized predictive entropy; tracked prompt
and outcome counts; and the ratio of semantic advantage RMS to task advantage
RMS. Replay telemetry is unchanged from E78: banked-mode survival, verified
bank size, replay actuator opportunities, applied replay gradient, and replayed
verified score.

The `semantic` arm must show a finite nonzero applied semantic advantage on at
least one eligible update and exact zero on every ineligible row. These
measurements are descriptive. They may motivate a separately registered
sensitivity study over {0.05, 0.10, 0.20}; they may not be used to re-scale,
re-select, or re-report E81's own coefficient.

## Integrity and failure policy

All 25 cells are submitted held from one hash-bound runtime snapshot and are
released only after their scheduler environments pass an exact audit. A
malformed environment, a snapshot diverging from the E78 snapshot outside the
two declared files, a missing or mismatched E78 pair, a duplicate run
directory, a non-finite loss, proposal or novelty leakage, a traceback, or a
missing pass-8 endpoint fails closed. Infrastructure interruption may resume
only from the same run's hash-bound checkpoint and exact bank, semantic
predictor, optimizer, data-cursor, and request-stream state. No failed
scientific run is silently replaced or excluded.
