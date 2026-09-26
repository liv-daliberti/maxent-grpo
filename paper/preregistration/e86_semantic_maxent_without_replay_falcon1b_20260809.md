# E86: fixed semantic MaxEnt without verified replay, Falcon3-1B

**Frozen before submission on 2026-08-09.**

## Question

Does the E83 result, whatever it turns out to be, hold in a second model
family? E86 is the Falcon3-1B replication of E83: fixed, validator-gated
semantic MaxEnt with the replay derivative off, at the same fixed coefficient,
evaluated against the same paired-difference estimand.

E79 established two arms on this cohort: compute-matched Dr.GRPO (`control`)
and uniform verified-likelihood replay (`replay`). E82 added replay plus
semantic MaxEnt. E86 adds the fourth cell and closes the Falcon 2x2 over the
two applied derivatives:

|                   | no semantic term  | semantic term |
|-------------------|-------------------|---------------|
| no replay gradient| E79 `control`     | **E86**       |
| replay gradient   | E79 `replay`      | E82           |

This is the same 2x2 E78/E81/E83 form on Qwen2.5-0.5B, cell for cell. The
cross-family question E86 answers is whether the interaction contrast

    (E82 - replay) - (E86 - control)

carries the sign it carries on Qwen, not whether either model family shows a
main effect.

E86 does not re-run E79 or E82. All three existing arms are inherited unchanged
as frozen comparators.

E86 is registered before E82 has a reportable endpoint and before E83's
endpoint has been read. Its design is fixed by the factorial, not by either
outcome, and nothing in this protocol may be revised once they are known.

## Design

- Model: Falcon3-1B-Instruct at the pinned revision
  `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf`, the revision E79 and E82 use,
  trained from the base model, not from any existing checkpoint.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split, identical files to E79 and E82, presented through the same
  Falcon prompt-surface twins and per-domain generation budgets E79 used.
- Seeds: 55, 56, 57, 58, and 59, paired within domain against all three
  existing arms.
- Training: exactly eight passes, hence 3,072 optimizer updates per run;
  group size 16, one PPO epoch, AdamW at a constant 2e-7 with zero warmup,
  betas (0.9, 0.999), no weight decay, gradient clipping at 1, rollout
  temperature 1, top-p 1, and beta_KL = 0.
- Evaluation and resumable model checkpoints: every 192 updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint.
- Placement: E79's placement is a deterministic function of domain and seed, so
  each E86 cell is pinned to the same physical node and GPU model as its E79
  pair and its E82 cell, and every paired difference is taken within one GPU
  model.

The new cohort has 5 domains x 1 arm x 5 seeds = 25 runs. With the three
inherited arms the analysis cohort is 100 runs.

PointMaze is excluded, as in E81, E82, and E83. Its interactive trainer is a
separate driver with no semantic-advantage path. E86 makes no claim about
PointMaze.

## Arms

### Inherited comparators, not re-run

`control` and `replay` are the E79 runs recorded in
`var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json`; `semantic` with
replay is the E82 run set in
`var/artifacts/e82_falcon_semantic_maxent_verified_replay_jobs.json`, except on
PantryPlan, where E82's column is superseded and the comparator is the E85
repair recorded in `var/artifacts/e85_pantry_semantic_repair_jobs.json` under
parent `e82`. See "PantryPlan is born repaired" below.

### Semantic MaxEnt without replay (`semantic_only`)

Identical to the E79 `control` arm — same passive verified bank, same capacity
of 16 observed modes per prompt, same one scheduled bank per optimizer update
in deterministic global round-robin order, same teacher-forced score traversal,
same backward-compute envelope, and the same **exactly zero** applied replay
derivative — plus one detached advantage term.

"Without replay" means what it means in the published control: the replay
bookkeeping and traversal are performed so the compute envelope matches, and
the replay score derivative is identically zero. The replay dose remains
declared at 0.10 and is never applied. E86 is therefore Dr.GRPO plus semantic
MaxEnt in its applied objective.

The semantic term is bit-identical in definition to E81's, E82's, and E83's.
For row `i` of a candidate group, let `a_i` be its canonical outcome key. A row
is **eligible** if and only if it is active, parseable, and validator-positive.
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
surrogate has exactly the entropy gradient. E86 therefore isolates that entropy
ascent, with no replay gradient present.

The coefficient is fixed at **eta = 0.10**, the same reference dose E81, E82,
and E83 use, carried over unchanged so the factorial varies one thing at a
time. E86 does not tune the coefficient and does not identify an optimal value.

## PantryPlan is born repaired

PantryPlan is a canonical-action task. In E81, E82, and E83 the semantic term
derived its outcome key with the free-form text extractor, which returns None
for an eight-token action sequence, so every PantryPlan row was scored
unparseable and the term contributed exact zero for the whole of training.
E85 re-ran those 15 cells against a runtime that binds the task's own
canonicalization surfaces.

E86 runs from that repaired runtime from its first update. Two consequences are
registered here rather than discovered later:

1. E86's PantryPlan cells need no repair cohort. Their acceptance gate is the
   same mechanism-only gate E85 uses — parseable fraction above 0.5 and
   eligible fraction above 0.1 per cell — and a cell that fails it fails
   closed as unrepaired rather than being reported.
2. E86's PantryPlan comparator is the E85 repair of E82's PantryPlan column,
   not E82's original cells. Both sides of that paired difference then carry a
   live semantic term. E82's superseded PantryPlan runs are audit-only evidence
   of the defect and are excluded from every E86 result. The `control` and
   `replay` comparators are unaffected: the defect required semantic MaxEnt and
   canonical actions together, and neither E79 arm enables the semantic term.

The repair branch is reachable only when the task resolves to a canonical
action task. Of the five domains, only PantryPlan does; Graph Coloring,
Countdown, Python Factors, and MathIR all run with
`canonical_action_task=none`, where the patched call site is byte-for-byte the
call it replaces. Those four domains therefore compare against E82 under an
identical runtime, and only PantryPlan's runtime differs from E82's original.

## Exact exclusions

As E83. The arm hard-disables the applied replay derivative, the one-time
canonical novelty bonus, the conditioned-bank balance KL, canonical-bank
entropy shaping, the balance actuator, adaptive or dual semantic coefficients,
entropy controllers of every kind, token entropy bonuses, sequence entropy,
SEED, counterfactual proposals, singleton escape, support-escape and proposal
mechanisms, signed-surrogate xDr, and reference KL.
There is no reward for discovering an outcome for the first time.
No exhaustive support, evaluation outcome, desired mode count, or desired
entropy is available to training, scheduling, stopping, or checkpoint choice.

## Compute and code matching

All four cells of the 2x2 carry the identical compute envelope: the same
sampling budget, the same optimizer-update count, the same bank traversal, and
the same backward envelope. They differ only in which of the two derivatives is
applied. The arms are not matched on the CPU-side cost of maintaining the
semantic predictor, which is not a gradient-carrying resource.

E86 runs from the E79 runtime snapshot with exactly three files replaced:
`src/oat_drgrpo/args.py`, which widens one validation predicate;
`ops/run_experiment.sh`, which adds the `compute_matched_semantic_maxent`
variant branch alongside the branches the other arms use; and
`src/oat_drgrpo/learner/grpo.py`, which binds the canonical-action
canonicalization surfaces at the semantic key derivation site. The first two
are the pair E81, E82, and E83 replace and are additive and unreachable from
any E79 arm's configuration. The third is E85's patch set, whose new branch
sits inside `if canonical_actions:` and is therefore unreachable for every
domain but PantryPlan and for every arm that does not enable semantic MaxEnt.
The launcher walks both trees and fails closed unless the divergence from the
E79 snapshot is exactly that patch set.

E86 is a paired comparison against runs already executing when it was
registered, not a simultaneously randomized one. Model, data, splits, seeds,
schedule, decoding, placement, and training code are pinned identical up to the
declared patch set; the residual difference is the wall-clock window.

## Outcomes and estimands

At every registered half-pass checkpoint report, separately by domain:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct modes@8; and
- excess multiplicity, `distinct@8 - pass@8`.

The primary comparison is the paired seed difference `semantic_only - control`
at pass 8 for `distinct@8` and `pass@8`: the main effect of semantic MaxEnt
with no replay present, in the second model family. The secondary comparisons
are the interaction contrast

    (E82 - replay) - (E86 - control),

which asks whether semantic MaxEnt does something different in the presence of
replay than in its absence, and `semantic_only - replay`, which ranks the two
single-component arms against each other. The cross-family comparison is
whether E86's interaction contrast agrees in sign with E83's, reported per
domain and descriptively. The trajectory summary is trapezoidal AUC over the
complete pass-0 through pass-8 half-pass grid.

Show all five paired seed differences and their mean and range.
Do not pool domains into one effect,
do not pool model families,
do not report a universal semantic-MaxEnt claim, and
do not select a best checkpoint.
A breadth gain that arrives with a `pass@8` or `mean@8` loss is reported as a
trade, not as a win. The interaction contrast and the cross-family sign
agreement are reported descriptively and are not tested for significance.

## Registered interpretation

Fixed now, before any of the four Falcon cells reports:

1. The Falcon 2x2 reproduces the Qwen 2x2's sign structure: the component's
   behaviour is reported as a property of the method rather than of
   Qwen2.5-0.5B.
2. Falcon shows a main effect where Qwen does not, or the reverse: reported as
   family-dependent, with the two 2x2 tables shown side by side and no pooled
   claim. The manuscript states that one family's result does not transfer.
3. The interaction contrasts disagree in sign across families: reported as
   evidence that composition with verified replay is family-dependent, which
   retires any claim that the components compose additively in general.
4. Neither family shows an effect in either condition: verified replay alone
   remains the method, and E81, E82, E83, and E86 are reported together as the
   registered negative result that retired the component across two families.

## Mechanism telemetry

As E83, with the same addition: the applied replay gradient must be exactly
zero on every update where a replay opportunity exists, and this is checked
rather than assumed. Every run also logs, passively: eligible fraction; the
effective semantic advantage's mean, min, max, and RMS; its positive, negative,
and zero fractions; the fraction with `|A_sem| > 0.05`; the predictive
probability quantiles; normalized predictive entropy; tracked prompt and
outcome counts; and the ratio of semantic advantage RMS to task advantage RMS.

Parseable fraction and eligible fraction are read on every domain, not only
PantryPlan, and a domain whose parseable fraction is at or near zero is
reported as a cell in which the treatment never fired rather than as a null
effect. That check is what E81 through E83 lacked.

Because E86 has no replay gradient, its verified bank is populated only by what
the policy rediscovers. Banked-mode survival and verified bank size are
therefore reported as outcomes of this arm, not as controls.

## Integrity and failure policy

All 25 cells are submitted held from one hash-bound runtime snapshot and are
released only after their scheduler environments pass an exact audit. A
malformed environment, a snapshot diverging from the E79 snapshot outside the
three declared files, a missing or mismatched E79 pair, a pair straddling two
placements, a nonzero applied replay gradient, a duplicate run directory, a
non-finite loss, proposal or novelty leakage, a traceback, a PantryPlan cell
that fails the mechanism gate, or a missing pass-8 endpoint fails closed.
Infrastructure interruption may resume only from the same run's hash-bound
checkpoint and exact bank, semantic predictor, optimizer, data-cursor, and
request-stream state. No failed scientific run is silently replaced or
excluded.
