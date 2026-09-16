# E120: within-bank target-weight mechanism ablation

Preregistered 2026-09-02 before any E120 training outcome was generated or
inspected.

## Question and sole treatment difference

E120 tests whether uniform weighting over discovered canonical keys, rather
than online replay itself, causes ReplayDr.GRPO's breadth effect. For the
selected prompt bank (B_x), both arms materialize and score the identical
retained exemplar for every key. The historical arm minimizes

[
L_{uniform}=-\sum_{c\in B_x}|B_x|^{-1}s_c ,
]

and the new arm minimizes

[
L_{frequency}=-\sum_{c\in B_x}
  \frac{n_c}{\sum_j n_j}s_c ,
]

where (s_c) is the same response-length-normalized current-policy exemplar
log likelihood and (n_c) is the cumulative number of validator-positive
fresh on-policy observations of key (c).

The implementation represents both vectors with a group-budget-preserving
weight whose entries sum to (|B_x|); the pre-existing loss normalizes by
their sum. This is algebraically the pair above and keeps the total replay
coefficient unchanged.

The switch is
`online_canonical_replay_key_weighting={uniform,fresh_frequency}`. The
`uniform` branch aliases the historical materialized mass-weight tensor and
performs no new arithmetic. The `fresh_frequency` branch changes only the
within-bank vector passed to the existing verified-likelihood loss.

## Fixed contract

The following are identical to the completed uniform comparator for every
model/domain/seed cell:

- online bank membership, capacity 16, admission rules, and one deterministic
  retained exemplar per canonical key;
- one nonempty prompt bank per optimizer update under the checkpointed global
  round-robin scheduler;
- all materialized exemplars, their ordering, response-length normalization,
  and the two score passes;
- replay coefficient 0.10 and Dr.GRPO per-rollout scaling
  ((G-1)/G^2), with (G=16);
- fresh Dr.GRPO advantages, optimizer, learning-rate schedule, checkpoint and
  evaluation cadence, prompts, seeds, sampling, and eight-pass horizon;
- no checkpoint selection, early stopping, or outcome-conditioned reruns.

Frequency counts are updated only by validator-positive fresh on-policy
rollouts. Replay rows never call the bank update. Proposal rows are disabled,
and frequency mode fails closed if a materialized key has count zero or if
proposal-priority mass weights are active.

## Registered blocks

The primary block contains 25 new frequency-weighted runs:

- Qwen2.5-0.5B-Instruct;
- Graph coloring, Countdown, Python factors, MathIR, and PantryPlan;
- seeds 43, 44, 45, 46, and 47.

The confirmatory scale block contains 20 new frequency-weighted runs:

- Falcon3-1B-Instruct, Graph coloring and PantryPlan, seeds 55--59;
- Qwen2.5-3B-Instruct, Graph coloring and PantryPlan, seeds 70--74.

The matched uniform comparators are the completed replay arms in E78, E79,
and E80R1. Reuse is admissible only if regression tests establish that the new
default `uniform` switch produces literal all-one target weights and leaves
the historical loss and gradient unchanged. No E120 result may be inspected
before this gate passes.

## Estimands and inference

The primary estimand is paired
(uniform-frequency) at pass 8 for
`distinct@8 - pass@8`. The co-primary safety endpoint is paired
(uniform-frequency) at pass 8 for `pass@8`. Report each
model/domain cell with its five paired seeds, the full Qwen-0.5B five-domain
mean, paired bootstrap 95% intervals, and all individual seed values.

Supporting endpoints are terminal raw `distinct@8`, terminal `mean@8`,
and the paired trajectory AUC through the registered checkpoints. These do
not replace or select the primary endpoint. No checkpoint is selected from
E120 outcomes.

Interpretation is fixed in advance:

- higher uniform breadth at comparable correctness supports canonical-key
  balancing as the mechanism;
- no meaningful difference means replay helps but key balancing has not been
  shown responsible;
- higher frequency-weighted performance requires revising the mechanism toward
  persistence or online discovery;
- an effect confined to Graph/PantryPlan is described as a finite-support,
  initially broad-domain result rather than a universal mechanism.

## Mechanism and identity telemetry

Every replay update records:

- each applied target weight and aligned fresh count in sorted canonical-key
  row order;
- target-vector normalized entropy and Gini;
- total target allocation to minimum-count and maximum-count keys;
- fresh count sum/min/max plus explicit zero replay/proposal count
  contributions;
- fingerprints of the selected prompt, bank membership, every outcome key,
  and the byte representation of the fresh advantage tensor;
- replay coefficient, per-rollout scale, score-pass count, materialized row and
  token counts, bank capacity, selected scheduler, and applied gradient norms.

The analysis must verify finite positive weights, per-bank weight sums equal
bank size, positive fresh counts in the frequency arm, exactly one replay
prompt when any bank is eligible, and the registered coefficient/scaling.
Identity fingerprints are diagnostic: after policies diverge they need not
match across arms, but they must align the recorded weights, keys, counts, and
advantages within every update.

## Compute and release policy

No PVL compute may be requested or used. E120 is restricted to the established
non-PVL `mltheory`/A100-A5000 placements and `allcs`/CS
A5000-A6000 placements inherited from E78/E79/E80R1. The launcher submits jobs
held, audits the effective scheduler record, and cancels the entire new cohort
if any partition, account, reservation, node feature, node name, GRES, command,
working directory, or exported environment contains the case-insensitive
substring `pvl`.

Science jobs are released only after source syntax, focused regression tests,
a 45-cell manifest audit, comparator-terminal checks, snapshot freezing, and
held-job scheduler audits succeed. Scheduler delay is not an analysis choice
and does not permit replacement seeds, altered placement, or endpoint
inspection.

