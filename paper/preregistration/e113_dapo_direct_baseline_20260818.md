# E113: DAPO direct comparative baseline

Date frozen: 2026-08-18, before submitting any E113 job and without inspecting
any E113 outcome.

## Question and rationale

The paper already compares ordinary GRPO/Dr.GRPO, UCPO (online redistribution
without memory), and RLEP-Dr (trajectory replay). It lacks a strong recent
on-policy RLVR optimizer whose mechanism was explicitly motivated by entropy
collapse. E113 adds DAPO (Yu et al., 2025) as that comparator. DAPO is treated
as a complete named recipe, not as an asymmetric-clipping ablation.

Primary hypothesis: at matched accepted optimizer updates, DAPO will retain
more correct-mode diversity than its completed family-matched control. The
direction and magnitude of task-accuracy changes are not assumed. Dynamic
sampling spends a variable number of rollout queries by design; query cost is
therefore a separately reported mechanism/cost endpoint, not force-matched
away.

## Frozen scientific cohort

- Models: Qwen2.5-0.5B-Instruct and Falcon3-1B-Instruct (Falcon revision
  `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf`).
- Domains: graph coloring, countdown, Python factors, MathIR, and pantry plan.
- Seeds: Qwen 43--47; Falcon 55--59.
- Cells: 5 domains x 5 seeds x 2 families = 50 DAPO runs.
- Matched controls: the 25 completed E78 Qwen control cells and the 25
  completed E79 Falcon control cells, paired by model family, domain, and seed.
- Frozen training rows: 384 per domain; eight complete accepted-update passes,
  3,072 accepted prompt groups per cell.
- Group size: 16; one prompt group per optimizer update; one PPO epoch; no KL.
- Learning rate and all task/model prompt and length surfaces are inherited
  cell-for-cell from the paired controls.
- Checkpoints/evaluations: every 192 accepted groups (half-pass), plus initial
  and terminal evaluation, matching the paired controls.

Two bounded learner smokes, one per model family, are non-scientific. Each uses
at most 32 training prompts and a 640-row query ceiling. All family cells
depend `afterok` on their smoke. Smoke rows must never enter paper estimates.

## Frozen DAPO implementation

E113 implements all four optimization components in the primary DAPO recipe:

1. Standard GRPO group-standard-deviation advantages (`critic_type=grpo`).
2. Clip-Higher with lower clip epsilon 0.20 and upper clip epsilon 0.28.
3. Token-level policy-loss aggregation over all active response tokens.
4. Dynamic sampling plus soft overlong reward shaping.

Dynamic sampling accepts a 16-row prompt group iff its binary verifier rewards
contain at least one zero and at least one one. A rejected group is charged to
`query_step` and `prompt_consumed`, but is discarded before the replay buffer
and PPO. Because the campaign's train batch contains one prompt, each retry
selects a deterministic prompt from the same frozen domain dataset using only
the registered seed, checkpointed learner step, and generation-batch index.
At most ten generation batches may be drawn for one accepted update, matching
the official DAPO recipe ceiling. Failure to obtain an eligible group by batch
ten fails the cell closed; it does not trigger tuning or replacement.

The soft overlong term is zero through the first 80% of each domain's frozen
generation ceiling, decreases linearly to -1 over the final 20%, and is added
to the task reward only for advantage construction. Raw binary task reward
remains the accuracy and dynamic-filter signal. The worst-case scientific
query ceiling is `3072 * 16 * 10 = 491520` rows per cell. Each smoke is capped
at `4 * 16 * 10 = 640` rows.

All entropy bonuses, xDr weighting, UCPO, RLEP, canonical replay/banks,
proposal mechanisms, passive discovery tracking, and other experimental
objectives are disabled. DAPO uses temperature 1.0/top-p 1.0 training rollouts
and the same deterministic and sampled evaluation contracts as its controls.

## Frozen analysis

Primary paired endpoints use the terminal accepted-update checkpoint:

- correct-mode entropy and normalized correct-mode entropy;
- correct-mode coverage / modes recovered;
- task accuracy;
- collapse incidence under the paper's already frozen collapse rule.

Report per-domain paired seed differences and family-level paired summaries,
with the same uncertainty procedure as the corresponding direct-comparator
table. Secondary mechanism/cost endpoints are generated batches per accepted
group, rejected all-zero/all-one group counts, total sampled rows, clipping
fractions, active-token denominator, and overlong-penalty incidence.

No E113 estimate may enter the manuscript until the 50-cell cohort passes the
existing terminal-artifact and provenance audits. Before terminal completion,
the paper may describe only the registered method and scheduler state and must
label results pending.

## Launch and integrity contract

- Submit every job held; audit the scheduler-expanded environment; write one
  atomic immutable ledger; only then release the smokes and dependent science
  cells.
- Freeze the complete current `src/oat_drgrpo` and `ops` trees in one
  content-addressed runtime snapshot and bind its identity in the ledger.
- Bind this protocol, launcher, E78 ledger, E79 ledger, and source manifest by
  SHA-256.
- Refuse pre-existing target directories or ledgers.
- On any partial submission/audit failure, cancel only newly submitted E113
  jobs and do not write a released ledger.
- Infrastructure-only requeue may resume from the most recent audited
  checkpoint without changing the scientific environment. Scientific changes
  require a dated amendment written before release.

## References used to freeze the recipe

- Yu et al. (2025), *DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale*, arXiv:2503.14476.
- The authors' official `verl-recipe/dapo` configuration and README, consulted
  for the 0.20/0.28 clip pair and ten-generation-batch ceiling.
