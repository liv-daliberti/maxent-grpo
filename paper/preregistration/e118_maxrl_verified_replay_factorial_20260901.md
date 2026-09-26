# E118: MaxRL × verified replay matched factorial

Date frozen: 2026-09-01, before any E118 training submission or endpoint read.

## Question

Does verified canonical replay preserve semantic support beyond the effect of
replacing the ordinary Dr.GRPO task estimator with finite-rollout binary MaxRL?

## Design

E118 extends the completed E78 Qwen2.5-0.5B-Instruct cohort at seeds 43 and 44
over all five registered domains: Countdown, Graph Coloring, Python Factors,
MathIR, and PantryPlan. The completed E78 Dr.GRPO and ReplayGRPO cells are the
two existing factorial arms. E118 adds exactly two arms:

- maxrl: binary MaxRL task objective with compute-matched replay traversal and
  an exactly zero replay derivative.
- replay_maxrl: the identical binary MaxRL task objective with the established
  verified_likelihood_per_rollout replay derivative at alpha 0.10.

This produces 5 domains × 2 seeds × 4 effective arms. Only the 20 new MaxRL
cells are submitted. No existing comparator is rerun.

## Frozen invariants

- Base model: Qwen2.5-0.5B-Instruct, identical frozen checkpoint to E78.
- Train/evaluation datasets and prompt templates: inherited per domain from
  the corresponding E78 cell.
- 384 training prompts, 8 passes, 3,072 optimizer updates.
- 16 on-policy rollouts per prompt; one PPO epoch; beta 0; learning rate 2e-7.
- Evaluation/checkpoint interval 192; E78 mode-coverage evaluation unchanged.
- MaxRL is applied only to fresh binary terminal task rewards.
- Replay exemplars are not inserted into a MaxRL rank group. Replay remains a
  separately differentiated current-policy verified-likelihood objective.
- All-failure and all-success groups have zero centered MaxRL task advantage.
- Any non-binary task reward fails closed.
- Semantic-MaxEnt, token-entropy, xDr, DAPO, RLEP, and counterfactual proposal
  objectives are disabled in both new arms.

## Estimands

Within domain and seed:

1. MaxRL - Dr.GRPO: fresh-rollout objective effect without replay.
2. ReplayMaxRL - MaxRL: replay effect under MaxRL.
3. ReplayGRPO - Dr.GRPO: existing replay effect under Dr.GRPO.
4. Difference in differences: whether replay and MaxRL are complementary.

Primary endpoints use the already registered E78 evaluation surface: greedy
verified accuracy, sampled verified accuracy, verified semantic mode coverage,
and retained discovered support across checkpoints. No endpoint may alter
training, placement, stopping, or inclusion.

## Compute and stopping

The 20 new cells are independent jobs submitted to account/partition mltheory
on node302 A100 GPUs. Scheduler availability determines start order. Every cell
runs to the frozen 3,072-update horizon; failures are resumed from the same run
directory and remain the same scientific cell.

## Interpretation boundary

Because the task reward is binary, the new estimator is MaxRL, exactly the
binary special case of ArgMaxRL. This experiment does not claim evidence about
continuous-reward ArgMaxRL.
