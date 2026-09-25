# E134 preregistration: Re:Dr against a tuned control

Date frozen: 2026-09-24, before submission of any E134 cell.

## The question this answers

A reviewer objects that the Dr.GRPO control is fragile: one prompt group of 16
per update, no KL, no entropy term and an unswept constant learning rate of
2e-7. The control loses `pass@8` against its own initialization in 6 of 15
comparisons (App. "Control behavior at fixed learning rates"), so part of
replay's gain may be rescue of a degenerate baseline rather than preservation
of diversity. The appendix already shows the gain survives restriction to the
controls that train (+.247 `pass@8`, +.136 PCMD over 6 comparisons, positive in
all six), and E129 already sweeps a reference-KL anchor. What no cohort does is
tune the control itself. E134 does, and then runs Re:Dr at the setting the
tuning picked.

## Stage 1: the control sweep (selection)

Nine control settings on the full Level-1 panel at Qwen2.5-0.5B. The baseline
is the completed E128 control and is not re-run.

| setting        | learning rate | prompts per update | entropy coef |
|----------------|---------------|--------------------|--------------|
| baseline (E128)| 2e-7          | 1                  | 0            |
| `lr5e8_b1`     | 5e-8          | 1                  | 0            |
| `lr1e7_b1`     | 1e-7          | 1                  | 0            |
| `lr5e7_b1`     | 5e-7          | 1                  | 0            |
| `lr1e6_b1`     | 1e-6          | 1                  | 0            |
| `lr2e7_b4`     | 2e-7          | 4                  | 0            |
| `lr5e7_b4`     | 5e-7          | 4                  | 0            |
| `lr1e6_b4`     | 1e-6          | 4                  | 0            |
| `ent1e3_b1`    | 2e-7          | 1                  | 1e-3         |

- **Learning rate** is `OAT_ZERO_LEARNING_RATE`, constant schedule as in E128.
- **Four prompts per update** holds the fresh-rollout budget fixed: 384 prompts,
  8 passes, 16 rollouts per prompt, as in E128. One update consumes four
  prompts' groups (64 rollouts), so there are 768 updates rather than 3,072.
  Five keys move together: `ROLLOUT_BATCH_SIZE` and
  `ROLLOUT_BATCH_SIZE_PER_DEVICE` to 4, `TRAIN_BATCH_SIZE` and
  `PI_BUFFER_MAXLEN_PER_DEVICE` to 64, and the save/resume cadence to 48
  updates, so checkpoints still fall every 192 prompts. The evaluation interval
  is counted in prompts and is unchanged. Low learning rates are not crossed
  with four prompts per update: a quarter of the updates at a rate already
  below the baseline cannot plausibly beat it.
- **Entropy bonus** is `OAT_ZERO_POLICY_ENTROPY_COEF`, which subtracts
  coef x (masked mean token entropy) from the loss. A cell is trusted only if
  `policy_entropy_loss` is nonzero in its training log.

Seeds 43 and 44 on all five domains: 8 new settings x 5 domains x 2 seeds = 80
cells, plus one smoke job per mechanical path (four prompts per update; entropy).

### Selection rule (registered)

For each domain, the tuned control is the setting with the highest terminal
`pass@8`, averaged over seeds 43 and 44, among the nine settings above,
measured by the same terminal evaluator as every other Level-1 cell. Ties
within .005 go to the higher terminal `mean@8`. A cell that diverges or
collapses keeps its measured value; it is not re-run or excluded. The rule
selects on correctness only, never on PCMD, so the tuning cannot favour
replay's metric.

Selection uses the same evaluation prompts the comparison reports, and the two
selection seeds reappear in stage 2. Both choices inflate the tuned control
(winner's curse), so they bias the comparison against replay. That is
intended.

## Stage 2: the comparison

For each domain whose selected setting is not the baseline:

- the tuned control on seeds 45, 46 and 47 (seeds 43 and 44 come from stage 1);
- Re:Dr at the identical setting on seeds 43 to 47: the E132 objective with the
  selected learning rate, batch and entropy keys layered on.

Where the baseline wins, the comparison is E128 against E132, which already
exist. At four prompts per update Re:Dr's code still admits exactly one replay
group per update, so replay rehearses a quarter as often per fresh prompt as
at the baseline. This is a dose reduction against replay and is kept rather
than patched.

## Registered predictions

1. **The gap persists over the tuned control.** The pooled paired Re:Dr minus
   tuned-control effect over 25 cells (5 domains x 5 seeds) is positive with a
   95% CI excluding zero, for both `pass@8` and PCMD.
2. **Tuning closes part of the `pass@8` gap, less of the PCMD gap.** The tuned
   control's `pass@8` exceeds E128 on the five-domain mean. Its PCMD stays
   below half of E132's PCMD effect over E128.

If prediction 1 fails on either metric, the paper reports the replay effect as
conditional on the untuned regime for that metric and scopes the claim to it.

## Reporting

Effects are reported as paired 95% CIs over seeds, not as counts of domains
passing a threshold. All stage-1 settings are reported, not only the winners.

## Frozen cohort

- Model: Qwen2.5-0.5B-Instruct, exactly as E128.
- Objective: `e78.fixed_objective("control")` with only the keys above moved;
  the replay traversal stays inert (`COMPUTE_ONLY=1`) in every control arm.
- Domains: Graph Coloring, Countdown, PythonFactors, MathIR, PantryPlan.
- Runtime: the snapshot E132 and E133 ran under,
  `diversity_comparators_40628bb13a366baa`.
- Placement: `cs` partition, `allcs` account, A5000, node203 or node204, as
  E128, E132 and E133.

## No source change

E134 declares no source change. Learning rate, prompts per update and the
entropy coefficient are existing training parameters.
