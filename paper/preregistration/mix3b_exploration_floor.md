# Pre-registration: Does candidate-level exploration lift the hard-stratum collapse floor?

**Registered:** 2026-07-15 (git commit timestamp is authoritative), before any
xDr.GRPO run on the mixed-difficulty pool was started.

## Motivating observation (already known at registration)

A single Dr.GRPO probe run (`cal3b_mix_grpo_s43`: Qwen2.5-3B-Instruct, G=32,
lr 2e-7, 2 epochs = 65,536 prompt consumptions on the 1,024-prompt
mixed-difficulty Countdown pool `var/data/cd_mix_probe`) learns the easy
stratum (greedy pass@1 0.609, pass@8 0.688) but collapses on the hard
multi-answer stratum: **pass@8 = greedy pass@1 = 0.023, distinct@8 = 0.023**
(eval seed 1001, K=8, final checkpoint). Eight samples recover nothing over
one; the failure is a collapse of the sampled candidate distribution, not a
capability ceiling. This observation motivated the present experiment and its
numbers were known before registration; the s43 baseline run is therefore
disclosed as pre-observed (see sensitivity analysis).

## Hypothesis

Candidate-level exploration (xDr.GRPO, tau = 0.05) trained on the same mixed
pool lifts hard-stratum sampled coverage above the Dr.GRPO collapse floor by
preserving multiple candidate modes during training on prompts where the
group carries multi-answer structure.

## Design (fixed before launch)

- Model / recipe: Qwen2.5-3B-Instruct; G=32; lr 2e-7; one optimization epoch
  per rollout batch; critic_type=drgrpo; beta=lambda=0; clip 0.2; T_max=192;
  temperature 1.0; bfloat16 — byte-identical launcher env to the probe run
  except the arm parameters below.
- Data: `var/data/cd_mix_probe` (1,024 mixed-difficulty Countdown training
  prompts); 2 epochs (65,536 prompt consumptions); leak-free held-out eval
  strata: `multi_answer` (hard, 128 prompts) and `easy_answer` (easy, 128).
- Arms: `grpo` (Dr.GRPO, tau = inf) and `xdr_tau0p05` (xDr.GRPO, tau = 0.05).
  No other arms.
- Training seeds: 43, 44, 45 per arm. `grpo_s43` is the pre-observed probe
  run and is reused as-is (not retrained).
- Evaluation: final saved checkpoint per run (rule unchanged from the paper);
  eval seeds 1001, 1002, 1003; K=8 samples at temperature 1.0, top-p 1.0,
  plus one greedy pass; the standard coverage-eval harness
  (`ops/eval_exact_answer_mode_coverage.py`).

## Primary outcome and analysis (fixed before launch)

- **Primary: hard-stratum (`multi_answer`) pass@8.** Linear probability model
  per Eq. 3 of the paper: treatment indicator + training-seed FE +
  evaluation-seed FE, prompt-clustered (CR1) standard errors,
  2 arms x 3 train seeds x 3 eval seeds x 128 prompts = 2,304 observations,
  128 clusters. The treatment coefficient is reported with 95% CI and raw
  p-value (single primary; no multiplicity adjustment).
- Secondary (Holm-adjusted within outcome, hard stratum): distinct@8,
  coverage@8, mean@8, greedy pass@1.
- Guardrail (Holm-adjusted): easy-stratum pass@8 and mean@8 — candidate-level
  exploration must not buy hard-stratum coverage at an easy-stratum accuracy
  cost.
- Robustness: run-clustered (arm x train seed) pass, as in the paper.
- Sensitivity: primary re-estimated excluding train seed 43 from both arms
  (removes the pre-observed baseline run and its matched treatment seed).

## Interpretation rules (fixed before launch)

- Positive primary (CI excluding 0) with flat-or-better easy-stratum
  guardrail: evidence that group-level exploration converts easy-stratum
  competence into hard-stratum coverage; reported as the mixed-pool result.
- Null or negative primary: reported as an honest boundary of the mechanism —
  candidate-level exploration does not rescue prompts whose rewarded modes
  the policy essentially never samples. No post-hoc arm additions or
  temperature changes will be reported as confirmatory.
- The baseline floor (pass@8 ~ 0.02 on 128 prompts) implies a detectable
  lift on the order of a few points; if the primary CI is too wide to
  distinguish a meaningful lift from zero, the result is reported as
  underpowered, not as evidence of absence.
