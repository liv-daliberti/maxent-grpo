# E30: fixed-draw evaluation variance for 0.5B free-form ModeBench

**Status: FROZEN — prospective diagnostic (2026-07-22, before evaluation).**

## Question

How much of the apparent bumpiness in the clean 0.5B free-form Dr.GRPO and
conditional-token MaxEnt trajectories is attributable to the stochastic K=8
evaluation rather than training dynamics?

## Frozen design

- Re-evaluate the terminal exported models from the clean E22-v2 Dr.GRPO
  controls and clean E27 higher-target conditional-token MaxEnt treatments.
- Environments: Countdown easy3 and graph coloring; use every frozen
  `multi_answer` evaluation prompt.
- Training seeds: 43, 44, and 45 in both arms and environments.
- Evaluation draws: four independent, fixed seeds 1001, 1002, 1003, and 1004.
- Each draw uses K=8, temperature 1, `top_p=1`, the original response budget,
  prompt template, verifier, terminal checkpoint, and the V0 vLLM engine used
  by the original E22/E27 inline evaluations (`VLLM_USE_V1=0`).
- Retain every completion, reward, normalized answer key, token length,
  prompt-level metric, per-draw summary, and checkpoint identity.
- The compatibility headline for future inline evaluations is the arithmetic
  mean of the four K=8 draw estimates. Report draw SD, SE, minimum, maximum,
  and all four raw values; do not smooth evaluation or training curves.

## Analysis

For each environment and arm, show all 12 terminal draw estimates (three
training seeds times four fixed evaluation seeds). Separate Monte Carlo
variation across evaluation seeds from between-training-seed variation.
Estimate the MaxEnt-minus-Dr.GRPO contrast with training-seed and evaluation-
seed fixed effects and prompt-clustered CR1 uncertainty; retain the pre-existing
run-cluster robustness calculation. Outcomes are pass@8, mean@8, coverage@8,
distinct@8, and deterministic greedy pass@1. This is a diagnostic, not a new
confirmatory efficacy test.

No curve smoothing, cherry-picking of evaluation seeds, or replacement of an
unfavorable draw is permitted.

## Results (added after evaluation)

Both preregistered evaluations completed under vLLM V0 (Slurm jobs 30046468
and 30046469). The complete raw attempts, per-prompt metrics, four draw-level
summaries, and aggregate findings are retained under
`var/artifacts/freeform_05b_repeated_eval_v1/`. An earlier pair of launches
selected vLLM V1; they were stopped before any metric inspection and their
partial output was quarantined unchanged under
`var/artifacts/freeform_05b_repeated_eval_v1_aborted_vllm_v1_30046466_30046467/`.

Evaluation Monte Carlo variation was small. For conditional-token MaxEnt,
pass@8 MC SD was 0.0050 on Countdown and 0.0040 on graph coloring, whereas
the mean absolute adjacent-checkpoint differences in the original raw
trajectories were 0.0703 and 0.0571. Thus the visible checkpoint variation is
about 14× larger than terminal evaluation noise in both environments. Across
the other reported metrics, adjacent-checkpoint variation was also generally
much larger than MC noise. The fixed draws expose uncertainty, but do not
explain away or smooth the trajectory bumpiness.

At the terminal checkpoints, MaxEnt minus Dr.GRPO was +0.0944 pass@8 and
+0.0286 coverage@8 on Countdown, and +0.2326 pass@8 and +0.0612 coverage@8
on graph coloring. Greedy pass@1 confidence intervals crossed zero in both
environments; graph mean@8 was also effectively unchanged. These results
support improved sampled success and mode diversity, not a claim that every
individual completion is more accurate. With only three independently
trained runs per arm, prompt-clustered intervals and six-cluster run
robustness should not be treated as additional training-run replication.
