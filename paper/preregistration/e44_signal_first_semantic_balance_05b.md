# E44: signal-first semantic balance at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

**Execution-only amendment (2026-07-23).** The first six held-and-released
`v1` jobs (`30072531--30072536`) exited before learner startup because
`ops/run_experiment.sh` did not yet dispatch the frozen
`signal_first_semantic_balance` variant. They produced no training or
evaluation observations. The missing dispatch and a matching launcher
preflight were added; the scientific protocol, seeds, budgets, estimands, and
gates below are unchanged. The replacement cohort is identified as `v2`, and
the failed `v1` identity, manifests, and logs remain preserved.

## Motivation and status

E43 applies a success-conditioned signed semantic-Shannon advantage outside
Dr.GRPO's task-reward centering. Its early trajectories were inspected and the
cohort was stopped before one complete training pass: the semantic actuator was
frequently inactive because sampled groups contained no eligible success, and
when active its RMS was small. E44 is therefore exploratory and
outcome-informed, not an independent confirmation of E43 or the broad
semantic-diversity hypothesis.

E44 tests the smallest combined mechanism supported by the prior program:

1. fixed-temperature xDr uses ordinary task advantage to concentrate detached
   candidate aggregation on sampled reward-bearing responses; and
2. E43's success-only signed semantic advantage balances rare versus common
   verified correct modes.

These two signals are deliberately kept on separate channels. The semantic
advantage is added once to the actor advantage, after Dr.GRPO task centering.
xDr weights are computed only from the task advantage captured before semantic
augmentation. Thus semantic novelty cannot be counted once in the policy
advantage and a second time in the aggregation weights.

E44 launches one new treatment arm on graph coloring and Countdown. MATH is not
included: E39 showed no final-answer diversity effect, and E43's stopped MATH
cohort had almost no eligible semantic signal. A MATH safety extension requires
a separate frozen gate after E44 ModeBench outcomes.

## Frozen combined update

For row `i` in a prompt group of size `G=16`, ordinary binary task reward is
centered exactly as in the retained Dr.GRPO controls:

`A_i_task = r_i - mean_j(r_j)`.

Let `T_i` be active response tokens and `T_max=192`. E44 fixes xDr temperature
`tau=0.05` and computes detached aggregation utility and row multiplier

`U_i_task = A_i_task * T_i / T_max`,

`w_i = G * softmax_i(U_i_task / 0.05)`.

Loss-masked rows are excluded from the softmax; active multipliers sum to the
number of active rows. An all-equal-reward group therefore receives uniform
weights. xDr cannot fabricate reward signal in an all-wrong group.

The semantic predictor and advantage are exactly E43:

- prompt-local history contains active, parseable, reward-positive outcomes
  only;
- current support uses eligible leave-one-out peers plus persistent history
  and one unseen bucket;
- pseudocount is `1`;
- surprisal is clipped at `5.0`;
- raw scale is `0.10 / 5.0`;
- wrong, unparseable, and loss-inactive rows receive zero semantic pressure;
- eligible semantic advantage is clipped symmetrically to `[-0.05, 0.05]`;
- history updates only after every row in the group is scored.

Writing `A_i_sem` for that frozen E43 semantic advantage, the actor advantage is

`A_i_actor = A_i_task + A_i_sem`.

The PPO/Dr.GRPO row loss uses `A_i_actor` and is multiplied by detached `w_i`.
No second group centering is applied. The task reward entering Dr.GRPO is
unchanged. No direct token entropy, sequence entropy, conditional-token
MaxEnt, outcome-collision reward, DIAYN option, SEED scaling, xDr controller, or
adaptive temperature is active.

## Cohort

- Model: local immutable
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Tasks:
  - graph coloring, 192 training prompts and 96 neutral evaluation prompts;
  - Countdown easy3, 384 training prompts and 128 neutral evaluation prompts.
- New arm: `signal_first_semantic_balance`.
- Training seeds: `43, 44, 45`.
- Group size: `16`.
- Budget: ten complete prompt-pool passes.
- Optimization: learning rate `2e-7`, one PPO epoch, `beta=0`, maximum norm
  `1`, rollout temperature `1`, top-p `1`, maximum response length `192`.
- Prompt: unrestricted neutral `qwen_boxed`.
- Checkpointing, actor synchronization, watchdog recovery, data, and resource
  topology match E43 ModeBench.

The six fresh jobs use prefixes:

- `gce44_signal_first_semantic_balance_05b_v2`; and
- `cde44_signal_first_semantic_balance_05b_v2`.

## Evaluation and estimands

Evaluate at initialization, every quarter pass, and ten passes. Every boundary
uses deterministic greedy pass@1 plus four fixed temperature-1, K=8 draws with
seeds `370100--370103`. Retain raw responses and draw-level values.

Primary quality endpoints are neutral pass@8 and mean@8, guarded by greedy
pass@1. Primary breadth endpoints are semantic coverage@8 and
distinct-correct@8. Report:

- terminal paired differences versus frozen E37 Dr.GRPO;
- trajectories and area under the coverage/distinct curves through ten passes;
- the latest common-horizon descriptive contrast with stopped E43;
- per-seed directions, not only pooled means; and
- time to fall below 125% of each seed's terminal Dr.GRPO coverage.

Mandatory mechanism telemetry includes task-only xDr weight-advantage mean/RMS,
xDr effective aggregation count and incorrect-mass share, semantic
positive/negative/zero fractions, semantic eligibility, semantic RMS, tracked
correct modes, response length, invalid fraction, and no-EOS fraction.

## Frozen interpretation

E44 supports the combined mechanism only if, in both tasks:

1. mean pass@8 and coverage@8 exceed paired E37 Dr.GRPO at ten passes;
2. at least two of three seeds improve both pass@8 and coverage@8;
3. mean@8 and greedy pass@1 each decline by no more than `0.03`; and
4. telemetry confirms xDr weights came from task-only advantages while
   semantic pressure was restricted to eligible successes.

Graph-only success is domain-specific evidence. Failure on all-zero groups does
not falsify xDr, because uniform weighting there is part of the frozen design.
No result may be described as confirmatory because E44 was selected after
inspection and cancellation of E43.

No E44 outcome may change the temperature, semantic coefficient, cap,
eligibility, support, task set, seed, endpoint, evaluation cadence, or success
rule above.
