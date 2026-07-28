# E41: separately centered predictive semantic-Shannon advantage at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Motivation and status

E38 adds a bounded predictive semantic-surprisal reward before Dr.GRPO's
current-group reward centering. Its realized semantic contribution is
therefore removed whenever all candidates in the group receive the same
bonus, even when the prompt-local predictive distribution assigns meaningful
probability to other outcomes.

E41 tests a narrowly targeted alternative. It keeps E38's prompt-local
predictor, answer key, leave-one-out construction, smoothing, clipping,
coefficient, history update, model, data, seeds, optimizer, and neutral
evaluations. Instead of treating predictive surprise as terminal reward, E41
forms a separately centered semantic policy-gradient advantage under each
row's detached predictive distribution, then adds it after ordinary Dr.GRPO
task-reward centering.

This treatment was selected after inspecting E37/E38 mechanism telemetry and
is exploratory rather than an independent confirmatory replication. E41
launches no new control, E37 collision, E38 predictive-Shannon, or E39 math
comparator. Those frozen trajectories are reused.

## Frozen predictive distribution

For prompt `x`, let `n_x(a)` be the exact count of canonical outcome `a` from
all completed earlier rollout groups for that prompt. In a current group of
size `G=16`, let `m_-i(a)` be the count among the other 15 candidates.

For row `i`, the explicit support is the union of historical outcomes and
outcomes observed among its leave-one-out peers, plus one reserved unseen
bucket. With pseudocount `alpha=1`, the detached posterior-predictive
distribution `q_i` is exactly the E38 distribution:

- an explicit outcome `a` has numerator `n_x(a) + m_-i(a) + 1`;
- the unseen bucket has numerator `1`; and
- the denominator is
  `N_x + G - 1 + alpha * (K_-i + 1)`.

Every observed final-answer outcome participates. Parse failures map to one
shared `INVALID` key. The stable prompt identity is the SHA256 digest of the
unpadded prompt token IDs. The full current group is added to history only
after every row has been scored. Predictor history is included in
optimizer-resumable checkpoints and restored exactly.

## Frozen semantic advantage

Let the clipped realized surprisal be

`s_i = min(-log q_i(a_i), S)`,

with frozen clip `S=5.0`. Compute the predictive clipped-surprisal expectation

`h_i = sum_a q_i(a) * min(-log q_i(a), S)`,

where the sum includes every explicit outcome and the unseen bucket. E41's
detached semantic advantage is

`A_i_sem = (0.10 / 5.0) * (s_i - h_i)`.

This is centered under the row-specific predictive distribution:
`E_{a ~ q_i}[A_i_sem]=0`. It is deliberately not centered by the empirical
mean of the current rollout group. Consequently, an all-identical sampled
group can retain a nonzero semantic learning signal.

Ordinary Dr.GRPO task rewards are centered exactly as in the reused controls:

`A_i_task = r_i - mean_j(r_j)`.

The policy-gradient advantage is

`A_i_E41 = A_i_task + A_i_sem`.

No second centering is applied after this addition. The semantic advantage is
in `[-0.10, 0.10]`; because a correct-versus-incorrect task-advantage
difference is one, the semantic term cannot reverse correctness ordering
between two candidates.

The new arm and `OAT_ZERO_VARIANT` are both
`semantic_shannon_advantage`. It pins:

- `OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10`;
- `OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0`;
- `OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0`; and
- `OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1`.

The retained E38 `semantic_shannon` variant explicitly pins the final flag to
zero. E41 contains no latent instruction, valid-answer catalogue, learned
classifier, correctness filter, direct token entropy, MaxEnt controller, or
canonical-action policy.

## Frozen domains and training contract

The nine fresh treatment runs comprise:

1. Countdown easy3, seeds `43, 44, 45`;
2. graph coloring `multi_answer`, seeds `43, 44, 45`; and
3. MATH12K-384 rows 0 through 383, seeds `43, 44, 45`.

All use:

- Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- group size 16 and ten complete prompt passes;
- learning rate `2e-7`, one PPO epoch, `beta=0`, maximum gradient norm 1;
- rollout temperature 1 and `top_p=1`;
- neutral free-form prompts;
- one GPU, vLLM colocated with the learner, and learner microbatch one; and
- one optimizer-resumable checkpoint per prompt pass, retaining the newest
  two, with watchdog requeue and exact predictor-history restoration.

Countdown and graph coloring reuse E37/E38 data, request structure, and
32-GB/four-CPU A5000-class resources. MATH reuses the exact E39
MATH12K-384/MATH-500 materialization, request structure, and 64-GB/eight-CPU
A5000-class resources.

## Frozen neutral evaluations

Countdown and graph coloring use the exact E37/E38 neutral evaluation:

- ordinary `qwen_boxed` prompts;
- greedy pass@1;
- four independent K=8 sampled draws at temperature 1;
- draw seeds `370100, 370101, 370102, 370103`; and
- quarter-pass evaluation cadence.

Within each task and seed, every E41 step-zero scalar metric and raw outcome
must agree exactly with the corresponding E37/E38 initialization.

MATH uses the exact E39 neutral evaluation:

- ordinary `qwen_math` prompts;
- all 500 held-out MATH-500 rows;
- greedy pass@1 and one deterministic K=8 draw with seed `390100`;
- initialization and passes 2, 4, 6, 8, and 10; and
- no duplicate terminal evaluation after the pass-10 weights were evaluated.

Within each seed, every E41 MATH step-zero scalar metric and raw outcome must
agree exactly with E39.

## Comparator and release integrity

The analytical prefixes are:

- `cde41_semantic_shannon_advantage_05b_v1`;
- `gce41_semantic_shannon_advantage_05b_v1`; and
- `mte41_math12k_384_semantic_shannon_advantage_05b_v1`.

The launcher audits and records immutable hashes for the E37, E38, and E39
identities and manifests. It materializes or audits the exact E39 math
boundary, freezes source and execution snapshots, and records hashes of the
protocol, launcher, source, execution surface, data identity, and comparator
artifacts.

All nine jobs are submitted held. Before release, the launcher verifies the
three tasks times three seeds, the E41 arm/variant and separate-advantage flag,
all excluded mechanisms, data, prompt, lengths, evaluation requests,
checkpoint/recovery fields, and task-specific placement resources. Any
partial or failed held cohort is cancelled and quarantined. Only the complete
audited nine-job cohort is released.

No E41 outcome may alter the coefficient, clip, pseudocount, predictive
support, data, cadence, seed, endpoint, or evaluation request.

## Reporting and interpretation

Report E41 head-to-head with reused Dr.GRPO, E37 semantic collision, and E38
predictive semantic-Shannon trajectories for Countdown and graph coloring,
and with the corresponding E39 arms for MATH.

Primary quality endpoints are neutral pass@8 and mean@8, guarded by greedy
pass@1. Countdown and graph coloring additionally report semantic coverage
and distinct-correct@8. Mandatory mechanism telemetry includes realized and
expected clipped surprise, semantic-advantage mean/min/max/RMS, predictor
normalization error, unseen fraction, invalid fraction, history size, and
semantic-to-task advantage RMS.

MATH remains a single-answer final-outcome stress test. Its apparent answer
diversity is mostly diversity among incorrect hypotheses and cannot establish
diverse correct proof strategies.
