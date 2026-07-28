# E4 prospective analysis plan: entropy-feedback xDr temperature

**Status: EXPLORATORY, PARTIALLY LAUNCHED (amended 2026-07-16).** The design was
created after inspecting the 0.5B long-run collapse and the ongoing 3B collapse
telemetry. The 7B graph-coloring campaign was launched first and its
initialization evaluations were observed before the 0.5B/3B extension below
was added. The extension is therefore mechanism-informed and post-hoc, not a
confirmatory test independent of those observations.

**Scale amendments, 2026-07-16:** Qwen2.5-7B-Instruct was added as the next
same-family scale after 3B before any E4 outcome was observed. The original
unlaunched 1.5B row was later replaced by 0.5B so the feedback experiment
aligns with the paper's actual 0.5B/3B/7B compute-divergence ladder. There is no
Qwen2.5 5B member in the maintained model workflow.

**Five-pass display and execution amendment, 2026-07-17:** after inspecting
trajectories from the already-running long-horizon jobs, the maintained
workflow was standardized to at most five prompt-pool passes at every scale
and in both environments. The canonical trainer now enforces that ceiling even
when an old launcher or restart manifest requests more. The compute-divergence
figure excludes archived points after pass five. This is a post-outcome scope
change, so the five-pass contrasts are descriptive; the original longer
horizons and estimand remain below as historical provenance rather than being
silently rewritten.

**Operational recovery amendment, 2026-07-19:** all affected 7B processes
failed at exactly learner step 1024 in vLLM 0.8.4's CuMem sleep allocator,
after landing step 896. The actor stack was identical across methods and seeds
(`none_dealloc` in `python_unmap_and_release`); this was not an objective or
outcome-triggered stop. Same-stamp recovery keeps vLLM resident
(`OAT_ZERO_VLLM_SLEEP=0`) with a 0.10 KV reservation and the already validated
two-A6000, microbatch-4, optimizer/activation-offload layout. Data, G=32,
learning rate, batch size, five-pass ceiling, and quarter-pass evaluation are
unchanged. Every deficient seed-arm is recovered uniformly.

## Question

Fixed-temperature xDr slows entropy loss and improves coverage at matched token
entropy, but it does not directly stabilize the policy. Can a label-free
feedback rule over xDr's candidate-aggregation temperature preserve the
informative-group and answer-mode coverage gains later in training?

This is not a token-entropy bonus. Token entropy is only the controller's
sensor; the intervention remains candidate-level credit assignment.

## Arms and controller

At each of Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct, and
Qwen2.5-7B-Instruct, run seeds 43, 44, and 45 for exactly three arms:

1. Dr.GRPO (uniform candidate aggregation; tau = infinity).
2. xDr.GRPO with fixed tau = 0.05.
3. xDr.GRPO initialized at tau = 0.05 with entropy feedback.

For the feedback arm, the first 64 globally averaged post-update token-entropy
observations run at tau = 0.05. Within each run,

```text
target = 0.8 * mean(first 64 entropy observations)
entropy_ema_t = 0.9 * entropy_ema_(t-1) + 0.1 * entropy_t
deficit_t = max(target - entropy_ema_t, 0)
tau_(t+1) = clip(0.05 * exp(-20 * deficit_t), 0.005, 0.05).
```

Thus feedback is one-sided: it leaves the known fixed-tau treatment unchanged
while entropy is above target and strengthens xDr's non-uniform aggregation
only after entropy falls below target. The globally reduced observation keeps
all learner ranks synchronized. The updated tau applies to the next update and
is checkpointed. No semantic labels, answer-mode counts, correctness-derived
controller state, or token-level entropy reward enter the rule.

The rule is a hypothesis, not an entropy constraint: changing aggregation tau
does not mathematically guarantee that entropy will rise or plateau.

## Matched recipe

At 0.5B and 3B, the feedback-only jobs reuse the exact recipes of the landed E1
Dr.GRPO and fixed-xDr curves rather than duplicating those controls. Graph
coloring uses the 192-prompt pool with G=16 for 24 passes at 0.5B and the
1,024-prompt pool with G=32 for 16 passes at 3B. Countdown uses the 384-prompt
easy3 pool for 24 passes with G=16 at 0.5B and 16 passes with G=32 at 3B.
Evaluation cadence, learning rate, rollout sampling, and seeds are inherited
unchanged from the corresponding E1 launcher. Disabling optimizer-state
archives for the extension changes storage only, not training.

The 7B graph-coloring campaign uses the 1,024-prompt pool, G=32, constant
learning rate 2e-7, and 16 pool passes. All three 7B arms were launched fresh.
Each job spans two A6000s; optimizer and activation state are CPU-offloaded.
Inline metrics and evaluations retain the trajectory, with one rolling model
snapshot and no ZeRO optimizer archive to protect shared storage.

The planned seed-9001 technical smoke did not complete its 64-update controller
warmup because OAT capped queries by `max_train`; all smoke attempts are
excluded from outcome analysis. The full seeds 43--45 campaign was subsequently
launched by explicit user decision after two-GPU initialization and training
were verified. This deviation and its timing remain recorded in
`ops/exp_scaling/CAMPAIGN_LOG.md`.

## Outcomes fixed before launch

The primary contrast at each scale is feedback xDr minus fixed-tau xDr on mean
multi-answer coverage@8 over the final four completed passes (21--24 at 0.5B;
13--16 at 3B and 7B), averaged within seed before the across-seed contrast.
Report the paired seed contrast, all seed trajectories, and an uncertainty
interval; do not promote the Dr.GRPO contrast to primary after seeing results.

Secondary mechanism outcomes are:

- late-window pass@8, distinct correct modes@8, and mean@8;
- token-entropy trajectory and its final-four-pass slope;
- fraction of informative mixed-reward groups, all-correct groups, and all-zero
  groups;
- time to sustained entropy below 80% of the fixed arm's own warmup mean;
- tau trajectory, activation time, and fraction of updates at the minimum;
- coverage-versus-entropy phase curve and effective aggregation count.

Greedy accuracy and mean@8 are guardrails. A late coverage gain accompanied by
a material loss on either is reported as a tradeoff, not stabilization for
free. The fixed training horizon is retained unless a run fails for an
operational reason; there is no result-dependent early stopping.

## Interpretation

- Higher late coverage, a non-declining late entropy trajectory, and a retained
  mixed-group fraction would be evidence that feedback stabilizes xDr under
  this recipe. It would still not establish general collapse immunity.
- Higher late coverage while entropy keeps declining means stronger collapse
  resistance or delay, not stabilization.
- No improvement over fixed xDr means aggregation tau alone is not a sufficient
  control actuator for the observed entropy loss.
- Frequent saturation at tau = 0.005 with continued collapse is a particularly
  clear actuator-failure result; the next intervention would need an explicit
  constraint or a different control variable, not post-hoc retuning of E4.

## Maintained entry points

```bash
bash ops/exp_scaling/launch_entropy_feedback_extension.sh all
OAT_ZERO_7B_ANALYTICAL_APPROVED=1 \
  bash ops/exp_scaling/launch_e4_tau_control_7b.sh
```

These commands submit jobs. The 7B command was executed before this amendment;
the feedback-only 0.5B/3B extension is recorded in the campaign log at launch.

After completion, the shared curve parser recognizes all three arm names:

```bash
python ops/exp_scaling/parse_scaling_curve.py --stamp-prefix cde4_taucontrol_05b_feedback_v1
python ops/exp_scaling/parse_scaling_curve.py --stamp-prefix cde4_taucontrol_3b_feedback_v1
python ops/exp_scaling/parse_scaling_curve.py --stamp-prefix gce4_taucontrol_05b_feedback_v1
python ops/exp_scaling/parse_scaling_curve.py --stamp-prefix gce4_taucontrol_3b_feedback_v1
python ops/exp_scaling/parse_scaling_curve.py --stamp-prefix gce4_taucontrol_7b_full_2xa6000_v1
```

Its preview reports a separate Dr.GRPO gap for fixed and feedback xDr; the
registered late-window paired-seed contrast is then computed from the emitted
tidy JSON without selecting checkpoints by outcome.
