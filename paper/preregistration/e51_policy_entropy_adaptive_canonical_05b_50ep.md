# E51: policy-entropy-adaptive canonical exploration at 0.5B

**Status: FROZEN BEFORE CORRECTED RELAUNCH — 2026-07-24**

## Motivation and scope

E51 is the user-directed from-scratch replacement for E50 after its live
trajectories showed that uncapped Haarnoja control could drive the canonical
entropy coefficient far above the registered base dose. E50 remains immutable
historical evidence. No E50 checkpoint, optimizer state, bank, controller
state, or partial trajectory is resumed or pooled into E51.

The deferred controller direction was documented before E51 as a
policy-token-uncertainty controller: canonical alpha should respond to the
model's own entropy score rather than the cumulative entropy of its discovered
canonical bank. E51 activates that direction prospectively in all three
executable ModeBench domains:

- graph coloring;
- Countdown easy3;
- executable Python factors.

Each domain receives a fresh contemporaneous Dr.GRPO control and treatment at
seeds 43, 44, and 45.

### Corrected-relaunch record

The initial E51-v1 launch used the direct multiplier `m_t / h_ref`. Live
telemetry established that model entropy was falling in graph coloring and
Countdown, while this multiplier reduced alpha and therefore weakened the
intended counter-pressure. That cohort (`30074925--30074942`) is superseded and
is not pooled with the corrected cohort.

The user directed a from-scratch E51 relaunch with the inverse response below.
The corrected runs use fresh `v2` run stamps, source snapshot, controller state,
optimizer state, banks, outputs, manifests, and identity. The v1 artifacts are
retained only as superseded diagnostic evidence.

## Policy-entropy controller

Let `h_t` be `train/entropy`, the masked mean of the current model's
full-vocabulary next-token categorical entropy over active response positions
in optimizer round `t`. This diagnostic already has identical semantics in
control and treatment and adds no second model inference.

For the first `W=64` observations, the treatment uses `alpha_t=0.10` and
records

`h_ref = (1/W) * sum_{t=1}^W h_t`.

The controller also maintains

`m_t = 0.9 m_(t-1) + 0.1 h_t`,

with the first observation initializing `m_t=h_t`. After warmup, define the
model's entropy-deficit multiplier

`d_t = h_ref / m_t`

and set

`alpha_(t+1) = 0.10 * d_t`.

There is no explicit lower or upper projection and no numerical epsilon that
would impose an implicit ceiling. Alpha may fall below or rise arbitrarily
above 0.10 as the model's current entropy moves relative to its own warmup
reference. A nonpositive EMA after warmup fails closed because the unbounded
inverse is then undefined. There is no Haarnoja/SAC loss, no Adam state, or
canonical-bank entropy target. The observation is reduced across learner
ranks, detached, and applied first to the next rollout group. A nonpositive
warmup mean, nonfinite observation, incompatible checkpoint unit, or missing
adaptive controller state also fails closed.

The canonical-bank estimator itself remains E44-OGS:

- validator-positive prompt-local growing support;
- leave-one-out canonical surprisal advantage;
- novelty coefficient `beta=0.50`;
- pseudocount `1`;
- surprisal clip `5`;
- exact group snapshot followed by atomic bank update;
- canonical advantage added once after ordinary Dr.GRPO task centering;
- bank and policy-entropy controller checkpointed independently.

The cumulative bank entropy remains telemetry only and never controls alpha.

## Frozen matched cohort

- Model:
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Domains and prompt pools:
  - graph coloring: 192 train, 96 neutral evaluation;
  - Countdown easy3: 384 train, 128 neutral evaluation;
  - Python factors: 384 train, 128 neutral evaluation.
- Arms:
  - `grpo`: ordinary matched Dr.GRPO with passive verified-discovery tracking;
  - `online_canonical_policy_entropy`: canonical exploration with the
    controller above.
- Seeds: `43,44,45`.
- Group size: `G=16`.
- Budget: exactly 50 complete prompt-pool passes.
- Learning rate `2e-7`, one PPO epoch, `beta=0`, maximum norm `1`.
- Rollout temperature `1`, top-p `1`, maximum response length `192`.
- Prompt template `qwen_boxed`, executable verifier, and `multi_answer`
evaluation split.
- Evaluation at initialization and every quarter pass: greedy pass@1 plus four
  deterministic temperature-1 `K=8` draws with seeds `440100--440103`.
- Rolling optimizer-resume checkpoint every complete pass; retain newest two.
- Graph placement: node302 A100, 8 CPUs, 64 GiB, seven days.
- Countdown and Python placement: non-MLTheory `allcs/lowprio` on one RTX 3090
  from node020/node022/node023/node024/node026, 8 CPUs, 64 GiB, seven days,
  following the frozen E51 placement amendment.

Run prefixes:

- `gce51_policy_entropy_adaptive_canonical_05b_50ep_v2`;
- `cde51_policy_entropy_adaptive_canonical_05b_50ep_v2_allcs`;
- `pye51_policy_entropy_adaptive_canonical_05b_50ep_v2_allcs`.

## Admission and interpretation

The treatment revalidates the exact sampled output in the learner and aborts
on actor/learner reward disagreement. Graph keys are the constraint-checked
coloring vector, Countdown keys are the executed normalized AST, and Python
keys are the return vector produced by the isolated worker.

The Dr.GRPO arm runs the same validators and bank passively with alpha and
novelty exactly zero. Discovery telemetry cannot influence its task reward,
advantage, loss, or gradient.

Primary reporting is the complete paired-seed trajectory through pass 50 for
pass@8, mean@8, valid coverage@8, distinct-correct@8, cumulative verified
discoveries, mean verified support per prompt, model token entropy, normalized
policy-entropy score, and alpha. E51 is exploratory because it was selected
after inspecting E50. Removing the accumulating dual state addresses E50's
runaway mechanism but does not by itself establish improved exploration or
quality.
