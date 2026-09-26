# E62: validator-filtered open-set proposals

**Status:** IMPLEMENTED; HELD PILOT MUST PASS BEFORE CROSS-DOMAIN EXPANSION
**Date frozen:** 2026-07-27
**Model:** Qwen2.5-0.5B-Instruct, exact local snapshot
**Initial domain:** executable Python factors
**Pilot arms:** E60 finite-bootstrap replay and E62 counterfactual proposals
**Training seed:** 9011 in both arms
**Pilot horizon:** one complete 384-prompt pass

## Pre-optimization R1 amendment

The first held release (`30126420`, `30126421`) exposed a configuration error
before the treatment created a run directory or completed optimizer step zero:
the proposal sampler requires the replicated free-form path, but the submitted
treatment retained its default `replicated_freeform_sampling=false`.
Validation stopped the treatment; its watchdog requeue was canceled, and the
still-starting control was canceled so it could not become an unmatched
comparison. No result from that attempt is used.

E62-R1 adds `replicated_freeform_sampling=true` and
`local_actor_weight_sync=true` to both arms, with vLLM sleep level one. The
scientific intervention, seed, horizon, optimizer, data, evaluation schedule,
information firewall, and gates below are unchanged. R1 uses fresh job IDs,
run stamps, identity, source snapshot path, and no checkpoint or run state from
the failed attempt. The live auditor also treats a newly created run directory
without a metrics file as startup-in-progress rather than a terminal failure;
actual traceback and runtime crash signatures remain fail-closed.

## Pre-optimization R2 amendment

R1 (`30126426`, `30126427`) reached the initial evaluation but stopped before
optimizer step zero. Both matched arms used the replicated one-prompt sampling
path with a 16-row candidate group on one learner rank, while the launcher
mistakenly set `train_batch_size_per_device=1`. The runtime contract requires
`train_batch_size_per_device * learner_world_size = num_samples`, hence
`1 * 1 != 16`. The watchdog-requeued jobs were canceled, and no R1 training
result or checkpoint is used.

E62-R2 changes only `train_batch_size_per_device` from 1 to 16 in both arms,
which restores the intended single optimizer minibatch containing the complete
16-response neutral group. It retains `rollout_batch_size=1`: that quantity is
the number of current prompts, not response rows. The held-job audit now binds
and verifies `num_samples=16`, `train_batch_size=16`,
`train_batch_size_per_device=16`, and `rollout_batch_size=1` before release.
R2 uses fresh job IDs, run stamps, identity, source and execution snapshots,
and no state from either failed attempt.

## Pre-optimization R3 amendment

Before R2 completed its initial evaluation or optimizer step zero, a
fail-closed mechanism audit found that its proposal lookup required a banked
exemplar from an earlier visit to the same prompt. The frozen pilot makes one
pass over 384 unique prompts, so that lookup could not actuate at all during
the pilot or during the first pass of a larger run. Jobs `30126435` and
`30126436` and monitor `30126437` were canceled at `latest_step=-1`; no R2
training result or state is used.

E62-R3 bootstraps the anchor from the ordinary 16-row neutral group for the
current prompt. It independently intersects positive task reward with the
executable validator, deduplicates by canonical outcome, and selects a
deterministic verified exemplar. If the current neutral group contains no such
exemplar, it falls back to a previously banked exemplar for the same prompt.
The proposal query still uses the same pre-update policy. Every outcome
verified in either the current neutral group or prior bank is treated as
already known and cannot be admitted as a novel proposal.

This amendment changes when a model-generated exemplar becomes eligible, not
what information training may read or what enters PPO. No gold answer,
support size, target entropy, target mode count, or evaluation metric is read.
The complete conditioned group remains support-only and is discarded from
PPO. R3 uses fresh job IDs, run stamps, identity, source and execution
snapshots, and no prior attempt state.

## R4 memory and evaluation-audit amendment

R3 (`30126443`, `30126444`) validated the current-neutral proposal telemetry
and completed update one, but both matched arms OOMed on the second policy
backward pass. The logical 16-response batch had been configured as one
physical 16-row microbatch on a 24 GB RTX 3090. Both jobs were canceled; no R3
checkpoint or result is used. The same live inspection also showed that each
evaluation checkpoint contains one separately labeled greedy trace plus four
sampled K=8 draws. The R3 auditor grouped all five rows, so it could not
recognize a complete four-draw evaluation.

E62-R4 retains one exact logical batch of 16 responses but uses physical
microbatches of four and four-step DeepSpeed gradient accumulation on the
single learner rank. The replicated-group validator now proves all of the
following before sampling and again before optimization:

- `train_batch_size = num_samples = 16`;
- the logical group divides exactly across learner ranks;
- each rank shard divides into complete physical microbatches;
- the live DeepSpeed accumulation width equals the number of microbatches in
  that shard.

Thus gradient accumulation changes peak memory only; it does not split group
statistics or change the logical optimizer batch. Canonical replay is still
evaluated once at the accumulation boundary using its existing exact
DeepSpeed scaling correction.

The R4 auditor now selects only
`evaluation_kind=fixed_seed_sampled_k_neutral` with unique draw indices
0--3; it explicitly excludes the greedy diagnostic. A fixture containing one
greedy row plus four sampled rows is part of the focused test gate. R4 uses
fresh job IDs, run stamps, identity, source and execution snapshots, and no
prior attempt state.

## R5 anti-copy retry amendment

R4 (`30126452`, `30126453`) ran stably through update 81 with no crash,
non-finite metric, conditioned PPO row, or audit violation. It exercised six
current-neutral anchors and generated 96 conditioned proposal rows. Of those,
56 were independently validator-positive and task-positive; all 56 had
exactly the anchor's executable outcome. There were zero known alternates,
zero novel outcomes, and zero admissions. The prompt-only one-shot actuator
therefore reproduced collapse rather than escaping it, so both jobs and
monitor `30126454` were canceled without using their result as evidence of
efficacy.

E62-R5 strengthens search without revealing any answer or semantic target:

- the prompt states that the collided executable outcome is forbidden;
- a fixed directive derived only from the public executable task contract asks
  for a concrete semantic mutation: a different returned divisor vector,
  vertex assignment, arithmetic expression tree, or algebraic state trace;
- proposal sampling uses fixed temperature 1.5;
- if a complete group contains no novel verified outcome, retry with a
  distinct fixed attempt instruction, up to three groups;
- stop immediately when any novel verified outcome appears.

The three-group maximum and temperature are frozen search-compute parameters,
not a desired mode count, support size, or target entropy. Every retry is
filtered by the same task-reward/validator intersection; same and already
known outcomes are discarded. Conditioned rows remain support-only and never
enter PPO. The actor reports the per-request temperature, and the auditor
checks one to three groups of exactly 16 rows for every anchored record.
R5 uses fresh job IDs, run stamps, identity, source and execution snapshots,
and no prior attempt state.

## R6 original-prompt temperature-sweep amendment

R5 (`30126486`, `30126487`) ran stably through update 106 with no crash,
non-finite metric, PPO contamination, or audit violation. The treatment
exercised five current-neutral anchors and generated 240 conditioned proposal
rows. Seven rows were independently validator-positive and task-positive, but
five reproduced the selected anchor and two reproduced outcomes already
present in the same neutral group. The remaining 233 rows were invalid. R5
therefore admitted zero novel outcomes: stronger conditioning traded validity
for textual diversity without crossing the singleton barrier. Both arms and
monitor `30126500` were canceled after this mechanism result; no R5 checkpoint
or efficacy result is used.

E62-R6 removes the answer-conditioned prompt entirely. A verified model
outcome is used only to trigger search and define the set of outcomes that
cannot be admitted. Each proposal group receives the untouched original task
prompt. Up to three independent groups use the fixed temperature schedule
`[1.0, 1.2, 1.4]`, stopping immediately after the first novel verified
outcome. This preserves the task's original output grammar while searching
progressively farther into the model's own distribution.

The schedule and three-group maximum are fixed search-compute parameters, not
an entropy target, desired support, alpha projection, or ground-truth-derived
"how high" signal. R6 reports the base, increment, last used temperature,
original-prompt group count, and conditioned-prompt group count. The auditor
requires every generated group to use the original prompt and requires the
conditioned-prompt count to remain zero. Proposal rows remain support-only and
never enter PPO. R6 uses fresh job IDs, run stamps, identity, source and
execution snapshots, and no state from prior attempts.

Neutral and proposal requests use explicit deterministic request seeds derived
from disjoint named streams. Extra proposal calls therefore cannot advance or
perturb later neutral-rollout randomness. The auditor compares every common
neutral step across the matched arms and fails on any request-seed mismatch.

## R7 telemetry-precision audit amendment

R6 (`30126549`, `30126550`) reached update 64 without a crash, non-finite
metric, RNG mismatch, conditioned prompt, or PPO proposal row. Its first
anchor generated three untouched-original-prompt groups and no novel
admission. The immutable auditor nevertheless failed because float32 telemetry
serialized the fixed temperature increment and final temperature as
`0.20000000298023224` and `1.399999976158142`, while it used Python's default
near-float64 `math.isclose` tolerance against 0.2 and 1.4. Jobs and monitor
`30126551` were canceled immediately after this bound audit failure; no R6
checkpoint or efficacy result is used.

E62-R7 changes only the auditor's comparison for these two float32 telemetry
fields to `rel_tol=0` and `abs_tol=1e-6`. The source mechanism, model, data,
seed, prompt sequence, optimizer, neutral/proposal request-seed derivation,
temperature schedule, information firewall, and gates are unchanged. R7 uses
fresh job IDs, run stamps, identity, source and execution snapshots, and no
state from R6.

## R8 external-cancellation recovery amendment

R7 (`30126568`, `30126569`) ran cleanly through update 120. At update 113,
proposal attempt two at temperature 1.2 generated one novel
validator-positive and task-positive Python return vector alongside six
copies of the known outcome. The novel mode was admitted support-only, no
proposal row entered PPO, verified mean support rose above one, and the live
audit retained zero violations. This establishes the first observed singleton
escape but is not a terminal efficacy result.

At 10:36:05 EDT, Slurm accounting records both training jobs and monitor
`30126570` as `CANCELLED by 363432`, exit code zero. Both matched jobs received
SIGTERM together; neither produced a traceback, OOM, non-finite metric, or
internal crash. The checkpoint cadence was terminal-only, so no resumable
checkpoint existed. R7 is therefore retained only as a nonterminal mechanism
observation.

E62-R8 changes only save/resume cadence from update 384 to every 96 updates,
retaining the latest two checkpoints. Prompt traversal, optimizer, controllers,
bank state, cumulative proposal counters, and deterministic request-stream
position are already checkpointed and fail closed on incompatible resume.
The scientific mechanism, seed, horizon, evaluation cadence, and all
information-firewall and efficacy gates are unchanged. R8 uses fresh job IDs,
run stamps, identity, source and execution snapshots, and no R7 state.

## Question

E57--E60 can preserve a verified mode after the policy discovers it, but the
Python experiments generally retain only one executable outcome per solved
prompt. The inverse controller detects low entropy; its replay actuator cannot
create a gradient toward an outcome absent from the verified bank.

E62 asks whether a model-self counterfactual proposal can cross that singleton
barrier without revealing an exhaustive answer catalogue, a desired number of
modes, a desired entropy, or evaluation feedback.

## Frozen R8 intervention

The ordinary neutral-prompt rollout group is generated exactly as in E60. If
the current group contains at least one task-positive and independently
validator-positive exemplar generated by this policy—or, as a fallback, the
same prompt has such an exemplar in the verified bank:

1. deduplicate current verified exemplars by canonical outcome and
   deterministically select one outcome key, falling back to a stored verified
   outcome for the same prompt;
2. leave the original formatted task prompt byte-for-byte unchanged;
3. generate up to three additional groups with width `num_samples=16`, using
   fixed temperatures 1.0, 1.2, and 1.4 in order and stopping on the first
   group containing a novel verified outcome;
4. independently intersect positive actor reward with the executable
   canonicalizer result;
5. discard the anchor outcome and every outcome already present in either the
   current neutral group or prior verified bank;
6. deduplicate new outcomes, retaining the lexicographically smallest token
   response for each;
7. admit those exemplars to the verified replay bank with support-only count
   one.

No proposal trajectory, reward, loss mask, behavior log probability, or
advantage is transported into PPO. PPO receives exactly the original 16
neutral rows. Proposal support is permitted only when canonical bank alpha and
novelty beta are both zero, preventing off-prompt proposal counts from entering
an on-policy bank advantage.

No exemplar text is decoded or inserted into a proposal prompt. Admission is
atomic. Validator/task disagreement is fail-closed telemetry and never crashes
training.

## Information firewall

Training may read:

- the current raw and formatted prompt;
- canonical keys from validator-positive responses generated by the same
  policy for that prompt;
- the existing executable validator;
- the fixed rollout-width compute budget;
- the model-entropy sensors and warmup/EMA state already frozen in E60.

Training may not read:

- `num_completions` or any exhaustive support size;
- a gold list of valid modes;
- a target number of modes;
- a target entropy or target normalized entropy;
- coverage, distinct@K, pass@K, correctness@K, or any evaluation result;
- a coefficient projection or alpha cap.

The three-group retry schedule and temperatures are compute/search parameters,
not semantic targets. E60's inverse coefficients remain unprojected.

## Same-seed Python pilot

The pilot contains exactly two fresh jobs:

- `verified_first_bootstrap_local_canonical`, seed 9011;
- `verified_counterfactual_canonical`, seed 9011.

Both use the same model, Python-factor train/eval artifacts, prompt order,
optimizer, 16 neutral samples, four independent K=8 evaluation draws,
evaluation cadence 96 updates, and 384-update horizon. E62 pays for up to
three extra proposal groups only on prompts with a verified exemplar, stopping
early on discovery. This is the mechanism's intentional compute cost and is
reported separately.

## Fail-closed pilot audit

Every job is bound to a held Slurm job ID, immutable source snapshot,
execution-surface snapshot, launcher, auditor, protocol, and manifest.

The pilot is structurally valid only if:

- both jobs reach update 384 without non-finite metrics or crash signatures;
- E62 reports `conditioned_rows_sent_to_ppo = 0` everywhere;
- E62 reports exactly 16 neutral PPO rows everywhere;
- all gold-support, desired-mode-count, and evaluation-feedback telemetry is
  zero;
- proposal generation occurs whenever the current neutral group or prior bank
  supplies a model-generated verified anchor;
- at least one genuinely new validator/task-positive outcome is admitted;
- checkpoint serialization contains the proposal counters and exact bank.

The efficacy signal is evaluated only after training and never fed back:

- at least two post-initial E62 evaluations have
  `distinct_correct@8 > pass@8`; and
- E62's final landed distinct@8 is not below the same-seed E60 mechanism
  control.

This one-seed pilot is a mechanism gate, not evidence of stable success. A pass
authorizes a fresh three-seed comparison. A failure triggers diagnosis of the
proposal mechanism; it does not authorize selecting a gold-derived support or
entropy target.

## Cross-domain expansion gate

The requested endpoint remains a fresh three-seed validation across graph
coloring, Countdown, Python factors, and MathIR. Completion requires high,
stable neutral-prompt mean distinct correct on all four domains, no crashes,
and the same information firewall. E62 is not declared successful from this
pilot alone.

## R9 validator-preserving actuator amendment

R7 established that untouched-prompt resampling can very occasionally escape
singleton support, but only one new Python outcome was admitted after 416
additional proposal rows. That actuator is too sparse to reliably restore
entropy after collapse. R8 was configured as a checkpoint-cadence recovery
but was not launched. E62-R9 supersedes it with a mechanism change and
therefore uses fresh job IDs, run stamps, identity, source and execution
snapshots, with no R7 state.

Before any stochastic fallback request, R9 applies a deterministic,
validator-preserving transformation to one response that the current policy
generated and that passed both task reward and the executable validator. In
the Python-factor pilot the rule is:

`f(n) -> n // f(n)`

and case-local versions of the same cofactor rule. If `f(n)` is a proper
divisor of public input `n`, its cofactor is also a proper divisor. The
actuator uses only the model-authored executable rule, the public evaluation
cases already required by the ordinary task validator, and this algebraic
identity. It never reads another valid answer, a support size, a desired mode
count, a target entropy, an evaluation score, or a coefficient bound.

Every derived surface is passed through the ordinary executable validator,
tokenized, decoded again, and revalidated to the same canonical outcome.
Already-known outcomes and failed tokenization round trips are rejected. All
remaining novel outcomes are admitted atomically to support-only replay. No
derived surface carries a reward, loss mask, behavior log probability,
advantage, or trajectory, and derived rows sent to PPO is exactly zero. If no
novel verified transformation exists, R9 retains R7's isolated
untouched-original-prompt temperature sweep as a fallback.

The implementation also defines corresponding public-constraint
transformations for the later cross-domain gate:

- Countdown uses sign-preserving expression-tree rewrites such as
  `a + b -> a - (-b)` and paired operand negation for multiplication or
  division. The ordinary Countdown validator rechecks target value and number
  use.
- graph coloring uses safe local recoloring and fixed-vertex-free
  two-color-component swaps. The ordinary graph validator rechecks all edges
  and public partial colors.
- MathIR commutes adjacent additive and scaling commands while transforming
  the additive expression by the same scale. The ordinary MathIR interpreter
  rechecks the complete state path and final equation.

These transformations are fixed by task semantics rather than by measured
coverage. They can create a usable verified replay gradient when sampled
valid support is singleton, which is the missing actuator identified in
E57--E60. The Python pilot audit now requires at least one transformation
success, at least one independently verified novel transformed outcome, zero
transform rows in PPO, zero forbidden feedback, and all prior stability,
matched-neutral-RNG, and terminal neutral-efficacy gates.

## R10 prompt-local lookup amendment

R9 is preserved as a separate immutable attempt. At update 192 it proved that
verified transformed support can transfer into neutral-policy multiplicity:
mean distinct correct at K=8 was 0.4121 while pass@8 was 0.2363. It also
revealed an unsafe generalization. The most common failed evaluation response
was `lambda n: n // 2` on prompts containing odd inputs. That surface came
directly from R9's global cofactor rewrite and reduced correctness relative to
the matched control.

R10 changes only the Python transformation serialization and adds a mechanical
replay-capacity reservation. It uses the same model, data, seed, optimizer,
neutral request stream, one-pass horizon, evaluations, unprojected
controllers, support-only admission boundary, and stochastic fallback as R9.
R10 uses fresh job IDs, run stamps, identity, source and execution snapshots,
and no state from R9.

For a model-authored function `f` that has independently passed the executable
validator on the public cases `n_1 ... n_m`, the validator records only the
outputs actually produced by that function, `d_i = f(n_i)`. R10 derives
cofactors `c_i = n_i // d_i` and serializes alternatives exclusively as a
complete prompt-local conditional lookup:

`lambda n: v_1 if n == n_1 else ... else v_m`

Each `v_i` is either the model-produced `d_i` or its forced cofactor `c_i`.
R10 does not emit a free-standing algebraic expression such as `n // 2`.
Thus a prompt-specific cofactor cannot be learned verbatim as a purported
universal factor rule. Every lookup is independently executed, tokenized,
decoded, and executed again before admission.

The transform reserves one fixed replay-capacity slot for each current or
prior model-generated verified outcome and truncates transformed outcomes
deterministically to the remaining configured replay slots. Replay capacity is
a storage/compute limit fixed before outcomes are observed; it is not a
desired support, entropy target, evaluation signal, alpha bound, or
ground-truth-derived "how high" value. When capacity is exhausted, no fallback
proposal compute is spent for that prompt.

The R10 mechanism gate remains terminal and neutral-policy based. Bank size or
teacher-forced replay entropy alone cannot pass it. R10 must retain zero
forbidden feedback and zero transformed PPO rows, produce repeated
post-initial `distinct@8 > pass@8`, and finish with distinct@8 no lower than
its exact same-seed control.
