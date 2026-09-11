# E24 prospective canonical MaxEnt graph-coloring 7B scaling

**Status: PROSPECTIVE EXPLORATORY SCALING COHORT, FROZEN BEFORE E24
OUTCOMES (2026-07-21).** This cohort extends the finite-action graph-coloring
Standard MaxEnt treatments used by E16 at 0.5B and E17 at 3B to
Qwen2.5-7B-Instruct. It does not revive E11's failed free-form sequence-MaxEnt
grid; E11's stamps, targets, group size, and response policy are invalid here.

## Scientific identity

- Environment: canonical three-node graph coloring.
- Model: the locally pinned `Qwen/Qwen2.5-7B-Instruct` snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Seeds: 43, 44, and 45.
- Methods: fixed Standard MaxEnt, proportional entropy control, and
  Haarnoja-style dual entropy control.
- Data: the frozen 192-row training and 96-row evaluation pools at
  `var/data/exact_answer_mode_probe`.
- Policy support: the 27 canonical three-color actions sampled by the
  learner-side fixed-shape restricted inverse-CDF path.
- Budget: five passes through the 192-row pool, with evaluation, parameter
  synchronization, and a rolling optimizer checkpoint every 48 prompts.
- Group size: 16 samples per prompt; evaluation reports greedy pass@1 and
  sampled pass@8, mean@8, distinct modes, and coverage at temperature one.

The original implementation was the immutable E16/E17 canonical source tree.
The four-GPU operational amendment below uses the narrowly derived immutable
tree with hash
`a02e2d242f797d65d788cfbcf8e278e675031d2f10de31edfca5cf88e07b4b43`.
The launcher reuses the frozen E17 operational scripts and requires
byte-identical digit-tokenizer files across the E16 0.5B and E24 7B models.

## Treatments and execution

Fixed MaxEnt uses `alpha=0.10`. Proportional control uses base `0.075`, maximum
`0.10`, EMA `0.9`, gain `4`, and immediate exact-entropy control. Haarnoja dual
uses base `0.075`, bounds `[0.05, 0.10]`, log-alpha Adam learning rate `0.005`,
and immediate control. Both adaptive methods target exact canonical-action
entropy `2.7974740052946436` nats. All methods retain E17's learning rate
`2e-7`, one PPO epoch, gradient clipping at `1.0`, no KL penalty, no token
entropy bonus, and no expected-length controller.

Each replacement run requests four node302 80-GiB A100s and 192 GiB host
memory in the `mltheory` partition/account. The amended 7B topology uses four
collocated learner ranks and one tensor-parallel actor, rollout batch four,
global batch 16, per-device microbatch four, ZeRO-2, optimizer and activation
offload, and vLLM sleep at ratio `0.25`. Expandable CUDA allocator segments
remain unset because they conflict with the sleep pool.

The launcher submits all nine jobs held, verifies exactly one job per
method/seed cell, and audits each held allocation's identity, resources,
canonical graph settings, budget, and checkpoint cadence before releasing the
cohort. A partial or mismatched cohort remains held. Release makes the jobs
scheduler-eligible; it does not require or force an immediate start.

## Pre-training operational amendment A1 (2026-07-21)

The nine `gce24_canonical_maxent_7b_v1` jobs repeatedly reached ZeRO
CPU-optimizer initialization and then exceeded their 96-GiB Slurm memory
cgroups before any optimizer update or eligible scientific outcome. They were
retired after exhausting or entering ineffective requeue loops.

At the user's direction, the replacement namespace is
`gce24_canonical_maxent_7b_v2_4xa100`, with four A100s and 192 GiB host memory
per individual run. The derived source replicates the same current canonical
prompt and complete 16-candidate action group on all four learner ranks for
group-relative advantage construction, then deterministically partitions the
16 candidates into four disjoint microbatches of four. DeepSpeed reduces the
shards into one global-batch-16 optimizer update. Group size, examples per
update, prompt order, update count, evaluation/checkpoint cadence, treatments,
and budgets are unchanged. The four-way collocated vLLM actor is
tensor-parallel across the same GPUs and sleeps during learner work; ZeRO-2
shards optimizer state across the learner ranks.

## Pre-training operational amendment A2 (2026-07-21)

The v2 jobs reached the first learner update but iterated the backward
microbatch loop over all 16 replicated candidates after each rank had already
received its disjoint four-candidate shard. Each rank therefore produced one
valid local microbatch followed by empty microbatches, and Qwen failed on a
zero-sized transformer forward before any optimizer update. Subsequent NCCL
watchdog failures were consequences of that rank-local exception. No v2 job
produced an eligible post-update outcome.

The replacement namespace is `gce24_canonical_maxent_7b_v3_4xa100_fix` and
uses immutable source hash
`044f6df047788dc8b67bbe224281a403c6d5eab04de89881f5a17cfe5c147cf9`.
The only source change makes the backward microbatch loop terminate at the
rank-local shard length. Every rank now performs exactly one four-candidate
microbatch, and the four disjoint shards still reconstruct the frozen global
batch of 16 through DeepSpeed. Model, data, methods, seeds, coefficients,
group size, prompt order, update count, cadence, budget, and four-A100
single-node topology are unchanged.

## Pre-training operational amendment A3 (2026-07-21)

The v3 seed-43 canary completed 12 optimizer updates, proving the rank-local
backward fix, then stopped at the first scheduled evaluation. The launcher
had expressed the frozen 48-prompt synchronization and checkpoint cadence as
48 learner updates even though the four-GPU rollout consumes four prompts per
update. Evaluation was correctly resolved to 12 updates, so OAT rejected it
while actor weights lagged the learner. No evaluation completed.

The replacement namespace is
`gce24_canonical_maxent_7b_v4_4xa100_evalsync_fix`. It expresses the same
frozen 48-prompt synchronization and checkpoint cadence as 12 learner updates
(`48 / rollout_batch_4`). Evaluation remains every 48 prompts. No model,
data, treatment, seed, prompt order, update count, budget, or topology changes.
