# E23 prospective canonical MaxEnt Countdown 7B scaling

**Status: PROSPECTIVE EXPLORATORY SCALING COHORT, FROZEN BEFORE E23
OUTCOMES (2026-07-20).** This cohort extends the finite-action Countdown
Standard MaxEnt treatment used by E16 at 0.5B and E17 at 3B to
Qwen2.5-7B-Instruct. It does not revive E11's failed free-form
sequence-MaxEnt grid. E11 stamps, targets, group size, and free-form response
policy are not valid E23 inputs.

## Scientific identity

- Environment: canonical Countdown easy3.
- Model: `Qwen/Qwen2.5-7B-Instruct` from the locally pinned snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Seeds: 43, 44, and 45.
- Methods: fixed Standard MaxEnt, proportional entropy control, and
  Haarnoja-style dual entropy control.
- Data: the frozen 384-row training and 128-row evaluation pools at
  `var/data/exact_countdown_easy3_probe`.
- Policy support: the audited 108 canonical three-digit action codes, sampled
  by the learner-side fixed-shape restricted inverse-CDF path.
- Budget: five passes through the 384-row pool, with evaluation, parameter
  synchronization, and rolling optimizer checkpoints every 96 prompts.
- Group size: 16 samples per prompt. The evaluation reports greedy pass@1 and
  sampled pass@8, mean@8, distinct modes, and mode coverage at temperature 1.

The original source implementation was the immutable E16/E17 canonical tree.
The four-GPU operational amendment below uses the narrowly derived immutable
tree with hash
`a02e2d242f797d65d788cfbcf8e278e675031d2f10de31edfca5cf88e07b4b43`.
E23 reuses the frozen E17 operational scripts and requires byte-identical
digit-tokenizer files across the E16 0.5B and E23 7B model snapshots.

## Treatment settings

| Method | Coefficient rule |
|---|---|
| Standard MaxEnt fixed | `alpha=0.10` |
| Standard MaxEnt proportional | base `0.075`, max `0.10`, EMA `0.9`, gain `4`, immediate exact-entropy control |
| Standard MaxEnt Haarnoja dual | base `0.075`, min `0.05`, max `0.10`, log-alpha Adam LR `0.005`, immediate exact-entropy control |

Both adaptive arms target exact canonical-action entropy
`3.9741470618167156` nats. All methods use learning rate `2e-7`, one PPO
epoch, PPO clip `0.2`, gradient clipping at `1.0`, no KL penalty, no token
entropy bonus, and no expected-length controller. Since canonical Countdown
has an invariant three-token action horizon, E11's free-form length/EOS
failure mode is structurally absent.

## Execution layout and release rule

Each replacement run requests four node302 80-GiB A100 GPUs and 192 GiB host
memory in the `mltheory` partition/account. ZeRO-2, optimizer and activation
offload, vLLM sleep at ratio `0.25`, global training batch 16, and per-device
backward microbatch four are fixed. Expandable CUDA allocator segments are
unset because they conflict with the vLLM sleep pool on the validated 7B
runtime.

The launcher first submits all nine jobs in Slurm's user-held state. Before
release it must verify the manifest contains exactly one job for every
method/seed cell and audit each job's pending/held state, run stamp, model,
seed, variant, canonical task, target budget, resource request, and checkpoint
cadence. An incomplete or mismatched cohort remains held. This operational
audit is not an outcome gate and does not inspect training results.

E23 is exploratory model-scaling evidence. Its results must remain distinct
from the failed E11 free-form grid and from graph-coloring 7B, which E23 does
not launch.

## Pre-training operational amendment A1 (2026-07-20)

The initial `cde23_canonical_maxent_7b_v1` jobs 30031135--30031143 were
submitted and released after the cohort audit. Fixed seed 43 reached argument
validation but no model initialization, evaluation, optimizer update, or
scientific outcome. Validation rejected rollout batch size two because the
frozen canonical learner-side sampler requires rollout batch size one. The
remaining jobs had not started. All nine v1 jobs were held and retired.

The replacement namespace is `cde23_canonical_maxent_7b_v2`. It changes only
`OAT_ZERO_ROLLOUT_BATCH_SIZE` from two to one, matching E16/E17's canonical
sampler contract; the two-GPU actor allocation remains unchanged. Model,
source, data, methods, seeds, group size, optimization, evaluation cadence,
checkpoint cadence, and resource class remain frozen above.

## Pre-training operational amendment A2 (2026-07-20)

The `cde23_canonical_maxent_7b_v2` fixed seed-43 startup passed argument
validation and loaded the learner model, then stopped before actor model
initialization, evaluation, or any optimizer update. OAT requires rollout
batch size to be divisible by the number of GPUs per actor; the canonical
sampler simultaneously requires rollout batch size one. The frozen runtime
therefore cannot execute canonical learner-side sampling with a two-GPU
tensor-parallel actor. All nine v2 jobs were held and retired.

The replacement namespace is `cde23_canonical_maxent_7b_v3`. It uses one
node302 80-GiB A100 and 96 GiB host memory per job, with one learner GPU and
one collocated actor GPU identity. Rollout batch one, vLLM sleep, ZeRO-2,
optimizer offload, activation offload, global batch 16, and microbatch four
remain fixed. This is the E17 one-actor execution topology with the larger
GPU memory class required by the 7B model; no scientific treatment parameter
changes.

## Pre-training operational amendment A3 (2026-07-21)

The nine v3 jobs repeatedly reached ZeRO CPU-optimizer initialization and then
exceeded their 96-GiB Slurm memory cgroups before any optimizer update or
eligible scientific outcome. They were retired after exhausting or entering
their ineffective requeue loops.

At the user's direction, the replacement namespace is
`cde23_canonical_maxent_7b_v4_4xa100`, with four A100s and 192 GiB host memory
per individual run. The derived source replicates the same current canonical
prompt and complete 16-candidate action group on all four learner ranks for
group-relative advantage construction, then deterministically partitions the
16 candidates into four disjoint microbatches of four. DeepSpeed reduces those
four shards into one global-batch-16 optimizer update. Thus group size,
examples per update, prompt order, number of updates, evaluation/checkpoint
cadence, treatments, and budgets remain unchanged. The four-way collocated
vLLM actor is tensor-parallel across the same GPUs and sleeps during learner
work; ZeRO-2 shards optimizer state across the four learner ranks.

## Pre-training operational amendment A4 (2026-07-21)

The v4 jobs reached the first learner update but iterated the backward
microbatch loop over all 16 replicated candidates after each rank had already
received its disjoint four-candidate shard. Each rank therefore produced one
valid local microbatch followed by empty microbatches, and Qwen failed on a
zero-sized transformer forward before any optimizer update. Subsequent NCCL
watchdog failures were consequences of that rank-local exception. No v4 job
produced an eligible post-update outcome.

The replacement namespace is `cde23_canonical_maxent_7b_v5_4xa100_fix` and
uses immutable source hash
`044f6df047788dc8b67bbe224281a403c6d5eab04de89881f5a17cfe5c147cf9`.
The only source change makes the backward microbatch loop terminate at the
rank-local shard length. Every rank now performs exactly one four-candidate
microbatch, and the four disjoint shards still reconstruct the frozen global
batch of 16 through DeepSpeed. Model, data, methods, seeds, coefficients,
group size, prompt order, update count, cadence, budget, and four-A100
single-node topology are unchanged.

## Pre-training operational amendment A5 (2026-07-21)

The v5 seed-43 canary completed 24 optimizer updates, proving the rank-local
backward fix, then stopped at the first scheduled evaluation. The launcher
had expressed the frozen 96-prompt synchronization and checkpoint cadence as
96 learner updates even though the four-GPU rollout consumes four prompts per
update. Evaluation was correctly resolved to 24 updates, so OAT rejected it
while actor weights lagged the learner. No evaluation completed.

The replacement namespace is
`cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix`. It expresses the same
frozen 96-prompt synchronization and checkpoint cadence as 24 learner updates
(`96 / rollout_batch_4`). Evaluation remains every 96 prompts. No model,
data, treatment, seed, prompt order, update count, budget, or topology changes.
