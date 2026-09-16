# E113-R4-R2: vLLM scheduler-config compatibility recovery

Date frozen: 2026-08-24, after both R4-R1 operational smokes failed and all
R4-R1 dependent science jobs were canceled at zero runtime, but before any R4
optimizer step or R4 scientific-cell allocation.

## Trigger and diagnosis

R4-R1 smoke jobs `30855240` (Qwen-0.5B) and `30855241` (Falcon-1B) passed the
repaired short-Ray-socket path and reached construction of the pinned vLLM
0.8.3 rollout engine. Both then failed before model rollout or training with:

`ValueError: max_num_batched_tokens (448) must be greater than or equal to max_num_seqs (1024).`

The upstream DAPO launcher sets `max_num_batched_tokens` to prompt length plus
response length. Its published 22,528-token context is larger than verl's
fixed `max_num_seqs=1024`, so the vLLM invariant holds implicitly. R4 retained
that launcher expression while substituting the frozen ModeBench context; the
Graph smoke context is only `256 + 192 = 448`, exposing the incompatibility.
The earlier Ray failure occurred first and masked this second launch blocker.

Neither smoke wrote `TRAINING_COMPLETE.json` or a scientific checkpoint. The
50 dependent jobs `30855242`--`30855291` were automatically canceled without
allocation, runtime, output directories, responses, or optimizer steps.

## Authorized compatibility-only change

Retain the pinned verl rollout concurrency limit `max_num_seqs=1024` and set:

`max_num_batched_tokens = max(prompt_length + response_length, max_num_seqs)`

The runner must pass both values explicitly. This is the smallest correction
that preserves verl's concurrency setting and satisfies vLLM's required
aggregate scheduler-cap invariant. It changes only rollout scheduling
capacity. Models, prompts, maximum per-sequence lengths, generation count,
samples, sampling distribution, seeds, reward, dynamic filtering, optimizer,
DAPO loss, stopping rule, source commit, image, and the 50-cell estimand remain
unchanged.

Before GPU allocation, recovery must validate `1024/1024` by constructing a
`vllm.config.SchedulerConfig` inside the exact pinned SIF. A new immutable
runtime snapshot must derive from the R4-R1 snapshot and differ only in the
runner and snapshot identity metadata.

## Two-stage recovery gate

To prevent an operational smoke failure from appearing as 50 failed science
cells, recovery is split into two transactions:

1. Submit, held-audit, record, and release only two fresh one-step Graph
   smokes, Qwen seed 43 and Falcon seed 55, on the approved non-preempting
   A6000 pool in partition `all`, account `allcs`, with 16 CPUs, 128 GiB,
   `Nice=0`, and a one-day limit.
2. Submit fresh copies of the exact 50 science cells only after both smokes
   are `COMPLETED` with exit `0:0`, each completion receipt identifies its
   submitted job, and the smoke output demonstrates one accepted upstream
   training step. The 50 replacements retain their seven-day limit and are
   dependency-gated on both successful smoke IDs.

If either smoke fails, no science replacement is submitted. A new blocker must
be diagnosed prospectively; the existing 50 zero-runtime cancellations remain
provenance only and are never counted as efficacy outcomes.

## Interpretation boundary

This amendment is blind to scientific endpoints because none exists. Smoke
success licenses the original frozen science matrix; it does not license an
efficacy claim. All prior smoke and canceled-job IDs remain immutable in the
authoritative ledger's recovery history.
