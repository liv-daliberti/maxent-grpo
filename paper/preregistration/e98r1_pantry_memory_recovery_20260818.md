# E98-R1 Pantry memory-safe execution recovery

Date: 2026-08-18.

## Trigger

The preregistered Pantry action-surface adapter was never tested to completion.
Its non-scientific job 30579529 executed one optimizer update, then failed while
waking the vLLM CuMem allocator on a contaminated A6000: other processes held
most of the device, the wake-up allocation raised CUDA out-of-memory, the actor
died, and the learner remained stale until the one-hour job limit. The audit
therefore never ran and all five dependent scientific replacements were later
cancelled without starting.

This is an operational failure of the disposable smoke, not a result from the
E98-R1 Pantry estimand. The repaired action representation remains frozen in
snapshot `e98r1_pantry_action_surface_a15c103a95a0bfee`.

## Frozen recovery

- Run a fresh 32-update seed-44 Pantry smoke on one A100-80GB on `node302`.
- Keep vLLM resident with `OAT_ZERO_VLLM_SLEEP=0`, reserve only 10% of device
  memory for its KV cache, and use expandable PyTorch allocator segments. This
  avoids the failed sleep/wake allocator path. These settings affect memory
  residency and capacity, not sampling, rewards, replay rows, or gradients.
- Reuse the existing E98-R1 audit unchanged. It must observe both registered
  dose pairs `(0,0)` and `(1,2)`, finite replay telemetry, and a terminal step
  of 32 before any scientific job becomes eligible.
- Submit five fresh scientific replacements behind the passed audit. Seeds
  43--44 retain the registered one-A5000 `cs` pool on
  `node202,node203,node204`; seeds 45--47 remain on one A100 on `node302`.
  The memory-residency settings above are applied to all five jobs.
- Preserve the run directories, seeds, frozen pools, action-surface snapshot,
  objective, optimizer, evaluation cadence, and 3,072-update horizon. No
  invalid prefix is resumed: none of the cancelled Pantry attempts produced a
  rolling checkpoint.

Submission is fail-closed. The smoke, audit, and scientific jobs are first
held and audited; ledgers are updated atomically; only then are all jobs
released. A failed smoke leaves the scientific jobs dependency-blocked and
non-executable.
