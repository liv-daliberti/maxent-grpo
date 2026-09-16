# E113-R1-M1: Qwen DAPO smoke memory-capacity recovery

**Frozen:** 2026-08-19 16:00 EDT, after the original E113-R1 Qwen learner
raised CUDA OOM and while the E113-R1 Falcon smoke was still running. No M1
job or full scientific successor had been submitted.

## Observed failure and boundary

E113-R1 Qwen job 30790111 exercised the registered Graph DAPO path for six
accepted optimizer updates. At update 7, an accepted 16-response batch padded
to 367 tokens reached backward on a 24 GB A5000 and failed while requesting an
additional 428 MiB. The terminal trace is `torch.OutOfMemoryError`, not
dynamic-sampling exhaustion, non-finite optimization, or verifier failure.
The six-update artifact and original R1 ledger remain immutable and never enter
an efficacy estimate.

The failure demonstrates insufficient device capacity for the frozen complete
16-response DAPO optimizer batch. The runtime adapter explicitly requires
`train_batch_size_per_device == num_samples == 16`, so reducing the microbatch
would change the registered DAPO adapter rather than repair its placement.
CPU Adam offload would also replace fused Adam with DeepSpeedCPUAdam. Neither
change is admitted here.

## Single replacement smoke

M1 replaces only the failed Qwen operational smoke:

- model/domain/seed: Qwen2.5-0.5B, Graph Coloring, seed 43;
- target: exactly 32 accepted updates;
- responses per prompt and optimizer batch: 16, unchanged;
- generation-batch cap: 10, unchanged;
- total query ceiling: 5,120, unchanged;
- objective, optimizer implementation and hyperparameters, model revision,
  prompts, verifier, evaluation, and immutable E113 snapshot: unchanged;
- device-capacity change only: an A6000 (48 GB) selected from
  `node[103-104,205-208,805]` in `lowprio`, with the same one-GPU process
  topology;
- fresh run directory, job name, and one-smoke/zero-science ledger.

Adam and activation offload remain explicitly disabled. The successful
trajectory therefore uses the same fused-Adam update and complete 16-response
batch as the failed job, with additional device memory only.

## Effective gate and reporting

The effective two-family operational gate is:

1. the original E113-R1 Falcon Graph smoke, if and only if its strict 32-update
   auditor passes; and
2. this E113-R1-M1 Qwen Graph replacement, if and only if the same strict
   auditor passes.

The failed original R1 Qwen smoke cannot count as a pass. M1 contains no
scientific job and cannot release a science cohort. The previously frozen
E113-R2 protocol is closed by its own rule because one original R1 smoke
failed; any full relaunch after this effective gate must use a fresh prospective
cohort name and protocol.

Submit M1 held, inspect the scheduler-expanded environment and A6000
placement, write one atomic ledger, and only then release it. There is no force
path, reuse of the failed output, or fallback to optimizer/offload changes.
