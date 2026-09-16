# E113-R3-S3: terminal DAPO-failure process cleanup

**Frozen:** 2026-08-19 16:45 EDT, after all 50 R3 science cells were released
and after Qwen Countdown seed 44 reached the registered ten-batch DAPO
exhaustion at learner step 1. No endpoint efficacy result was inspected.

## Observed process-lifetime defect

Job 30790926 generated ten all-zero groups, raised the registered terminal
`DAPO dynamic sampling could not produce a non-constant reward group within 10
generation batches` error at learner step 1, wrote no optimizer update and no
completion receipt, and stopped logging. Its Launchpad actor processes remained
alive after the learner died, leaving Slurm `RUNNING`. Wrapper-level requeue was
already disabled, as required, but the frozen `train.sh` watchdog itself was
not enabled.

## Cleanup-only amendment

For exact E113-R3 job IDs 30790925 through 30790974, the mutable Slurm entry
wrapper may enable its existing process-group watchdog with:

- poll interval 15 seconds;
- stale-metrics threshold 300 seconds with 900 seconds startup grace;
- the existing fatal child-process signatures plus the exact DAPO ten-batch
  exhaustion signature; and
- the actual `%x-%j.out` log path.

The wrapper must verify the immutable E113 snapshot, `dapo` variant, E113-R3
run-stamp prefix, disabled wrapper requeue, and zero watchdog restarts before
activating this amendment. On the fatal signature it terminates only that job's
process group and exits nonzero. It must not requeue, resume, resample, raise the
ten-batch cap, or alter any learner input. Jobs already allocated before this
amendment retain their copied wrapper environment and may require the same
evidence-gated administrative cancellation after learner death.

This amendment changes resource cleanup and scheduler observability only. The
scientific outcome remains the original one-shot DAPO feasibility failure.
