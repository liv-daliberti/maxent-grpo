# E66/E68 MathIR deterministic-seed recovery amendment

**Date:** 2026-07-28  
**Scope:** paired execution repair from the last common checkpoint

## Observed failure

All six paired E66/E68 MathIR jobs reached optimizer step 4294 and then failed
at step 4295 before that update with:

```text
ValueError: Seed must be between 0 and 2**32 - 1
```

The replicated-group permutation schedule used
`seed + 1_000_003 * learner_step + ppo_epoch` directly as a legacy NumPy
`RandomState` seed. The schedule first leaves NumPy's unsigned 32-bit seed
domain at this boundary. Every run retains a complete step-4224 model,
optimizer, RNG, sampler, and online-bank checkpoint.

## Frozen repair

For both arms, construct a new source snapshot from that arm's original frozen
snapshot and change exactly:

- `src/oat_drgrpo/replicated_group.py`, adding a named deterministic seed
  helper that returns the historical expression modulo `2**32`; and
- `src/oat_drgrpo/learner/grpo.py`, using that helper for the replicated-group
  permutation.

This is identical to the historical schedule for every representable prior
seed, including all updates through step 4294. It changes no model, data,
objective, sampling distribution, proposal logic, controller, optimizer
hyperparameter, evaluation, or compute budget.

## Paired resumption

Submit one fresh held job for each original E66/E68 MathIR seed. Reuse the
original Slurm `SubmitLine` and preserve the original run stamp so analysis can
stitch the new attempt after the old step-4224 prefix. The only submission
mutations are:

1. point `OAT_ZERO_SOURCE_ROOT` at the corresponding two-file-patched frozen
   source;
2. use a fresh node-local scratch path;
3. set `OAT_ZERO_INITIAL_RESUME_DIR` to the original checkpoint directory and
   `OAT_ZERO_INITIAL_RESUME_TAG=step_04224`; and
4. attach the machine-readable recovery-record path.

The original protocol identity remains in force. All six jobs must be held
and audited together before release. The recovery record binds the amendment,
launcher, original and patched source trees, original and recovery job IDs,
archived submission lines, failure-log tails, and complete checkpoint file
hashes.

The original seed-overflow traceback is a registered execution interruption
only for these six original attempts. Any uncaught traceback in a recovery
attempt remains fatal. No observed E66/E68 outcome is used to alter either arm.

