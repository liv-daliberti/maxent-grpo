# PantryPlan task-bound MaxEnt integration qualification v2

**Status: FROZEN AFTER V1-R1 JOBS 30200958/30200959 AND AUDIT 30200961, BEFORE V2 JOBS — 2026-07-30**

## Why this is a new qualification

The v1-r1 pair completed 32 updates with positive task reward at every update,
but its audit stopped Stage B. Every actor-positive six-bit response was passed
raw to the generic ModeBench validator, producing 6--11 actor/verifier
disagreements per update, zero tracked canonical outcomes, zero semantic or
novelty effect, and no replay groups. The task-specific actor had already
decoded the same mask into a fully verified Pantry witness. Thus v1-r1 is an
immutable mechanism-path failure, not a negative comparison of applied
MaxEnt versus Dr.GRPO. This v2 integration qualification does not relabel it.

## Frozen bridge and pair

The bridge uses the task-decoded, verifier-ready Pantry witness only to obtain
the canonical outcome key. Policy scoring, PPO loss, replay likelihood, token
counts, and gradients remain attached to the original six mask token IDs. The
bridge is active only for task-bound canonical ModeBench actions and fails if
there is not exactly one decoded witness per sampled action sequence.

V2 retains the same Qwen2.5-0.5B model, Pantry v2 data, seed 76201, initialization,
16 rollouts, optimizer, six-bit action horizon, `grpo_compute_matched` control,
E70 treatment coefficients, capacity 16, warmups 64, one global replay group
per update, and all information boundaries. It runs 96 updates over three
prompt passes so the unchanged warmup 64 is crossed and applied replay is
actually tested. The maximum query budget is 1,488; checkpoints and the final
audit target update 96. This remains development-only and is not a paper seed.

## Pass boundary

Both arms must complete exactly 96 matched updates with zero actor/verifier
disagreements, positive verified reward, and an online two-mode prompt. The
treatment must apply a nonzero semantic or novelty advantage and a nonzero
replay gradient after eligibility. The control must traverse identical replay
compute, expose nonzero raw replay telemetry when eligible, and apply exactly
zero replay gradient. All prior finite, six-bit support, no-gold-feedback,
no-projection, source, data, model, protocol, identity, manifest, scheduler,
metric, and log checks remain.

A pass authorizes only the ten fresh Pantry Stage-B jobs at seeds 43--47. A
failure stops Pantry from the 80-run grid; no prompt, seed, coefficient,
threshold, horizon, data row, or outcome is substituted after observation.
