# PantryPlan paired verified-MaxEnt mechanism smoke v1

**Status: FROZEN AFTER THE PASSING R2-R1 CONTROL AUDIT AND BEFORE EITHER PAIRED JOB — 2026-07-30**

## Antecedent and scope

Pantry job `30199933` completed exactly 32 updates. Its audit-only repair
receipt reports every plumbing check true, positive verified reward on all 32
updates, and 428 sampled multimode evaluation prompts. This authorizes one
development-only paired mechanism smoke. The pair is not part of seeds 43--47
and cannot itself be reported as a paper result.

## Frozen pair

- arms: `grpo_compute_matched` and
  `verified_first_global_replay_canonical`;
- model, Pantry v2 data, source snapshot, optimizer, prompt order, seed
  `76201`, 32 training rows, 16 rollouts, 32 optimizer updates, six binary
  decisions, evaluation cadence, and placement contract are byte-identical;
- learning rate `2e-7`, one PPO epoch, clip/default optimizer settings from the
  passing R2 cell, maximum query budget 496;
- one retained global replay group per update, capacity 16, with identical
  replay scoring and backward traversal in both arms.

The treatment is the exact E70 method: success-conditioned signed semantic
Shannon coefficient 0.10 and cap 0.05, open-set inverse adaptation with warmup
64 and EMA 0.90, novelty beta 0.50, split verified-mass and known-mode replay
coefficients 0.10, replay warmups 64 and EMA 0.90, pseudocount 1, surprisal
clip 5, no projection, and no target support or evaluation feedback.

The control retains passive canonicalization, bank updates, replay selection,
scoring, and backward traversal. Semantic, novelty, replay-mass, and
replay-balance effects on parameters are exactly zero. Its manifest label is
`grpo`; its runtime variant is `grpo_compute_matched`.

## Pass criteria

Both jobs must complete the exact 32 updates from the same initial model. Each
must retain the passing six-bit support and finite-training invariants, observe
positive verified reward, and discover at least one two-mode prompt. Prompt,
rollout, optimizer-update, and replay traversal counts must match.

Treatment must show a nonzero applied semantic or novelty advantage and, once
replay is eligible, a nonzero applied replay-gradient norm. Control must report
compute-only replay, exact-zero applied replay-gradient norm, and nonzero raw
replay telemetry whenever an eligible group exists. Both must report no gold
support feedback, no projection, and the global scheduler with one group per
step. All source, ops, model, data, protocol, identity, manifest, scheduler,
metrics, and log hashes are independently audited.

A pass authorizes the ten Pantry Stage-B jobs and nothing else. Failure stops
Pantry before those jobs; no prompt, seed, threshold, pass, or coefficient is
changed after outcome.
