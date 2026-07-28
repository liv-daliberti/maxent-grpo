# E69 Gate 2 temporal route-observer amendment

Date frozen: 2026-07-28, before any pass-5 or pass-6 Gate 2 evaluation
existed and while every Gate 2 physical run was nonterminal.

## Reason for the measurement correction

The frozen training implementation first calls `observe_neutral` for the
current prompt batch and only afterward calls
`scheduled_cross_prompt_replay_groups`. The original
`post_replay_cross_prompt_neutral_reproductions` counter increments only when
that prompt-route record is first created or when a proposal-only record first
graduates. It does not increment when an already-neutral route is reproduced
on a later visit. Therefore, for an ordinary neutral discovery, the replay
marker is necessarily installed after the only event that the counter accepts.
The counter is not a valid implementation of the preregistered temporal
diagnostic.

This was identified from execution order and the counter definition after live
telemetry showed zero accepted events despite active cross-prompt replay. No
terminal Gate 2 outcome, pass-5 outcome, or pass-6 outcome was available. No
task-quality threshold, arm, seed, coefficient, checkpoint, or training datum
is changed.

## Frozen prospective observer

Training and its frozen source snapshot remain untouched. An independent,
read-only observer captures the checkpointed verified-route-library state at
every pass checkpoint for the four executable domains. For each hashed
route-prompt pair it stores:

1. whether the pair had already been marked as a target of cross-prompt replay;
2. the neutral observation count at that checkpoint.

A post-replay neutral reproduction is counted only when:

1. a pair is already replay-marked in checkpoint `t`;
2. the same replay-target pair remains present at a later checkpoint `t+1`;
3. its neutral count strictly increases between those checkpoints.

The increase is a conservative temporal lower bound: every counted neutral
observation occurred after a checkpoint at which cross-prompt replay for that
route-target pair had already happened. Pair identities are stored only as
SHA-256 values. Snapshot sets and counts must be monotone; any disappearance or
decrease is a hard integrity failure.

The observer ingests every completed pass checkpoint deterministically. A CPU
monitor runs every three minutes because training retains only two optimizer
checkpoints. Existing snapshots are immutable: re-observing a recorded pass
must produce the same compact state.

## Gate interpretation

The preregistered mechanism condition remains uniform and unchanged in spirit:
at least three executable domains must show at least one neutral reproduction
on a route-prompt pair that had previously been targeted by cross-prompt
replay. The corrected checkpoint observer replaces only the structurally
incapable in-process counter for this condition.

The task-quality and support gates remain exactly as frozen. The temporal
diagnostic remains observational rather than a causal estimate; causal claims
continue to come from the compute-matched arm comparison.
