# E111 proposal-retention resume recovery amendment (2026-08-18)

## Trigger

Three released E111 mechanism-gate jobs (30674729, 30674733, and 30674754) completed valid optimizer updates and wrote checkpoints, but a later restart loaded the model and optimizer state and then failed before another optimizer update with `ValueError: proposal retention state refers to a non-proposal exemplar`. This amendment uses only runtime logs and checkpoint structure; no task endpoint, evaluation breadth, correctness, or E111 outcome is inspected.

## Root cause

A proposal-admitted exemplar begins in `proposal_only_outcomes`. If the neutral policy later produces that verified outcome, `score_and_update` correctly adds it to the on-policy count predictor, removes the proposal-only label, and leaves the longitudinal admission-retention record in place with `converted_on_policy=True`. The checkpoint writer preserves this valid lifecycle. The old loader nevertheless required every tracked admission to remain proposal-only forever, contradicting the tracker schema and rejecting valid converted state.

## Frozen repair

The shared E111 runtime source is changed only in `admission_retention.py` and `online_canonical_bank.py`. The tracker exposes which admissions have converted on-policy. Resume validation now requires every tracked admission to retain an exemplar and accepts exactly one of two states: (1) unconverted and proposal-only, or (2) converted and present in the neutral on-policy counts. Mixed, missing, or falsely relabeled states still fail closed. The serialized schema, counters, exemplars, optimizer state, model state, and training-progress state are unchanged.

This is a checkpoint-deserialization invariant repair. It changes no prompt, sample, reward, canonicalizer, proposal generator, support admission, replay schedule, replay weight, semantic advantage, loss, gradient, optimizer update, evaluation cadence, model placement, seed, or treatment environment. Existing in-memory processes are not signaled or reset; the repaired validation is used only on a later process start or restart.

## Evidence and scope

Before installation, the old and new source digests are recorded. The full online-canonical-bank production suite must pass, including a regression that saves a proposal after neutral on-policy conversion, reloads it exactly, and continues to reject inconsistent lifecycle state. The installed runtime files must exactly match those tested root files. The repair applies prospectively to all 15 released E111 cells because they share the same source snapshot, while the three observed failures remain the trigger evidence. PointMaze remains excluded.
