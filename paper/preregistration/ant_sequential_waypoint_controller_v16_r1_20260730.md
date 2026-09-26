# Ant sequential-waypoint controller v16 r1

**Status: FROZEN DURING CONFIGURATION AND BEFORE TRAINING OR EVALUATION — 2026-07-30**

The v16 implementation initially passed total seven-waypoint episode duration
to the inherited single-waypoint 300-step speed check, although the protocol
intended a controller-speed check. Before any v16 job was submitted or any
v16 weights were trained, r1 defines `median_success_steps` as the median
duration of the 448 constituent successful waypoint segments. It separately
records median total sequence duration without gating on it.

No map, curriculum, pattern, seed, initialization, optimizer, budget,
learning rate, reward, success threshold, or evaluation outcome changes.
