# AntMaze harder-task repair: sequential controller v17

**Status:** FROZEN BEFORE V17 OPTIMIZATION  
**Date:** 2026-07-30  
**Role:** Secondary post-outcome engineering repair; excluded from the original
80-cell estimator.

## Antecedent and reason for a new gate

Controller v16 completed its immutable 64-episode gate and failed with 25%
sequence success, 25% success on every development map, and zero success for
12 of 16 held-out patterns. The failure was healthy and finite; it exposed
loss of directional/turn competence under sequential hand-offs. The v16 model
and its failed development trajectories are not used as v17 initialization or
training data.

V17 restarts from the exact admitted v11 weights
`e6d202bd525be5469135b35b63b2bf0884459cc630b71e76f8c49e86dcb8f913`.
The v16 aggregate outcome is disclosed as an adaptive engineering antecedent.

## Frozen training intervention

- PPO optimizer is reset.
- Seed: `73017`.
- CPU workers: 8.
- Total transitions: 4,000,000.
- Learning rate: `2e-7`.
- Waypoint distance: 4 MuJoCo units.
- Segment budget: 400 environment steps.
- Success radius: 0.45.
- Episode budget: 2,400 steps.
- Four generic 15x15 maps contain only peripheral diagnostic walls.
- The curriculum contains all eight cardinal/diagonal headings.
- Sixty-four single-waypoint anchor entries preserve the admitted reset-state
  directional envelope.
- Twenty-four longer entries cover clockwise, counter-clockwise, and diagonal
  hand-offs.

The anchor/sequence mixture is chosen before optimization. No result-dependent
early stopping or checkpoint selection is allowed.

## Fresh immutable gate

The development gate contains 24 sequence patterns on four new 17x17 maps,
for 96 episodes. It uses nonzero reset noise and seed offset 9,000,000. No
development trajectory is accessed by the training environment or optimizer.
No development pattern is identical to a training sequence.

The receipt passes only if all checks hold:

- exactly 96 episodes;
- overall sequence success at least 0.90;
- success at least 0.75 for every one of the 24 patterns;
- success at least 0.75 on every map;
- unhealthy termination rate at most 0.10;
- median successful segment length at most 300 steps;
- all final distances finite.

Failure stops v17. The v15 map, v15 route slate, language prompts, language
samples, MaxEnt outcomes, and Dr.GRPO outcomes are never substituted into this
controller gate.

## Downstream authorization

A passing v17 receipt authorizes only a new, separately frozen binding of the
unchanged v15 harder AntMaze map/route slate to controller v17. That binding
must pass executable route admission and a fresh 0.5B viability gate before
any paired or five-seed language-policy run is launched. A v17 failure is a
recorded negative controller result.
