# AntMaze harder-task repair: continuing-task controller v19

**Status:** FROZEN BEFORE V19 OPTIMIZATION OR V19 GATE EXECUTION  
**Date:** 2026-08-04  
**Role:** Secondary post-outcome engineering repair; excluded from the original
80-cell estimator and from the completed AntMaze Stage-B estimator.

## Sealed antecedent and disclosed diagnosis

Controller v18 remains an immutable failed result: 26/96 recorded sequence
successes (0.270833), minimum map success 0.208333, minimum pattern success
zero, and a recorded `unhealthy_termination_rate` of 0.520833. No v18 result
is relabeled or admitted.

Post-outcome source inspection established that the v18 termination label was
confounded. The pinned `AntMaze_UMaze-v5` wrapper discards the underlying Ant
termination returned by `ant_env.step` and defines `terminated` solely as
being within 0.45 of the maze task goal when `continuing_task=False`. Several
v18 closed-loop patterns set the maze goal to the reset cell or cross it before
the commanded sequence is complete. V18 training and evaluation nevertheless
treated this task-goal event as an unhealthy Ant termination. A disclosed
development-only replay on the already-inspected first v18 map changed only
`continuing_task=True`; it removed every one-step termination and reduced the
reported termination rate to zero. V11 and v18 then achieved 0.333 and 0.375
sequence success, respectively, with remaining failures being ordinary
waypoint timeouts. This replay is diagnostic only and is never used for v19
admission.

V19 initializes from the exact admitted v11 checkpoint
`e6d202bd525be5469135b35b63b2bf0884459cc630b71e76f8c49e86dcb8f913`.
It does not initialize from failed v17 or v18 weights. The sealed v18 receipt
hash is
`55398e12736325c8e02791659a439be9a36354112893cd26ba315ab27a6c8229`.

## Frozen intervention

- PPO optimizer reset; seed `73019`; eight CPU workers.
- 6,000,000 transitions at learning rate `5e-7`.
- Four generic 17x17 training maps; no v15 geometry or trajectory.
- Waypoint distance 4.0, positional radius 0.45, segment budget 400, and
  stable-arrival planar speed at most 1.0.
- `continuing_task=True` and `reset_target=False` throughout controller
  training, so the irrelevant maze task goal cannot terminate a waypoint
  curriculum episode.
- Ant health is read explicitly from the pinned inner Ant environment. A
  health failure terminates and penalizes the controller episode.
- The fixed curriculum contains 256 single-waypoint anchors, two copies of
  all 64 ordered heading pairs, all 64 repeated-heading handoffs, and 64
  two-handoff patterns. Its total is 512 patterns of length one through four.
- Stable-arrival braking and reward mechanics otherwise match v18.
- No result-dependent early stopping, checkpoint selection, threshold change,
  or seed change is allowed.

The first v18 development map and trajectories are not loaded by optimization.
The v15 route slate, language prompts or samples, MaxEnt outcomes, and Dr.GRPO
outcomes do not enter optimization.

## Fresh immutable v19 gate

The v19 gate contains 24 frozen length-eight patterns on four new 21x21 generic
maps, for 96 episodes. The pattern slate is disjoint from every v19 training
pattern and from the v17 and v18 development pattern slates. None of the v19
maps appeared in an earlier controller gate. Position noise is 0.1 and the
evaluation seed offset is 13,000,000. Training never accesses a v19 gate map,
pattern trajectory, or outcome.

Gate execution also uses `continuing_task=True`, keeps the fixed maze task
goal unchanged, and reads Ant health explicitly. All checks must pass:

- exactly 96 episodes;
- overall sequence success at least 0.90;
- success at least 0.75 for every one of the 24 patterns;
- success at least 0.75 on every map;
- explicit Ant health-failure rate at most 0.10;
- zero maze-task termination events;
- median successful segment length at most 300;
- every recorded arrival has planar speed at most 1.0;
- all final distances and speeds finite.

Failure stops v19. Passing authorizes only a separately frozen v19 worker
binding and executable admission on fresh route identities. It does not by
itself authorize a language-model cohort.

## Downstream firewall

The completed original AntMaze Stage B remains independently terminal and
eligible. Any harder-route extension must, after a v19 pass, freeze a new
executor identity, route admission slate, cross-node check if required, and
fresh 0.5B development viability gate before any paired or five-seed LM run.
No v15/v18 LM sample was launched, and no v19 LM sample is authorized here.
