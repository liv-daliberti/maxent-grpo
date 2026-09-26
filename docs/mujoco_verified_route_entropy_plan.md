# 0.5B language policies in MuJoCo goal-reaching environments

Status: PointMaze viability failed; AntMaze route admission failed after controller v5 passed; no maze training runs  
Date: 2026-07-29

## Admission outcome

PointMaze passed its executable audit: 32 real routes over 16 maps, 3,200
registered perturbations with zero failures, and 5.56 executions/s against the
2/s floor. The frozen Qwen2.5-0.5B viability gate and its one permitted
assistant-prefill repair each produced zero verified routes. PointMaze
therefore stops before training.

AntMaze controller v5 passed its prospective eight-heading command-following
gate. Development tuning then established upper and lower successful route
programs, but the independently frozen route-admission slate failed on its
first fresh map. AntMaze is therefore ineligible before data materialization or
0.5B sampling; the frozen map slate was not replaced after observing failure.

## Corrected decision

`PointMaze` and `AntMaze` are environments, not policy classes. They therefore
belong in the requested 0.5B comparison when the policy is the same language
model used by the other rows.

The model receives a textual maze/start/goal specification and emits one
bounded action program. A trusted, networkless worker executes that program in
the pinned MuJoCo environment. Environment success gives task reward; a
topology-bound route extractor gives a mode key only for successful rollouts.
The two experimental arms remain objective-equivalent plain Dr.GRPO and the
frozen E58-style online verified MaxEnt treatment.

This is a language action-program interface to continuous-control
environments. It is not conventional neural-network PPO, and it is not an
action-entropy baseline.

## Action interfaces

### PointMaze

The output alphabet is a preregistered finite codebook of quantized 2-D force
vectors: the eight compass directions plus `COAST`. Each token is applied for
a fixed number of simulator steps. The response may contain at most the frozen
horizon's number of tokens. No magnitude, duration, or free-form numeric
parameter is model-controlled.

### AntMaze

The output alphabet is the preregistered set of eight compass-heading locomotion commands. Controller v5 has no `STOP` token. Each command is held for a fixed
control window and is executed by one frozen low-level Ant controller shared
by both arms. The controller, weights, observation normalization, command
period, and action clipping are part of the environment snapshot and cannot
learn during the 0.5B experiment.

AntMaze must be labeled “0.5B language action program + frozen locomotion
controller” in the figure. That is still a 0.5B language policy over the
environment, but it must not be described as the LM directly emitting raw
20-Hz joint torques.

## Why these environments fit

The official PointMaze environment uses a continuous two-dimensional force
action, a discrete maze map, custom maze support, and a sparse success reward
when the achieved goal is within 0.5 m of the desired goal. Start and goal
cells can be fixed at reset. AntMaze uses the same maze task with the
eight-actuator MuJoCo Ant.

Official references:

- PointMaze: <https://robotics.farama.org/envs/maze/point_maze/>
- AntMaze: <https://robotics.farama.org/envs/maze/ant_maze/>
- MuJoCo: <https://mujoco.readthedocs.io/en/stable/overview.html>

The audit runtime is pinned to Python 3.11.7, `gymnasium` 1.2.2,
`gymnasium-robotics` 1.4.2, `mujoco` 3.10.0, and `numpy` 2.4.6. Its Point and
Ant environment sources and XML assets are hash-bound in the admission
receipts. Only the frozen PointMaze development viability samples have been taken; no maze training run or evaluation split has been sampled.

## Executable success and identity

For a frozen maze, reset seed, start, goal, and episode horizon:

1. parse the bounded action program against the row-specific alphabet;
2. reset the hash-pinned environment with the registered seed;
3. execute exactly the fixed token-to-control mapping;
4. require the environment's sparse goal condition;
5. recover directed bottleneck crossings from the achieved-goal trajectory;
6. collapse consecutive repeat crossings with registered hysteresis; and
7. use the ordered directed gate sequence as the route key.

Micro-actions, speed, body pose, controller noise, and nearby continuous
trajectories collapse when they traverse the same topological route. Routes
passing different obstacle sides or corridor sequences remain distinct.
Malformed programs, simulator exceptions, NaNs, wall penetration, timeouts,
goal near misses, ambiguous gate contacts, and unsuccessful episodes receive
reward zero and no key.

The verifier returns both success and identity from the same worker response.
The trainer never imports MuJoCo and never accepts an LM-declared route label.

## Admission gate

Before policy sampling:

- freeze at least four maps with two to four topological start-goal routes;
- enumerate the free-cell graph and its admissible route catalogue;
- place bottleneck gates without inspecting learned trajectories;
- replay at least 100 perturbed successful trajectories per route;
- require route-key invariance to reset noise, timing, and sub-cell variation;
- require distinct obstacle-side routes never to collide;
- test loops, recrossings, corner grazing, timeout, and near-miss failures;
- demonstrate at least two successful action programs per admitted prompt;
- measure worker throughput under the exact networkless execution boundary;
- for AntMaze, separately certify the frozen controller's command following;
  and
- hash every map, gate set, action codebook, controller asset, simulator asset,
  prompt row, and audit artifact.

No PointMaze or AntMaze row is admitted merely because its parser unit tests
pass.

## Matched 0.5B arms

Use the same checkpoint within each row and the same OAT prompt-completion
training path as the other benchmark domains.

1. **Dr.GRPO:** `grpo_compute_matched`, with the task objective unchanged and
   the same passive verifier/replay compute as treatment but zero replay
   derivative.
2. **Verified MaxEnt:** the frozen E58
   `verified_first_global_replay_canonical` treatment: semantic coefficient
   0.10, novelty beta 0.50, replay alpha 0.10, one persistent-hash
   round-robin replay group per update, capacity 16, warmup 64.

Match checkpoint, prompt order, rollouts, optimizer steps, token limits,
simulator executions, reset/evaluation seeds, and evaluation draws. A route
key may enter the verified bank only after the environment reports success.
No gold route catalogue, target entropy, or evaluation trajectory enters
training.

## Metrics and reporting

Use the shared four-column surface:

- greedy success;
- `mean@8`: mean successful-rollout fraction across the eight samples;
- `pass@8`: probability that at least one of eight programs succeeds; and
- `distinct@8`: mean number of distinct successful route keys.

Also retain route entropy conditional on success, certified route coverage,
path length, simulator failure counts, state-visitation heat maps, environment
steps, verifier throughput, and wall-clock cost as diagnostics.

The rows participate in the requested all-environment 0.5B figure, but pooled
effect estimates must retain an interface stratum: AntMaze has a frozen
low-level controller, while the other rows do not.

## Recorded ladder outcome

1. Passed: PointMaze parser, worker, route identity, data split, deterministic
   execution, perturbation, and throughput gates.
2. Failed: PointMaze initial and one-time repaired 0.5B development viability
   gates, each with zero verified completions.
3. Passed: prospective Ant controller v5 eight-heading command-following gate.
4. Failed: frozen AntMaze route admission on the first fresh map; no map
   replacement or 0.5B sampling followed.
5. Stopped: both maze rows before training and the immutable 80-job atomic
   cohort. E70 Stage A proceeds only on the four established non-maze rows.

A failed environment remains an explicit ineligible row rather than being
silently replaced by conventional PPO or dropped from the requested surface.
