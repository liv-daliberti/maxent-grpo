# E75: PointMaze waypoint 0.5B mechanism pilot

Date frozen: 2026-08-03  
Status at freeze: prospective, hypothesis-generating

## Question

Can the existing verified-discovery and replay approach improve route diversity
in PointMaze when the language policy controls adjacent-cell waypoints instead
of raw force tokens?

The experiment also tests a specific failure mechanism from the earlier
force-policy work: immediate verified-likelihood replay may entrench the first
route before another mode is discovered.

## Immutable domain

The domain is `point-waypoint-v1`. The language policy chooses one adjacent
prompt-visible free cell. Its legal mask removes walls only and does not use
goal distance, route identity, shortest paths, or revisit history. A
hash-bound deterministic PD controller performs local MuJoCo actuation.
Success and canonical route identity are computed from the continuous
trajectory. A verified completion must cross exactly one directed corridor
gate.

The source and execution trees, controller hash, environment hash, data
identity, base-model tree, and protocol are bound in the launch identity.

## Data firewall

The deterministic seed is 88100.

- train: 64 maps
- dev: 32 maps
- eval: 64 maps
- each map: three to five separated barrier corridors
- rotations are exactly balanced within each default split
- fingerprints cannot overlap across splits

Every advertised logical route must execute successfully in the pinned MuJoCo
runtime before the dataset is admitted. Certified programs remain in
`identity.json`; they are not placed in model prompts.

The warm start uses train maps only. Every certified train route is replayed
through the public worker. Per-decision loss is normalized over the legal
action subset, and inverse route-length weights give each demonstrated route
equal summed weight per epoch.

The evaluation split is not loaded by SFT, development qualification, or
online training.

## Fixed model and seeds

Base model:
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.

- SFT seed: 88401
- online/evaluation seed: 88501
- SFT epochs: 3
- SFT effective batch: 32 decisions
- online samples per update: 16
- online horizon: 64 decisions
- online passes: 1 over 64 training maps
- replay capacity: 16
- evaluation K: 8

## Development gate

The shared warm start is evaluated once on all 32 development maps. Online
jobs have an `afterok` dependency on a qualification program and cannot run
unless all criteria pass:

- at least 24/32 maps have a success among eight samples;
- at least 16/32 maps expose at least two canonical routes among eight;
- each of the 3-, 4-, and 5-corridor strata has at least one successful map;
- aggregate mean success is in [0.10, 0.70];
- the receipt and metric hashes match the dataset identity;
- the evaluation performed zero optimizer updates.

A failed gate is a result. Thresholds must not be changed in place.

## Online arms

All admitted arms start from the identical warm-start checkpoint and use
common random-number schedules.

1. `grpo`
2. `verified_first_global_replay_canonical`
3. `verified_first_delayed_singleton_replay_canonical`

The control traverses compute-matched replay slots with zero replay
derivative. Both treatments use the same verified novelty signal. The delayed
arm disables verified-mass replay while only one route is known for a prompt;
the unchanged mass-plus-balance replay activates once a second route is known.

Primary development outcomes are mean success, pass@8, distinct correct
routes@8, and modes per successful sample, reported per map and in aggregate.

## Evaluation firewall

No final-evaluation job is submitted with this launch. Arm selection,
exclusions, stopping decisions, and the final analysis must be frozen before
one saved checkpoint is evaluated once on all 64 evaluation maps. Development
metrics are exploratory and may not be reported as untouched evaluation
results.

## Execution and stopping

The launch is a dependency chain:

1. full data certification and train-route materialization;
2. shared 0.5B SFT;
3. frozen development gate;
4. three one-pass online arms, only after a passing gate.

Jobs use one A6000 each for GPU stages. Jobs do not requeue automatically.
Missing artifacts, hash drift, non-finite values, worker errors, gate failure,
or scheduler-contract mismatch fail closed. Replacement jobs require a new
identity rather than overwriting this experiment.
