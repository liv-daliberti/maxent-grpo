# E75R1: PointMaze waypoint 0.5B weak-warm-start successor

Date frozen: 2026-08-04  
Status at freeze: prospective, hypothesis-generating successor to closed E75

## Prior result and reason for a new experiment

E75 stopped at its prospectively frozen development gate. Its three-epoch,
348-update shared warm start reached mean success 0.97265625 on all 32
development maps, with at least two verified routes on 31/32 maps. The allowed
admission interval was [0.10, 0.70], so the three online jobs were correctly
blocked. E75 remains immutable and excluded from this successor's online
estimates.

E75R1 tests the same waypoint-policy mechanism with two changes fixed before
any E75R1 model is trained or evaluated: a fresh geometry-disjoint dataset and
a 32-update shared warm start. The development gate, online arms, optimizer,
sample count, horizon, and final-evaluation firewall are unchanged.

## Immutable domain and fresh split

The domain remains `point-waypoint-v1`. The language policy selects one
adjacent prompt-visible free cell. Its legal mask removes walls only and does
not use goal distance, route identity, shortest paths, or revisit history. A
hash-bound deterministic PD controller performs local MuJoCo actuation.
Success and canonical route identity come from the continuous trajectory, and
a verified completion must cross exactly one directed corridor gate.

The deterministic E75R1 data seed is 88101:

- train: 64 maps
- dev: 32 maps
- eval: 64 maps
- each map: three to five separated barrier corridors
- rotations are exactly balanced within each split
- fingerprints cannot overlap across splits or with any of E75's 160 maps

The old E75 dataset identity and all 160 excluded fingerprints are hash-bound
into the new data identity. Every advertised route must execute successfully
in the pinned MuJoCo runtime before admission. Certified route programs are
never placed in model prompts.

## Fixed weak warm start

Base model:
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.

The shared warm start uses only E75R1 train maps and the same route-balanced,
dynamic-legal-support cross-entropy objective as E75.

- SFT seed: 88402
- shuffled epoch budget: 1
- fixed optimizer-update cap: 32
- effective batch: 32 decisions
- learning rate: 2e-5
- warmup: 10 updates, followed by linear decay over the fixed 32-update budget

The 32-update cap was selected once, before E75R1 data materialization or model
evaluation, from E75's exposed training trajectory: E75 first approached
in-sample action accuracy 0.6--0.7 around updates 30--40. There is no adaptive checkpoint
search and no second E75R1 warm-start attempt if the gate fails.

## Unchanged development gate

The shared warm start is evaluated once on all 32 fresh development maps.
Online jobs use an `afterok` dependency and run only if all criteria pass:

- at least 24/32 maps have a success among eight samples;
- at least 16/32 maps expose at least two canonical routes among eight;
- every 3-, 4-, and 5-corridor family has at least one successful map;
- aggregate mean success lies in [0.10, 0.70];
- receipt, metric, and dataset hashes agree;
- the gate performs zero optimizer updates.

A failed gate is the terminal E75R1 result. Thresholds and the warm-start budget
must not be changed in place.

## Online arms

All admitted arms start from the identical E75R1 warm-start checkpoint and use
common random-number schedules with online seed 88502:

1. `grpo`
2. `verified_first_global_replay_canonical`
3. `verified_first_delayed_singleton_replay_canonical`

Each arm receives 64 training-map updates, 16 samples per update, a 64-decision
horizon, replay capacity 16, and learning rate 2e-7. The matched control
traverses replay slots with zero replay derivative. Both treatments use the
same verified novelty signal. The delayed arm suppresses verified-mass replay
until a second route has been discovered for that prompt.

Primary development outcomes are mean success, pass@8, distinct verified
routes@8, and modes per successful sample, reported per map and in aggregate.

## Evaluation firewall and stopping

No final-evaluation job is submitted. Arm selection, exclusions, stopping
decisions, and final analysis must be frozen before one saved checkpoint is
evaluated once on all 64 E75R1 evaluation maps. Development results are
exploratory and cannot be described as untouched evaluation results.

The dependency chain is full E75R1 data certification and train-route
materialization, 32-update shared SFT, frozen development gate, then the three
one-pass online arms. GPU stages use one A6000 on node208 through partition
`all`; jobs do not requeue automatically. Missing artifacts, geometry
overlap, hash drift, non-finite values, worker errors, gate failure, or
scheduler-contract mismatch fail closed. Replacement jobs require a new
identity rather than overwriting E75R1.
