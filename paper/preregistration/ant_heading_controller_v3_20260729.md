# Ant heading controller v3

Status: frozen before training  
Frozen: 2026-07-29

## Prospective separation

Controller v1 passed all navigation-heading gates but failed the optional STOP
gate. Controller v2 passed STOP but failed every navigation and health gate.
Both weights, receipts, logs, sources, and failure decisions remain preserved.
Neither version saw a maze map, route, language completion, or experimental
arm outcome.

Version 3 removes STOP from the environment interface. The 0.5B AntMaze
policy emits one of exactly eight compass headings. STOP is not needed to
reach a goal and is not replaced by another action. Version 3 is trained from
scratch on those eight headings; it does not reuse either failed model.

## Frozen training

- environment: open-plane `Ant-v5`, no maze walls;
- commands: N, NE, E, SE, S, SW, W, and NW, uniformly sampled;
- seed: 73003;
- total timesteps: 2,000,000;
- parallel workers: 8;
- episode horizon: 300;
- policy: PPO MLP `[256, 256]`;
- learning rate: `3e-4`;
- per-worker rollout length: 256;
- aggregate batch size: 512;
- epochs per update: 10;
- gamma: 0.99;
- GAE lambda: 0.95;
- PPO clip: 0.2; and
- explicit entropy coefficient: 0.

The heading reward and native reward scaling are byte-for-byte inherited from
the v1 source. The v3 source and the inherited v1 source are both hash-bound
in the receipt.

## Frozen open-plane admission

Run three deterministic 300-step episodes per heading using seeds derived
from `1073003`. Admit v3 only if:

- each heading's mean projected displacement is at least 2.0;
- overall mean projected displacement is at least 4.0;
- early unhealthy termination across all 24 episodes is at most 0.10; and
- every episode and metric is present and finite.

No training rerun, replacement seed, coefficient change, checkpoint
selection, or threshold change is allowed after inspection. Failure makes v3
and the requested AntMaze row ineligible.

## Firewall and subsequent gate

Only open-plane Ant may be loaded during training and this admission test.
Passing it only permits the frozen model to enter the separately specified
maze route and throughput audit. Both 0.5B arms must use the same controller
bytes, command duration, observation construction, and clipping.

Artifacts:

- model: `var/maze_runtime/controllers/ant_heading_v3.zip`;
- receipt: `var/maze_runtime/controllers/ant_heading_v3.training.json`; and
- logs: `var/artifacts/logs/ant-heading-controller-v3-<job>.{out,err}`.
