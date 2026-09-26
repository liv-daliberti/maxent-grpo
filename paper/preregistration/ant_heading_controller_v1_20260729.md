# Ant heading controller v1

Status: frozen before full training
Frozen: 2026-07-29

## Role and firewall

Train one command-conditioned low-level Ant controller to serve as an immutable
part of the AntMaze environment interface. This is not a MaxEnt or Dr.GRPO
arm. It is trained once on open-plane `Ant-v5`, then shared byte-for-byte by
both 0.5B language-policy arms.

The training process may not load a maze map, route gate, ModeBench prompt,
0.5B completion, verified bank, treatment label, or confirmatory outcome. Its
only command is one of eight compass unit vectors or the zero vector.

## Runtime

- Python 3.11.7;
- Gymnasium 1.2.2;
- Gymnasium-Robotics 1.4.2;
- MuJoCo 3.10.0;
- Stable-Baselines3 2.7.1;
- CPU PyTorch 2.6.0;
- NumPy 2.4.6; and
- training source: `ops/train_ant_heading_controller.py`.

The runtime package inventory, controller ZIP, training source, worker source,
and evaluation receipt are SHA-256 bound before AntMaze admission.

## Frozen training

- environment: `Ant-v5`, no maze walls;
- seed: 73001;
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

Each episode samples one of the nine commands uniformly and holds it fixed.
For a nonzero command, the shaped reward is twice projected velocity minus
0.25 times absolute lateral velocity, plus the native healthy term and scaled
native control/contact terms. For the zero command, it penalizes speed. This
reward is auxiliary controller engineering and is never reported as a paper
outcome.

The completed 2,048-step smoke model is execution validation only and is
ineligible for AntMaze.

## Frozen open-plane admission test

After training, run exactly three deterministic 300-step episodes for each of
the nine commands using new seeds derived from `1073001`. Admit the controller
to the maze gate only if all conditions hold:

- every one of the eight headings has mean projected displacement at least
  2.0 MuJoCo distance units;
- mean projected displacement across heading episodes is at least 4.0;
- early unhealthy termination rate across all 27 episodes is at most 0.10;
- mean displacement under the zero command is at most 2.0; and
- every evaluation episode and metric is present and finite.

No retraining, seed replacement, coefficient change, or threshold change is
allowed after this result is inspected. A failure makes AntMaze ineligible for
the requested cohort unless a new controller version is prospectively designed
and clearly separated.

## Subsequent maze gate

Passing open-plane command following is necessary but not sufficient. The
frozen controller must then execute at least two topologically distinct
successful routes on every admitted AntMaze development map, pass route-key
perturbation/collision tests, and meet the registered worker throughput bound.
No controller parameter changes after the open-plane gate.

## Artifacts

- model: `var/maze_runtime/controllers/ant_heading_v1.zip`;
- receipt: `var/maze_runtime/controllers/ant_heading_v1.training.json`;
- Slurm log: `var/artifacts/logs/ant-heading-controller-v1-<job>.{out,err}`;
- runtime audit: `var/artifacts/maze_runtime_audit.json`; and
- eventual maze admission audit:
  `var/artifacts/ant_maze_controller_admission_audit.json`.
