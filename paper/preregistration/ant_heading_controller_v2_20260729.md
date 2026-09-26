# Ant heading controller v2

Status: frozen before training
Frozen: 2026-07-29

## Why this is a new prospective version

Controller v1 passed its frozen heading-motion and health gates but failed its
stationary-command gate: mean STOP displacement was 3.2235 against a maximum
of 2.0. The v1 model, receipt, source hash, and Slurm logs remain immutable and
ineligible. No maze route or 0.5B policy outcome has been observed.

Version 2 makes exactly these prospective controller changes:

- the zero command is sampled on one half of open-plane training episodes,
  while each of the eight headings is sampled with probability `1/16`;
- the controller observes displacement from the current primitive's starting
  position in addition to the native Ant observation and command; and
- the zero-command reward penalizes `4 * speed + 1.5 * displacement`.

The nonzero heading reward, PPO architecture, optimizer settings, evaluation
seeds, and admission thresholds are unchanged from v1. These changes were
frozen before v2 training or evaluation.

## Role and information firewall

Train one command-conditioned low-level Ant controller to serve as an
immutable part of the AntMaze environment interface. It is not a MaxEnt or
Dr.GRPO arm and is shared byte-for-byte by both 0.5B language-policy arms.

Training may load only open-plane `Ant-v5`. It may not load a maze map, route
gate, ModeBench prompt, 0.5B completion, verified bank, treatment label, or
confirmatory outcome. The local displacement feature is reset to zero at the
start of each commanded primitive during eventual maze execution.

## Frozen runtime and training

- Python 3.11.7;
- Gymnasium 1.2.2;
- Gymnasium-Robotics 1.4.2;
- MuJoCo 3.10.0;
- Stable-Baselines3 2.7.1;
- CPU PyTorch 2.6.0;
- NumPy 2.4.6;
- source: `ops/train_ant_heading_controller_v2.py`;
- seed: 73002;
- total timesteps: 3,000,000;
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

For a nonzero command, reward is twice projected velocity minus 0.25 times
absolute lateral velocity, plus the native healthy term and scaled native
control/contact terms. For STOP, the command reward is negative four times
speed minus 1.5 times displacement from the primitive start, with the same
native terms. This auxiliary reward is never a paper outcome.

## Frozen open-plane admission test

After training, run exactly three deterministic 300-step episodes for each of
the nine commands using seeds derived from `1073002`. Admit v2 only if:

- every heading has mean projected displacement at least 2.0;
- overall mean heading-projected displacement is at least 4.0;
- early unhealthy termination rate across all 27 episodes is at most 0.10;
- mean STOP displacement is at most 2.0; and
- all 27 episode records and all metrics are present and finite.

No retraining, seed replacement, coefficient change, or threshold change is
allowed after this result is inspected. A failure makes v2 ineligible.

## Subsequent maze gate

Passing open-plane command following is necessary but not sufficient. The
frozen controller must execute at least two topologically distinct successful
routes on every admitted development map, pass route-key
perturbation/collision tests, and meet the registered worker throughput bound.
No controller parameter may change after the open-plane result.

## Artifacts

- v1 failure receipt:
  `var/maze_runtime/controllers/ant_heading_v1.training.json`;
- v2 model: `var/maze_runtime/controllers/ant_heading_v2.zip`;
- v2 receipt: `var/maze_runtime/controllers/ant_heading_v2.training.json`;
- Slurm log:
  `var/artifacts/logs/ant-heading-controller-v2-<job>.{out,err}`; and
- eventual admission audit:
  `var/artifacts/ant_maze_controller_v2_admission_audit.json`.
