# Ant maze-blind waypoint controller v7

**Status: FROZEN BEFORE V7 TRAINING — 2026-07-29**

## Antecedent and purpose

Controller v6 was frozen before execution and failed its fresh open-plane gate:
65/96 successes (0.677), with per-heading rates from 0.333 to 0.833. It
nonetheless had finite metrics, only 0.0625 unhealthy termination, and median
211 successful steps. The v6 evaluation seeds and its earlier maze route slate
are consumed development evidence. They are not reused here.

V7 tests whether the mismatch came from spending most curriculum episodes on
shorter waypoints and randomly imbalanced headings. This is development-only.
It cannot become a paper result and cannot authorize any language-model arm.

## Frozen training intervention

- initialization: exact v6 model
  `ac21f650cb25141670d2316b8f9dd691956e6c23d97791152a826835d61b7d17`;
- optimizer: reset PPO optimizer, as in v6's v1 initialization;
- environment: open-plane `Ant-v5`, never AntMaze;
- observation: Ant proprioception plus normalized relative waypoint vector;
- target distance: exactly 4.0 units, matching the route runtime;
- heading schedule: eight compass headings, cyclically offset by worker rank;
- workers: 8; episode horizon: 400;
- training seed: 73007; total transitions: 5,000,000;
- learning rate: 1e-5; all other PPO and reward settings unchanged from v6.

The intervention was chosen from aggregate v6 diagnostics. No v6 episode
trajectory, maze map, maze gate observation, language-model output, MaxEnt
outcome, Dr.GRPO outcome, or evaluation prompt is loaded.

## Fresh evaluation and decision

Evaluation uses seed base 2,073,007 (`training seed + 2,000,000`), which has
not appeared in any earlier Ant controller gate. There are 12 deterministic
episodes for each of eight headings at distance 4.0 (96 total).

Pass only if all hold:

- overall success rate at least 0.90;
- every heading success rate at least 0.75;
- unhealthy termination rate at most 0.10;
- median successful episode at most 300 steps;
- all terminal metrics finite.

A pass permits exactly one separately frozen fresh-map route gate. A fail makes
v7 ineligible; thresholds cannot be relaxed and no map may be substituted.
