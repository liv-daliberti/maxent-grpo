# Ant maze-blind waypoint controller v8

**Status: FROZEN BEFORE V8 TRAINING — 2026-07-29**

## Antecedent

V7 remains a failed development gate. On its fresh 96-episode open-plane
evaluation it achieved 86/96 successes (0.8958), versus the frozen 0.90
threshold. Heading 1 achieved 8/12 (0.667), versus the unchanged 9/12
per-heading requirement. All other headings were at least 10/12, unhealthy
termination was 2/96, and median successful duration was 169.5 steps. No v7
episode trajectory is reused below.

## Prospective v8 intervention

V8 tests whether the near-threshold v7 controller benefits from a conservative
continuation rather than another reward or interface change.

- initialization: exact v7 model
  `2bb5d192698639a294ac8266ae9be68484c9d493f4f7ba4317ee06d2d4b04442`;
- reset PPO optimizer;
- 3,000,000 additional open-plane `Ant-v5` transitions;
- learning rate `5e-6`;
- seed `73008`, 8 workers, 400-step horizon;
- exactly the v7 four-unit targets, cyclic eight-heading schedule, reward,
  observation, network, PPO settings, and 0.45 success radius; and
- no heading-specific oversampling selected from the v7 evaluation.

The training process may load the v7 weights but no v7 evaluation episode,
maze map, route-gate observation, language-model output, MaxEnt outcome, or
Dr.GRPO outcome.

## Fresh evaluation and decision

Evaluation uses seed base `3073008` (`73008 + 3,000,000`), absent from every
earlier controller gate, with 12 deterministic episodes for each of the same
eight headings. The v6/v7 thresholds remain unchanged:

- overall success at least 0.90;
- every heading at least 0.75;
- unhealthy termination at most 0.10;
- median successful duration at most 300 steps; and
- all metrics finite.

A pass permits exactly one separately frozen fresh-map v8 route gate. A fail
makes v8 ineligible. No threshold may be relaxed and no map may be substituted.
