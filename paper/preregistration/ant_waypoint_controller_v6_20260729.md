# Ant waypoint controller v6

**Status: FROZEN BEFORE FULL V6 TRAINING — 2026-07-29**

## Separation and purpose

The v5 heading controller passed its open-plane gate, but its frozen route
slate failed on the first fresh map. That outcome remains final; no map or
route is replaced. Post-outcome feedback experiments around the unchanged v5
weights were development-only and failed to give one symmetric rule for both
obstacle sides.

V6 is a separately versioned waypoint controller. A future language policy
will choose adjacent grid cells; v6, if admitted, may handle only locomotion
to the selected cell. Full v6 training and evaluation load open-plane
`Ant-v5` only. They may not load a maze map, route gate, language prompt,
MaxEnt result, Dr.GRPO result, or v5 fresh-map trajectory.

## Development record

Three 50k-step smokes are permanently ineligible. Random initialization gave
0/8 waypoint successes. Warm initialization at learning rate `3e-4` gave 1/8
and unhealthy termination 0.75. Lower-rate warm initialization gave 1/8 and
unhealthy termination 0.125. The final maze-blind 1--4-unit curriculum gave
2/8. These smokes fixed the input scaling, learning rate, health penalty, and
curriculum before this freeze. Their weights may not initialize the full run.

## Frozen controller training

- initialization: exact open-plane v1 heading-policy weights, SHA-256
  `526b669bb14cf8a07b44a1c725f94411632a3ac8a8e84966db09336cc96e4b0f`;
- observation: native position-excluded Ant observation followed by relative
  waypoint displacement divided by 4 and clipped coordinatewise to `[-1,1]`;
- targets: one of eight compass directions and distance 1, 2, 3, or 4,
  uniformly sampled in open plane;
- seed `73006`, 3,000,000 timesteps, eight workers, 400-step horizon;
- PPO MLP `[256,256]`, learning rate `3e-5`, rollout 256 per worker,
  aggregate batch 512, ten epochs, gamma 0.99, GAE 0.95, clip 0.2, and
  explicit entropy coefficient 0;
- reward: 20 times distance progress, minus 0.02 times remaining distance,
  minus 0.01 per step, plus 0.10 control reward and 0.05 contact reward, plus
  0.10 healthy survival, plus 50 on reaching 0.45, and minus 25 for unhealthy
  termination before success.

## Frozen open-plane gate

Evaluate exactly 12 deterministic 4-unit waypoint episodes per heading using
seeds `1073006 + 1000 * heading_index + replicate`. Admit v6 to a separately
frozen maze-route gate only if all are true:

- all 96 records and metrics are present and finite;
- overall success rate is at least 0.90;
- every heading success rate is at least 0.75;
- unhealthy termination rate is at most 0.10; and
- median steps among successful episodes is at most 300.

Passing does not authorize a 0.5B sample, training arm, or paper result.
Failure leaves AntMaze ineligible; no v6 maze execution may then occur.
