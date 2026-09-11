# PointMaze grid-action development pilot v1

**Status: FROZEN BEFORE THE FIRST GRID-ACTION MODEL COMPLETION — 2026-07-29**

## Scope

The frozen PointMaze v1 and v2 low-level force-pulse gates both failed with
zero verified routes. Those outcomes remain final. This separated development
pilot asks whether the same pinned 0.5B model can instead choose a discrete
cardinal route while a deterministic controller handles point-mass dynamics.

Passing is only a development interface signal. It cannot authorize training,
replace either failed gate, or enter a paper comparison.

## Frozen action and execution boundary

- One model token moves one legal cardinal grid cell: `N`, `E`, `S`, or `W`.
- The generated program must avoid walls and end at the prompt-visible goal.
- Guided decoding enforces only cardinal-token syntax and a prompt-local
  shortest-path-length to shortest-plus-four range. It is not given feasible
  paths, route fixtures, bottleneck identities, or the route catalogue.
- A hash-bound PD controller targets the center of the selected next cell. It
  may not choose a cell, repair a path, cross a wall, or change a model token.
- Success and mode identity still require execution in the pinned external
  MuJoCo worker and the existing topology-bound directed-gate extractor.

Before this freeze, controller calibration on the four development maps used
only deterministic graph fixtures and established 8/8 successful executions,
covering both directed-gate routes on every map. No model completion or
evaluation map was used in controller calibration.

## Immutable model sample

- exact four-row `dev/multi_answer` split from
  `var/data/point_maze_modebench_v1`;
- Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- temperature `1.0`, top-p `1.0`, seed `75101`;
- 64 completions per prompt and first-16 viability prefix;
- response budget 96 tokens and model context 1024 tokens;
- at least two prompts with a verified completion in their prefix; and
- at least one prompt with two distinct executed route keys in its full sample.
