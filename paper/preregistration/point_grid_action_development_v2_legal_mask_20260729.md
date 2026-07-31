# PointMaze grid-action development pilot v2: legal action mask

**Status: FROZEN BEFORE THE FIRST V2 MODEL COMPLETION — 2026-07-29**

## Antecedent and separation

The prospectively frozen v1 grid-action pilot executed zero verified routes.
All 256 outputs obeyed cardinal syntax, but 246 hit walls and the ten
wall-legal strings ended outside the goal. The v1 receipt and its 0/4 decision
remain final. The deterministic grid controller separately executed both
routes on all four development maps (8/8).

V2 changes only the action mask. It does not change maps, MuJoCo controller,
route identity, model, sample counts, or decision thresholds.

## Legal-mask boundary

For each prompt, enumerate every path that:

- begins at S;
- uses only cardinal one-cell moves;
- never enters a wall or revisits a cell; and
- has length from the prompt-local shortest distance through shortest plus 4.

The choice set includes every such path regardless of endpoint. It is not
filtered for reaching G, and successful paths or route identities are neither
labeled nor weighted. Thus the environment masks illegal actions but does not
supply a solution. Counts fixed before sampling are 250, 8,798, 5,796, and
4,719 total choices; respectively 32, 994, 366, and 553 happen to end at G.

## Immutable development sample

- exact four-row `dev/multi_answer` PointMaze v1 split;
- pinned Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- temperature/top-p `1.0`, seed `75102`;
- 64 completions per prompt, first 16 as the prefix;
- 96-token output budget and 1024-token context;
- at least two prompts with one prefix success; and
- at least one prompt with two executed route keys in the full sample.

Passing means only a development interface signal and cannot authorize
training or a paper result.
