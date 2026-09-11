# PointMaze Qwen2.5-0.5B development-only viability gate

**Status: FROZEN BEFORE THE FIRST MODEL COMPLETION — 2026-07-29**

## Scope

This gate tests only whether the pinned Qwen2.5-0.5B-Instruct base model can
produce verifier-positive PointMaze language-action programs often enough to
support the later online-RL comparison. It is not a training run, an
experimental arm, or a paper outcome.

The input is the exact `dev/multi_answer` split of
`var/data/point_maze_modebench_v1`: four prompts, one from each frozen map
family. The train split, certified route fixtures, route catalogue, and
evaluation split may not enter the prompt, completion context, or sampling
decision.

## Sampling contract

- model:
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`;
- prompt template: `qwen_boxed`;
- temperature `1.0`, top-p `1.0`;
- 64 completions per development prompt under sampling seed `75001`;
- the first 16 completions per prompt are the intended training-group
  viability prefix;
- response budget 192 tokens and model context 1024 tokens; and
- every completion is executed by the hash-pinned external MuJoCo worker.

The evaluator records model text only for these development prompts. A
completion is correct only when the worker reaches the goal and returns a
topology-bound route key. Textual claims of success or route identity do not
count.

## Decision

The gate passes only if all four prompts complete without an infrastructure
or identity failure and:

1. at least two of four prompts have one verifier-positive completion in their
   first 16 samples; and
2. at least one of four prompts exposes two distinct successful route keys in
   its full 64 samples.

Failure permits one prospective prompt-format repair based only on recorded
parser/instruction failure categories. That repair must be frozen before a
fresh full sample and cannot change maps, route identity, simulator settings,
or numeric thresholds. A second failure stops PointMaze before training.

Passing authorizes only the later matched Gate-A training smoke. It does not
authorize any of the 80 confirmatory jobs.
