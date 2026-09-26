# PointMaze Qwen2.5-0.5B viability v2: prompt-format repair

**Status: FROZEN BEFORE THE FIRST V2 MODEL COMPLETION — 2026-07-29**

## Reason for the repair

The frozen v1 gate (Slurm job `30184635`) completed with zero verified
completions across four development prompts. Inspection was limited to
parser/instruction behavior: sampled responses did not preserve the required
`\boxed{...}` answer envelope. No evaluation prompt, certified route
program, route catalogue, or training outcome was inspected.

V1 explicitly permits one prospective prompt-format repair. V2 changes only
the assistant response prefill: the model context ends with the literal
prefix `\boxed{`, and verification reconstructs the complete assistant
response as that fixed prefix plus the generated continuation. This is an
ordinary prompt prefill; it does not alter generated tokens, action parsing,
MuJoCo execution, route identity, maps, thresholds, or the model.

## Immutable sampling contract

- exact `dev/multi_answer` split from
  `var/data/point_maze_modebench_v1`;
- pinned Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- temperature `1.0`, top-p `1.0`;
- 64 completions per development prompt under fresh seed `75002`;
- first 16 completions per prompt are the training-group viability prefix;
- response budget 192 tokens, model context 1024 tokens; and
- the same hash-pinned external MuJoCo worker and topology-bound route keys.

## Decision

The unchanged gate passes only if:

1. at least two of four prompts have a verifier-positive completion in their
   first 16 samples; and
2. at least one prompt exposes two distinct successful route keys in its full
   64 samples.

Failure stops PointMaze before training. Passing authorizes only a matched
Gate-A training smoke, not its five-seed confirmatory cells.
