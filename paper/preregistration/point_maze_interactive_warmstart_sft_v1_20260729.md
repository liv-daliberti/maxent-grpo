# PointMaze interactive shared warm start and development gate v1

**Status: FROZEN BEFORE THE FIRST MODEL UPDATE — 2026-07-29**

## Prerequisite

Train-only materialization v1 passed with 8 maps, 16 independently verified
route episodes, and 644 public-state-to-action examples. Its examples SHA-256
is `fc53db6ad16debf4ffeb26b2479212db6709b78155b087ea578ab184fa5ac356`.
No dev/eval row was loaded and no model was sampled during materialization.

## Frozen shared SFT

- base: Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- update all model parameters in BF16;
- seed `75201`, three epochs, batch size 4, gradient accumulation 7;
- exactly 69 AdamW optimizer steps;
- learning rate `2e-5`, linear warmup 7 steps then linear decay;
- weight decay `0.01`, gradient norm cap `1.0`;
- context cap 1536 tokens; and
- cross-entropy renormalized over the same nine single-token option labels used
  by the online policy. Environment/context tokens receive no loss.

The SFT process may load only the frozen train-only examples. It receives no
terminal reward, verifier call, development row, evaluation row, MaxEnt
outcome, or Dr.GRPO outcome.

## Frozen post-SFT development gate

After saving the checkpoint, run the unchanged closed-loop finite-action gate
on exactly `point_maze_modebench_v1/dev/multi_answer`:

- four development maps and no evaluation map;
- 64 rollouts per map, first 16 as the prefix;
- temperature/top-p `1.0`, seed `75103`;
- at most 96 one-token decisions, each followed by five simulator steps;
- at least two prompts with a prefix verified route; and
- at least one prompt with two verified canonical route identities among all
  64 rollouts.

The action mask exposes only the fixed nine-token public force alphabet. No
planner route, certified program, intermediate verifier result, or training
target is present online.

Passing authorizes only a separately frozen one-seed matched
verified-MaxEnt-versus-Dr.GRPO smoke. Both arms must start from the exact same
checkpoint. Failure stops PointMaze before online training.
