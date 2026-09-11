# AntMaze v12 Qwen2.5-0.5B development viability

**Status: FROZEN AFTER THE V12-R1 THREE-NODE PASS AND BEFORE ANY V12 MODEL COMPLETION — 2026-07-30**

This development-only gate is authorized by job 30200962: 216/216 exact
real-simulator route validations on three distinct nodes. It uses the exact
four-row `dev/multi_answer` split from the admitted v12 slate, the pinned
Qwen2.5-0.5B-Instruct snapshot, temperature 1.0, top-p 1.0, seed 107312, 64
completions per prompt, and the first 16 as the training-group viability
prefix. The response budget is 64 tokens, context 1024, and batch size 4.

The prompt uses assistant prefill `\boxed{` and the already-audited v10 public
interface reminder, because v12 changes the controller targeting rule rather
than the language interface. It reveals only compass orientation, wall
semantics, one adjacent-cell target per token, allowed tokens, registered
length range, and boxed output. No evaluation row, admitted witness, route
catalogue, controller trajectory, answer key, or canonical route identity is
placed in context.

The gate passes only if at least two of four development prompts have a
verifier-positive completion among their first 16 samples and at least one
prompt exposes both canonical route keys among all 64 samples. Every positive
is executed by the exact v12 source/controller/topology verifier. A pass
authorizes one prospectively frozen paired MaxEnt-versus-Dr.GRPO training
smoke; a failure stops AntMaze before language-model training. No prompt, seed,
sample count, route, map, threshold, or controller is changed after outcome.
