# AntMaze v10 Qwen2.5-0.5B development viability

**Status: FROZEN DURING V10 CONTROLLER TRAINING, BEFORE ITS OUTCOME OR ANY V10 MODEL COMPLETION — 2026-07-29**

This gate is conditional on passes from the frozen v10 controller, 11x11
route admission, and three-node exact-slate replay. It may not run if any
antecedent is missing or failed.

## Immutable sampling contract

- exact four-row `dev/multi_answer` split from the admitted
  `var/data/ant_maze_modebench_v10` slate;
- pinned Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- temperature 1.0, top-p 1.0, seed `107310`;
- 64 completions per prompt, with the first 16 fixed as the training-group
  viability prefix;
- response budget 64 tokens, context 1024 tokens, batch size 4;
- assistant prefill `\boxed{` and the fixed `ant_maze_v10` grammar reminder;
- exact route-job source snapshot and hash-bound persistent MuJoCo worker; and
- no evaluation row, certified witness program, route catalogue, controller
  trajectory, or answer key in model context.

The grammar reminder states only public interface facts already present in the
task: compass orientation, wall semantics, one adjacent-cell target per token,
allowed tokens, registered length range, and boxed output. It does not reveal
either admitted upper/lower witness.

## Decision

The gate passes only if at least two of four development prompts have a
verifier-positive completion among their first 16 samples and at least one of
four prompts exposes both canonical route keys among all 64 samples. Every
positive is executed by the exact admitted v10 controller and topology
verifier.

A pass authorizes only one prospectively frozen, paired MaxEnt-versus-Dr.GRPO
online-training smoke with a shared initialization and matched trajectory
budget. It does not authorize confirmatory seeds or a paper result. A failure
stops AntMaze v10 before language-model training; no prompt, seed, sample
count, or threshold repair is allowed.
