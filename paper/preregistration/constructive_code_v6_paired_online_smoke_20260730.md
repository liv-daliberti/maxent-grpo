# ConstructiveCode v6 paired online-training smoke

**Status: FROZEN AFTER THE V6 EXECUTABLE GATE PASSED, WHILE CODER VIABILITY JOB 30203245 WAS STILL PENDING, AND BEFORE ITS OUTCOME OR ANY ONLINE TRAINING — 2026-07-30**

## Conditional authorization

This development-only pair launches only if the v6 executable gate admits all ten tasks with 960/960 selected-suite replay records and no hard violation, and the frozen Qwen2.5-Coder-0.5B viability receipt reports `status=pass`, `decision=eligible_for_paired_online_training_smoke`, 192/192 terminal worker records, at least one prefix-success task, at least one multimode task, and zero hard violations. A failed or incomplete antecedent stops ConstructiveCode before online training. These jobs are not paper seeds and their weights are discarded.

## Frozen pair

- arms: compute-matched plain Dr.GRPO and `verified_first_global_replay_canonical`;
- shared initial checkpoint: local Qwen2.5-Coder-0.5B-Instruct revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, loaded independently by both arms;
- common development seed: 78101;
- training tasks, in order: 359B, 988A, and 1399D; evaluation tasks are never loaded;
- schedule: one pass, three optimizer updates per arm, 16 independently sampled complete programs per update;
- prompt: the exact v6 viability system message and unchanged public statement only;
- sampling: temperature 1, top-p 1, no top-k truncation, maximum 1,024 generated tokens, with deterministic task/update request namespaces; and
- execution: strip only an exact complete bare/Python Markdown fence, then execute each candidate exactly once in the v6 hash-pinned Python 3.10 networkless Landlock/seccomp worker against the gate-selected official checker suite.

Every update executes exactly 16 new candidate programs per arm. Ordinary syntax, runtime, and wrong-answer outcomes are verified negatives. A timeout, isolation/output violation, checker-wrapper disagreement, missing identity, missing terminal record, or nonfinite latency fails the job. No candidate receives repair, continuation, retry, checker text, test input, or verifier feedback.

## Frozen objective and compute traversal

Optimizer is AdamW at `2e-7`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay zero, clip epsilon 0.2, and gradient-norm cap 1. Each prompt group uses 16-rollout Dr.GRPO task centering. Treatment adds the E58 verified-only mechanisms outside task centering: success-conditioned signed semantic Shannon coefficient 0.10, online canonical novelty beta 0.50, verified replay mass coefficient 0.10, known-mode balance coefficient 0.10, replay capacity 16, one persistent-hash global round-robin prompt per update, and 64-observation unprojected warmups. Canonical keys are only the official-validator-emitted behavior tuples; program text, AST, formatting, emitted hash, compiler trace, and failure type are ineligible.

Control performs the same canonicalization, passive bank updates, replay selection, detached score construction, two replay score passes, and backward traversal while applying exact-zero semantic, novelty, mass, and balance derivatives. Treatment applies the registered derivatives when eligible. On-policy teacher forcing is padded to a fixed 16 × 1,024 response-token rectangle. Replay is padded to a fixed 16-row rectangle and scored twice even when the discovered bank is empty or smaller than capacity; public padding rows cannot execute, enter a bank, or affect a gradient. Thus candidate request count, official-checker execution count, response-token ceiling, policy score slots, replay score slots, optimizer steps, and backward traversal are identical across arms.

## Information boundary and pass criterion

No reference program, certified witness, canonical key, checker source, private test, reward, suite outcome, gate statistic, development result, evaluation row, or future sample enters a prompt. Only terminal checker acceptance and validator-emitted behavior keys update reward and the online verified bank after a candidate has completed.

The independent audit requires both arms to start from one byte-identical checkpoint; exactly three ordered updates, 48 fresh model requests, and 48 terminal candidate executions per arm; at least one verified candidate and one verifier-distinct multimode training task per arm; zero hard worker violations; finite losses, log probabilities, gradients, coefficients, and latencies; exact fixed policy/replay traversal; exact-zero applied control exploration/replay derivatives with raw compute telemetry; applied treatment derivatives whenever eligible; and a second networkless official-checker replay of every stored emitted program with identical acceptance and canonical key.

A pass authorizes only a separately frozen ten-cell, 12-pass ConstructiveCode Stage-B protocol. No task, threshold, prompt, model, suite, seed, token budget, worker, objective, or sample count may change after this outcome.
