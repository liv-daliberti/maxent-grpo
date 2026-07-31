# ConstructiveCode v6 Stage B: clean 0.5B verified MaxEnt versus compute-matched Dr.GRPO

**Status: FROZEN AFTER THE V6 EXECUTABLE GATE PASSED, WHILE CODER VIABILITY JOB 30203245 WAS STILL PENDING, AND BEFORE ITS OUTCOME, THE PAIRED-SMOKE OUTCOME, OR STAGE-B SUBMISSION — 2026-07-30**

## Conditional authorization

This protocol may launch only if the exact v6 executable gate and Qwen2.5-Coder-0.5B viability gate pass, followed by a passing immutable paired online smoke with decision `eligible_for_ten_constructive_code_stage_b_jobs`. A failed or incomplete antecedent stops ConstructiveCode; no final cell is submitted. Development weights and seed 78101 are never reused.

## Frozen Cartesian product

- arms: compute-matched plain Dr.GRPO and `verified_first_global_replay_canonical`;
- seeds: 43, 44, 45, 46, and 47;
- shared initial checkpoint: byte-identical local Qwen2.5-Coder-0.5B-Instruct revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`;
- training tasks, in order: 327B (`ordered_sequence`), 659C (`unordered_set`), 1283C (`assignment`), and 1102B (`unordered_partition`);
- evaluation tasks, in order: 361B (`ordered_sequence`), 1294C (`unordered_set`), and 149C (`unordered_partition`);
- schedule: exactly 12 complete passes, 48 optimizer updates, 16 fresh complete-program rollouts per update, maximum 1,024 generated response tokens; and
- no resume, checkpoint selection, seed replacement, task substitution, extra pass, repair, continuation, retry, or result-dependent extension.

## Prompt, execution, objective, and compute match

The prompt and sampling surface are identical to the v6 paired smoke: the exact Coder system message, unchanged public statement, temperature 1, top-p 1, and no top-k truncation. No reference program, certified witness, behavior key, checker source, private test, reward, suite result, or evaluation outcome enters context. Strip only an exact complete bare/Python Markdown fence, then execute every candidate exactly once in the hash-pinned Python 3.10 networkless Landlock/seccomp worker against the gate-selected official suite.

Optimizer and online objectives are exactly those qualified by the paired smoke: AdamW at `2e-7`; 16-rollout Dr.GRPO task centering; success-conditioned signed semantic Shannon coefficient 0.10; online canonical novelty beta 0.50; verified replay mass and known-mode balance coefficients 0.10; replay capacity 16; one persistent-hash global round-robin prompt per update; and unprojected 64-observation warmups. Only official-validator-accepted behavior tuples are canonical keys.

Treatment applies the registered exploration and replay derivatives. Control executes the same sampling, candidate programs, official-checker calls, canonicalization, passive banks, replay selection, detached score construction, two replay score passes, and backward traversal with exact-zero applied semantic, novelty, mass, and balance derivatives. Fixed 16 × 1,024 on-policy rectangles and 16-row replay rectangles make response-token ceilings and score traversal identical across arms; padding cannot execute, enter a bank, or affect gradients.

## Frozen evaluation

The three evaluation tasks are loaded only inside final paper jobs. Evaluation occurs before training and after every update, giving all 49 quarter-pass coordinates from pass 0 through pass 12. At each coordinate, each task receives one greedy program plus four deterministic temperature-one K=8 replicates: 99 programs total. Every program executes once without feedback in the same frozen worker. Evaluation updates no optimizer, tracker, bank, prompt order, request namespace, or training schedule.

Report greedy success, mean@8, pass@8, and the number of validator-distinct accepted behavior keys@8 overall and by witness family. Registered anchors are passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12; terminal pass 12 and trapezoidal AUC over those anchors are primary.

## Fail-closed terminal audit

The independent audit requires the exact ten-cell manifest; 48 ordered updates and 49 evaluations per cell; 768 training requests and executions per cell; four complete K=8 evaluation draws per task and coordinate; zero hard worker violation; finite losses, log probabilities, gradients, coefficients, and latencies; exact fixed policy/replay traversal; matched execution and response-token ceilings across arms; exact-zero applied control exploration/replay derivatives; applied treatment derivatives whenever eligible; and an identity-bound second official-checker replay of every stored training candidate with identical acceptance and behavior key.

It also binds the v6 gate, Coder viability, paired smoke, source, operations, model, data, runtime image, sandbox launcher, checker builds, protocol, manifest, scheduler, metrics, and candidate ledgers. Missing or failed cells remain visible and prevent ConstructiveCode from becoming a terminal paper row.
