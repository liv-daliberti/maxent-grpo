# E39 MATH12K-384 free-form final-answer outcome entropy

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Question and interpretation

E39 adds a third domain to the E37/E38 head-to-head. It asks whether the
catalogue-free final-answer outcome objectives used on Countdown and graph
coloring transfer to ordinary free-form mathematics:

1. matched free-form Dr.GRPO;
2. semantic collision entropy, the E37 duplicate-outcome penalty; and
3. predictive semantic Shannon entropy, the E38 bounded prompt-local
   surprise objective.

This is deliberately a transfer test of the existing mechanisms, not a new
math-specific diversity algorithm. No latent instruction, list of valid
answers, strategy label, answer-option codec, correct-answer collapse, or
reasoning-path classifier is added.

MATH is a single-answer domain. Consequently, E39 cannot establish diversity
over distinct correct solution strategies. Its intrinsic rewards explore
normalized final-answer hypotheses, which are mostly errors away from the one
correct equivalence class. Moreover, the catalogue-free key normalizer is not
a complete symbolic quotient: two algebraically equivalent correct renderings
can retain different normalized strings. The figure and monitor must therefore
label this row as a **single-answer stress test**. Pass@1, mean@8, and pass@8
are the quality endpoints. `distinct_correct@8` is only the number of distinct
normalized correct representations; mode coverage is not interpreted or
plotted for this row.

Keeping these limitations visible is essential. Changing the intrinsic update
only for math—for example by masking final-answer tokens or collapsing all
correct responses—would answer a different question and would no longer be a
three-domain test of the E37/E38 mechanisms.

## Frozen data

- Training parent: the locally pinned SEED-GRPO `math_12k` DatasetDict at
  commit `325cb1a20bb60f8efd4cdc77a1565491c29fd289`; the data were introduced
  at `ffa64bfdeaeda20029e831a749714f68079d7f9c`.
- Parent training Arrow SHA-256:
  `125db2efb27057f37b383d44110f3b7d49a1f55636b01f737ac5fd8cc27cf829`.
- Training slice: source rows 0 through 383, in source order. The materialized
  slice has exactly 384 unique, nonblank problems, contains all seven subjects
  and all five recorded difficulty levels, and has no exact
  whitespace-normalized problem overlap with evaluation.
- Evaluation: all 500 rows of the byte-exact held-out MATH-500 artifact already
  audited by `ops/math500/import_oat_math.py`; evaluation Arrow SHA-256:
  `d383d13c807e2904d0db6f8d98496a0574c0a1d51f40331b920c849cf226ef5a`.
- Materialized root:
  `var/data/math12k_384_math500`, with `train/train` and `eval/math`.
- `ops/math500/materialize_e39_math12k_384.py` verifies the parent hashes, commits,
  row order, schema, tokenizer admission, row counts, and leakage boundary and
  writes an atomic identity manifest. All 384 rendered `qwen_math` prompts must
  fit the 1,024-token prompt bound.

MATH-500 is evaluation-only. Neither its answers nor any policy outcome may be
used to choose the slice, coefficient, cadence, seed, stopping point, or
mechanism.

## Matched training contract

- Model: Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Prompt: ordinary neutral `qwen_math`; no latent or diversity instruction.
- Input/output columns: `problem` / `answer`.
- Full `math_verify` correctness grader.
- Training seeds: `43, 44, 45`.
- Group size: 16.
- Ten complete passes over the same 384 training prompts.
- One PPO epoch, learning rate `2e-7`, `beta=0`, maximum gradient norm 1,
  rollout temperature 1, and `top_p=1`.
- Maximum prompt, train response, and evaluation response lengths: 1,024
  tokens; model context bound: 2,048.
- One GPU per run, learner microbatch one, vLLM colocated with the learner.

The three arms differ only as follows:

| Arm | Frozen intrinsic rule |
|---|---|
| `grpo` | none |
| `outcome_collision` | E37 duplicate normalized-answer penalty, coefficient `0.10` |
| `semantic_shannon` | E38 predictive normalized-answer surprise, coefficient `0.10`, surprise clip `5.0`, pseudocount `1.0` |

Both treatments retain the shared invalid outcome for parse failures. The
predictive Shannon tracker is prompt-local, scores a group before updating its
history, and is checkpointed and restored exactly.

## Full MATH-500 evaluation and compute boundary

Evaluation uses all 500 MATH-500 prompts, not a monitoring subset. Greedy
pass@1 and one deterministic K=8 sampled draw are evaluated at initialization
and after passes 2, 4, 6, 8, and 10. The sampled draw seed is `390100`,
temperature is 1, and greedy temperature is 0.

There are exactly six evaluations per run. Because the final update is itself
the scheduled pass-10 boundary, the subsequent terminal model-export call must
detect that its policy `global_step` was already evaluated and skip a redundant
seventh evaluation of unchanged weights.

The two-pass interval is an explicit frozen sparse-cadence exception to the
repository's quarter-pass default. At one K=8 draw, each evaluation produces
approximately 5,000 completions (ordinary greedy, coverage-evaluator greedy,
and eight sampled responses per prompt), or approximately 30,000 evaluation
completions per run. The unmodified quarter-pass/four-draw contract would
produce roughly 697,000 evaluation completions per run and would make
evaluation exceed training by more than an order of magnitude.

Training saves an optimizer-resumable checkpoint once per 384-prompt pass,
retains the two newest checkpoints, and supports watchdog requeue. Evaluation
is sparse; recoverability is not.

## Cohort integrity and release gate

The analytical prefix is
`mte39_math12k_384_semantic_entropy_05b_v1`. The launcher must:

1. materialize and audit the exact frozen data;
2. freeze source and execution snapshots and record their hashes;
3. submit all nine jobs held;
4. verify three arms times three seeds, all algorithm, data, length, evaluation,
   recovery, and placement fields, and the protocol/manifest identities;
5. cancel and quarantine any incomplete held cohort; and
6. release only the complete audited nine-job cohort.

After initialization evaluations land, all plotted scalar metrics and raw
K=8 prompt outcomes must agree exactly across the three arms within each
training seed. Any step-zero mismatch fails the experiment and blocks use of
post-update outcomes.

## Reporting

The maintained E37/E38 figure gains a third row,
`MATH-500 (train MATH12K-384)`. It shows all three seeds, all three methods,
the full ten-pass x-axis, and the same quality and mechanism telemetry as the
other rows. The math coverage panel is marked not applicable. The caption
states that distinct correct answers are normalized string representations
and that the intrinsic objective explores final-answer outcomes/errors rather
than proof strategies.

No coefficient, key definition, data row, evaluation draw, or endpoint may be
changed after any E39 post-update outcome is observed. A later experiment may
test reasoning-path representations or answer-token masking, but it must use a
new protocol and cannot be relabeled as E39.
