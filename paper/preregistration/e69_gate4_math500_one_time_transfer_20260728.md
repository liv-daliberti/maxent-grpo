# E69 Gate 4 one-time held-out MATH-500 transfer

Date frozen: 2026-07-28, after Gate 2 launch and before any terminal Gate 2 or
Gate 3 outcome was available.

This document instantiates Gate 4 of
`e69_verified_route_successor_protocol_20260728.md`. It freezes the unsealing,
evaluation, and final classification before any E69 MATH-500 checkpoint
prediction or score exists.

## Unsealing condition

Gate 4 may be submitted only after:

1. the frozen Gate 2 audit passes with zero integrity violations;
2. all 30 unique Gate 3 physical runs are terminal with zero integrity
   violations;
3. the Gate 3 identity binds the exact seed-43 reuse and new seed-44/45 jobs;
4. the six terminal free-form-MATH checkpoints pass a finite-tensor checkpoint
   audit; and
5. this protocol, the evaluation implementation, final analyzer, and panel
   code are committed.

Gate 4 proceeds after a clean, complete Gate 3 regardless of whether Gate 3's
internal efficacy classification is positive. This prevents selective
unsealing. A failed execution-integrity audit does not unseal MATH-500; it
permits only a same-identity infrastructure repair.

## Frozen checkpoints and data

The evaluated models are exactly the pass-6 checkpoints after training global
step 2,304 (the existing saver labels this post-update export
`saved_models/step_02305`) for seeds 43, 44, and 45 from:

- compute-matched Dr.GRPO; and
- the endpoint-only replay arm that represents E69 under the registered
  free-form-MATH route abstention.

No checkpoint averaging, best-pass selection, duplicate attempt, seed
replacement, or further optimization is allowed.

Evaluation uses exactly all 500 rows of the established
`math12k_384_math500/eval/math` artifact. Its ordered-row SHA-256 remains
`1576fd11df21dc705a7c85000f232031212225cd9c00520faa26f6bdfc751166`,
and its Arrow file SHA-256 remains
`2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7`.
The recorded normalized train/evaluation problem overlap must be zero.

## Frozen requests and verifier

Each checkpoint receives two neutral evaluation requests over the same ordered
500 prompts:

- greedy: one response, temperature 0, seed 0;
- sampled: eight responses, temperature 1, top-p 1, seed `690401`.

Both use the exact `qwen_math` prompt, a 1,024-token response limit, a
2,048-token model limit, no route request, and the ordinary full
`math_verify` verifier used during E69 MATH training. Raw prompt indices,
responses, verifier rewards, verifier timeout/error diagnostics, finish
reasons, and token counts are preserved in one immutable result per
checkpoint. A result file must be absent before submission and is written by
atomic rename.

Six evaluation jobs are submitted as one held cohort, audited for exact
checkpoint/data/code/request identity, and then released together. Scheduler
preemption may resume an incomplete same-identity output, but a completed
checkpoint result is never regenerated.

## Frozen analysis

For each arm and seed report:

- greedy verified accuracy;
- sampled mean correctness@8;
- sampled pass@8;
- response-token length and verifier timeout/error rates; and
- all 500 paired per-prompt contributions.

Report successor-minus-control deltas for every seed and the three-seed mean.
For each primary metric, compute a 10,000-replicate crossed paired
seed-and-prompt bootstrap with seed `690402`, resampling the three paired seeds
and matched MATH-500 prompt indices exactly as frozen for Gate 3. The interval
is descriptive; every raw paired seed delta remains visible.

The final paper panel has five areas: Graph, Countdown, Python factors, MathIR,
and **held-out MATH-500 transfer**. MATH12K route-dev remains a development
diagnostic and is not relabeled as held-out transfer.

The final primary result is **successful exploration** only if:

- terminal three-seed mean greedy and pass@8 successor-minus-control deltas
  are at least `-0.02` in each of the four executable domains;
- held-out MATH-500 greedy, mean@8, and pass@8 deltas are each at least
  `-0.02`;
- at least three executable domains satisfy Gate 3's frozen positive-support
  condition;
- Gate 3's frozen cross-prompt mechanism-presence condition passes; and
- every Gate 2, Gate 3, checkpoint, Gate 4, verifier, and compute audit is
  clean.

If support improves but any task-quality condition fails, classify the result
as a mechanism result. Otherwise classify it as null or negative as indicated
by the frozen component checks. Report every outcome without tuning,
checkpoint reselection, or rerunning MATH-500.
