# ConstructiveCode v7 train-only Coder warm start

**Status: frozen after the v6 0/192 development gate and before v7 SFT or any
post-SFT model sample — 2026-07-30.**

V6 executed all 192 frozen Qwen2.5-Coder-0.5B-Instruct requests and produced
zero checker-accepted programs with no hard sandbox or checker violation.  The
failure includes both source-format errors and semantically incorrect but
executable programs.  The v6 result remains a failed gate and cannot authorize
online training.

## Train-only corpus

V7 retains the admitted v6 ten-task slate and split.  SFT may load only train
tasks 327B, 659C, 1283C, and 1102B.  For each task it starts from the 48
`known_label=correct` Python 3 submissions already replayed by the independent
v6 executable gate, sorts them by `(UTF-8 code byte length,
submission_sha256)`, and takes the first 16.  This yields exactly 64 examples,
16 from each of the four witness families.  Selection uses no model output,
reward, development result, or evaluation row.

Each example is the public statement under the exact Qwen ChatML system/user
prompt used by the viability evaluator, followed by the complete accepted
Python source as the assistant target.  Loss is restricted to assistant source
and its terminal ChatML token.  Development tasks 359B, 988A, and 1399D and
evaluation tasks 361B, 1294C, and 149C are never loaded by materialization or
SFT.

## Frozen SFT

- base: Qwen2.5-Coder-0.5B-Instruct snapshot
  `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`;
- seed 77201; four epochs over the deterministic 64-example corpus;
- seeded full-corpus shuffle per epoch, microbatch one, gradient accumulation
  eight, exactly 32 optimizer updates;
- AdamW, learning rate 1e-5, betas (0.9, 0.999), epsilon 1e-8, zero weight
  decay, gradient norm cap 1.0, BF16, maximum sequence length 4096;
- one shared resulting checkpoint for both prospective online arms.

## Frozen post-SFT development gate

Load only the unchanged three v6 development tasks.  Draw 64 programs per
task with seed base 77102, temperature 1, top-p 1, maximum 1024 new tokens,
and the same exact checker suites, sandbox, canonicalizers, prefix of 16, and
information firewall as v6.  Passing still requires at least one prefix-success
task and at least one task with two accepted canonical keys.  Any hard
violation or failed threshold stops before paired or final online training.
