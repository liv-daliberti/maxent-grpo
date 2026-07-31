# ConstructiveCode v8: preregistered 1.5B coder capacity retry

Status: prospective; frozen before any 1.5B sample  
Date: 2026-07-30

The original ConstructiveCode extension plan fixed a two-rung capacity ladder:
Qwen2.5-Coder-0.5B-Instruct first, followed by exactly one
Qwen2.5-Coder-1.5B-Instruct retry if the smaller coder failed. The executable
v6 gate passed 960/960 selected-suite replays. The frozen base-0.5B probe and
the later train-only 0.5B warm-start probe each executed all 192 development
requests and accepted 0/192. The warm start removed Markdown-fence failures
but did not transfer semantic correctness. This invokes the already registered
capacity rung; it does not change prompts, tasks, checkers, or pass criteria.

## Frozen model and information boundary

- model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`;
- Hugging Face commit:
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`;
- no SFT, adapter, retrieval, reference program, checker source, hidden input,
  canonical key, or previous completion is placed in model context;
- development problems only: `359_B`, `988_A`, and `1399_D`;
- evaluation problems `361_B`, `1294_C`, and `149_C` are not loaded.

## Frozen sampling and executable decision

- seed 77101, 64 samples per problem, and prefix 16;
- temperature 1.0, top-p 1.0, maximum 1,024 new tokens;
- the exact v6 prompt, selected test suites, networkless Python sandbox,
  released checker, wrapper, and semantic canonicalizer;
- pass only if at least one of three tasks has an accepted sample in its first
  16 and at least one task has two distinct accepted behavior keys among 64;
- require 192/192 terminal worker records and zero hard protocol, isolation,
  checker-equivalence, timeout, or output-bound violations.

A pass authorizes a separately labelled 1.5B paired online smoke. It does not
turn the ConstructiveCode row into a 0.5B result and it must not be pooled
silently with the seven 0.5B environments. A failure stops ConstructiveCode
before online training under the registered capacity ladder.
