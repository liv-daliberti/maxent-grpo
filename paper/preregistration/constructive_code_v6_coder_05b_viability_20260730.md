# ConstructiveCode v6 Qwen2.5-Coder-0.5B development viability

Date frozen: 2026-07-30, after the v6 960/960 executable gate passed and
before any ConstructiveCode language-model sample.

This probe launches only when `constructive_code_v6_gate_audit.json` is an
exact pass for all ten tasks and 960 selected-suite records, with zero gate or
checker-equivalence violation. It binds the new v6 gate and its complete failed
v5 antecedent. A failure stops ConstructiveCode v6 before online training. A
pass authorizes only the separately frozen paired online-training smoke; this
probe is not a paper seed.

## Frozen information boundary and sampling

- model: Qwen2.5-Coder-0.5B-Instruct at revision
  `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`;
- development tasks, in order: 359B, 988A, and 1399D;
- evaluation tasks 361B, 1294C, and 149C are never loaded;
- 64 independent candidates per development task, 192 requests total;
- seed formula `77101 + 10000 * row_index + sample_index`;
- temperature 1, top-p 1, no top-k truncation, at most 1,024 response tokens,
  and an 8,192-token model context;
- exact Coder system message and unchanged public problem statement only;
- no reference program, known-label replay, checker source, private test,
  canonical key, suite outcome, reward, split outcome, or evaluation row enters
  model context;
- strip only an exact complete bare/Python Markdown fence, then execute each
  candidate once in the hash-pinned networkless Python 3.10 worker against its
  frozen v6 official checker suite.

## Frozen decision

The receipt passes only when all 192 requests produce terminal worker records,
no execution/checker/isolation/output/timeout hard violation occurs, at least
one development task has an accepted candidate among its first 16 samples, and
at least one development task has two or more distinct accepted canonical
witnesses among all 64 samples. The receipt records every request and execution
identity, accepted behavior key, latency, checker build, source/gate/execution
hash, and proof that no evaluation row was loaded.
