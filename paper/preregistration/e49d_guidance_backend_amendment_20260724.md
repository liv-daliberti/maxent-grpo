# E49D structured-output backend amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

The replacement Qwen72 service passed eight simultaneous audit-schema
requests, then the combined menu/route/audit workload again terminated
xgrammar with a fatal dynamic EBNF compilation error. Thus the failure is in
this vLLM build's xgrammar dynamic-schema compiler, not a particular
mathematical schema, model generation, or GPU.

The frozen Qwen72 service now starts vLLM with its supported
`--guided-decoding-backend guidance` backend. The OpenAI JSON schemas,
prompts, seeds, temperatures, output caps, and all deterministic local
validators are unchanged. Endpoint identity now records and every E49D
consumer requires `structured_output_backend=guidance`; a stale xgrammar
endpoint therefore fails closed.

This changes only the implementation used to enforce the same structured
output language. It does not alter mathematical execution, double auditing,
maximal-clique certification, singleton fallback, policy data, runtime
validation, reward, E46 controller, cohorts, schedule, or gates.
