# E69 MATH route Gate 1 RPN-v2 amendment

Date frozen: 2026-07-28

This amendment is prospective with respect to the RPN-v2 sample archive. It
follows the sealed failure of `math-route-v1` on job `30158805`; it does not use
or authorize any MATH-500 result.

## Motivation and single permitted change

The v1 archive contained 174 task-correct samples and zero valid route blocks.
None of the task-correct responses complied with the nested JSON trace
interface. The arithmetic executor and independent terminal-answer gate were
therefore never reached. RPN-v2 changes only the route serialization and its
prompt:

- a route is a whitespace-delimited reverse-Polish stack program;
- numeric tokens must be exact leaves copied from the problem;
- unary and binary operations execute in a closed allowlist;
- intermediate results cannot be declared;
- stack consumption defines dependencies;
- acceptance requires one terminal stack value, at least one operation, no
  unused token, ordinary task correctness, and agreement with the boxed answer;
- route identity is the executed, value-free operation tree.

The legacy JSON-v1 parser remains accepted by the validator for historical
compatibility, but the RPN-v2 prompt requests only RPN-v2.

## Frozen RPN-v2 coverage rerun

The population, model revision, 128 prompts, eight samples per prompt,
temperature 1, top-p 1, 1,024-token limit, seed `690101`, automatic thresholds,
and deterministic 50-row manual queue are unchanged from the parent protocol.
Formatting stability is tested by a whitespace-only perturbation of the RPN
block rather than JSON pretty-printing.

The output root is
`var/artifacts/e69_math_route_gate1_base_rpn_v2`. The input-identity hash must
cover the parent protocol, this amendment, the route parser, task verifier,
prompt template, sampler, and sealed materialization manifest.

Failure again returns to trace-language design or abstention; it does not
permit Gate 2 or inspection of MATH-500 scores.
