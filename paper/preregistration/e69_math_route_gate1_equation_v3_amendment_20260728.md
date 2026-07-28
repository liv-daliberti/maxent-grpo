# E69 MATH route Gate 1 equation-v3 amendment

Date frozen: 2026-07-28

This amendment is prospective with respect to the equation-v3 sample archive.
It follows the sealed failures of JSON-v1 and RPN-v2. The already opened
JSON-v1 archive was used as mechanism-development data for the deterministic
extractor and is not reused as the equation-v3 gate archive. No MATH-500
prediction, answer, or score was inspected.

## Motivation and permitted change

The explicit JSON and RPN interfaces competed with task correctness at
Qwen2.5-0.5B scale. Equation-v3 therefore uses the ordinary frozen
`qwen_math` prompt and extracts only arithmetic equation chains already present
in the natural derivation. It adds no prompt-visible trace request.

The admission boundary is:

- only content inside LaTeX math spans is parsed; prose is ignored;
- route blocks, if any, are ignored by the equation extractor;
- a small custom parser accepts exact integers, decimals, fractions, variables,
  parentheses, and a closed arithmetic-function allowlist;
- every numeric leaf must first occur in the problem or have been established
  by an earlier independently checked equality;
- square and cube exponents are structural operation syntax, matching the
  existing zero-argument `square` and `cube` operators;
- an equality can authorize an unchecked literal only when another side
  independently evaluates to exactly that value;
- a bounded one-variable equation can authorize only exact real solutions;
- the boxed terminal must be reachable through at least one checked operation;
- route identity is the value-free operation/dependency tree;
- no Python `eval`, untrusted code execution, prose interpretation, or model
  judge is used.

Correct responses whose reasoning requires an unimplemented semantic fact
(for example, interpreting an intercept, extracting a polynomial coefficient,
or executing a coordinate transform) fail closed. The development estimate on
the previously opened JSON-v1 archive is recorded only as a design diagnostic,
not as a gate result.

## Frozen equation-v3 coverage run

The sealed 128 MATH12K route-development prompts, eight samples per prompt,
temperature 1, top-p 1, 1,024-token response limit, and seed `690101` remain
unchanged. The prompt changes to the ordinary task-first `qwen_math` template,
whose exact population is frozen by
`var/data/math12k_384_route_dev128_v1/EQUATION_V3_PROMPT_MANIFEST.json`.

The output root is
`var/artifacts/e69_math_route_gate1_base_equation_v3`. The input identity covers
this amendment, the parent protocol, both route modules, the task verifier,
template, sampler, sealed materialization manifest, and equation-v3 prompt
manifest.

The parent automatic thresholds remain unchanged. Formatting stability is
tested by changing only whitespace around equality signs in LaTeX math spans.
If the automatic gate passes, the deterministic 50-row queue must still receive
zero false-admission judgments before MATH route novelty is enabled.

Failure selects the parent protocol's preregistered abstention path: free-form
MATH keeps ordinary task reward and endpoint identity but receives no route
novelty or cross-prompt route replay. Failure does not authorize MATH-500
inspection.
