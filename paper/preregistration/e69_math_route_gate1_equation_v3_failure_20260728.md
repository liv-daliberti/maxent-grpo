# E69 MATH route Gate 1 equation-v3 failure and abstention record

Date recorded: 2026-07-28

The frozen natural-derivation equation-v3 gate completed on Slurm job
`30159313` (`COMPLETED`, exit `0:0`, node302, elapsed 00:03:00). The automatic
gate failed:

- task-correct samples: 263 / 1,024;
- trace-parse/execute samples: 69 / 1,024;
- accepted route samples: 37 / 1,024;
- route coverage among task-correct samples: 37 / 263 = 0.140684;
- accepted distinct signatures: 14;
- prompts with at least two accepted signatures: 2;
- signatures recurring across disjoint prompts: 1;
- formatting-unstable accepted rows: 0;
- deterministic manual-review queue: 37 / 50.

The coverage threshold was 0.80, the multi-route threshold was eight prompts,
and the cross-prompt recurrence threshold was three signatures. Because fewer
than 50 routes were accepted, the mandatory manual gate cannot be populated
and is not reached. This is a hard Gate 1 failure, not a near pass.

Together, the JSON-v1, RPN-v2, and equation-v3 attempts show that this 0.5B
policy does not expose sufficiently broad, independently executable
free-form-math routes. Equation-v3 improved task correctness by removing the
prompt-visible trace request, but safe route extraction covered only a small
subset of correct natural derivations. Extending the extractor to interpret
unverified prose, geometry conventions, counting arguments, or arbitrary
symbolic transformations would weaken the registered admission boundary and is
not authorized.

## Registered abstention selected for Gate 2

The parent protocol's free-form-MATH abstention path is selected:

- MATH12K training and route-development evaluation use the ordinary
  `qwen_math` task prompt and established `math_verify` answer verifier;
- free-form MATH receives ordinary task reward and the existing
  prompt-local `math_verified_answer` endpoint identity;
- no MATH response receives route novelty, route proposals, or cross-prompt
  route replay;
- E66, E68, and E69 therefore reduce to the same endpoint-only replay treatment
  on free-form MATH, while Dr.GRPO remains the matched no-replay control;
- fixed proposal-shaped sampling-only requests are still charged and discarded
  in every arm so the Gate 2 compute contract remains matched;
- Graph, Countdown, Python, and MathIR retain their exact executable route
  identities and the full E69 route successor.

MATH route-development task quality remains a Gate 2 criterion. MATH-500
predictions and scores remain sealed until the algorithm, stopping rule,
checkpoint choice, and confirmatory analysis are frozen.

Artifact identities:

- input identity:
  `b742147aeeaad7ce2dd1cb8e9a1c642dd5dd090defb5df6ebf0caaa92130744e`;
- source commit: `cb0ae2b31349920ea8cd47731a6d25be84170e9d`;
- `automatic_summary.json`:
  `54c58596c57564c37a027489ea6171bfd9e7ff6523f8cec20b7984a244543692`;
- `sample_archive.jsonl`:
  `d8d2ed2dc0f1b50c6dd36f1a1587f008963c5852f3af559fff1c61710dba616a`;
- `manual_review_queue.jsonl`:
  `ccf81846082844ea0f8ab4912af624eb9eb7e4e35076a0b15422ad4566122266`;
- Slurm stdout:
  `aa93704ca7b389871321730968585929736acc8afe2bbb11540141e021c7209e`;
- Slurm stderr:
  `adcc64ae2641c270de708c12afcf0a5c68b880f1e6accc3829ef468fa6c02a24`.
