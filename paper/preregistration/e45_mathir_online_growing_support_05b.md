# E45-MIR: Online Growing-Support Canonical MaxEnt on Executable Algebra (0.5B)

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Question

Can E44-OGS's online canonical discovery signal extend from exact combinatorial
answers to mathematical strategies when the mathematical output is a small,
actually executed intermediate representation rather than free-form math text?

E45-MIR is a new experiment. It does not alter E44-OGS or reuse an E44 result as
if it had been preregistered for math.

## Scope and interpretation

This is a deliberately restricted synthetic linear-algebra benchmark, not a
claim about unrestricted MATH-500 reasoning. The purpose is to test the
mechanism at Qwen2.5-0.5B scale under a validator that can bind strategy
identity to execution.

The model receives a concrete linear equation and emits only a MathIR program,
such as:

`sub(b);div(a)`

The public prompt gives exactly one validated starter program for each equation
shape. Both arms receive the identical starter. The online bank still starts
empty and is never populated from the starter, the hidden reference, or a gold
answer catalogue: a strategy enters the bank only after the policy generates
it and the executable validator accepts it. Thus the primary discovery outcome
is a valid canonical path different from the public seed path.

## MathIR linear-v0 contract

Two equation families are frozen:

1. `a*x + b = c`
2. `x/a + b = c`

Each prompt supplies exact integer bindings for `a,b,c`; construction guarantees
a unique rational solution and nonzero required divisors. Train and evaluation
formal instances are disjoint.

The only model commands are `add(E)`, `sub(E)`, `mul(E)`, and `div(E)`, separated
by semicolons, with at most four commands. Expression `E` is parsed by a
handwritten bounded parser over the prompt symbols, `x`, and
`add/sub/mul/div/neg`. Numeric literals, Python, prose, arbitrary functions, and
unknown symbols are not in the language.

Each command is applied by the interpreter to both sides of the current
equation. Both resulting sides are then exact-normalized with rational symbolic
arithmetic. Multiplication or division by an expression containing `x` is
rejected; every concrete denominator and reversible multiplier/divisor must be
nonzero. Revisited normalized states are rejected. A program is valid only if
execution ends with `x` structurally isolated and the exact executed value
equals the unique solution computed from the initial formal equation.

Whitespace, a terminal semicolon, commutative/associative ordering, and an
all-program code fence are formatting aliases. They do not create new
strategies.

The canonical key is the ordered sequence of exact-normalized equation states
created by that same successful interpreter execution. There is no separate
model-written derivation or answer string from which a key can be fabricated.
`validated_modebench_outcome_key(y, reference)` is the single admission
boundary used by the online bank.

## Frozen data

- Root: `var/data/mathir_algebra_v0_probe`
- Train: 384 rows in `train/train`
- Evaluation: 128 held-out rows in `eval/multi_answer`
- Generator seed: 4500
- Train row digest:
  `4725b97a6db30148989331a1b6735fa9a71e6381faf532260bce1ad74bd6aa09`
- Evaluation row digest:
  `24e73de2feed8aff73a67d9264621ac18254eb2fc76a8157f392d27297ccc11f`

The grammar has bounded programs but an intentionally non-enumerated discovered
support. Certified strategy lists are validator tests, not an exhaustive answer
set. E45 therefore reports observed distinct valid strategies and does not
report normalized total-mode coverage; an unknown denominator must remain
unknown.

## Bootstrap feasibility gate

Artifact: `var/artifacts/e45_mathir_bootstrap_probe_v1.json`.

The frozen base model was sampled at the training distribution
(`temperature=0.5`, `top_p=0.9`) for 16 samples on four held-out prompts per
family:

- 40 / 128 samples valid (31.25%)
- 8 / 8 prompt groups contained at least one valid sample
- both families had 100% group-level reward availability
- 0 valid non-seed strategies

The experiment proceeds because task reward exists before RL while the proposed
discovery target is absent before RL.

## Matched arms

For each seed `43,44,45`:

- `grpo`: ordinary Dr.GRPO.
- `online_canonical_maxent`: identical Dr.GRPO plus the E44-OGS online
  canonical-bank advantage.

The treatment freezes:

- entropy coefficient `alpha=0.10`
- first-discovery coefficient `beta=0.50`
- pseudocount `1.0`
- surprisal clip `5.0`
- group-snapshot scoring followed by one bank update
- duplicate first discoveries split within the group
- canonical-bank advantage added after task Dr.GRPO centering
- bank state included in resume checkpoints

Alpha is fixed. E45 does not use Haarnoja/SAC temperature adaptation.

## Shared training contract

- Model:
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`
- One A100 GPU per job on `node302`
- `G=16`
- learning rate `2e-7`
- ten complete 384-prompt passes
- rollout temperature `0.5`, top-p `0.9`
- maximum generated length 64 tokens
- no KL penalty
- no token entropy, semantic entropy, xDr, DIAYN, or canonical finite-action arm
- identical prompts, data order, seeds, optimizer, rollout budget, and
  evaluation requests between paired arms

The public starter is part of the matched prompt, not a warm-up phase or SFT
phase. Training runs normally from the frozen base checkpoint.

## Evaluation and estimands

At step zero and every 96 processed prompts, evaluate the same held-out set
with:

- greedy valid rate
- sampled mean correctness at 16
- sampled any-correct at 16
- sampled distinct valid canonical strategies at 16
- sampled distinct non-seed valid strategies at 16
- sampled any-non-seed-valid at 16
- four deterministic repeated draws beginning at seed 450100

Primary estimand: seed-paired treatment-minus-control change in
`sampled_distinct_nonseed_correct_at_16`.

Supporting discovery estimands are any-non-seed-valid@16, total distinct
valid@16, cumulative newly verified outcomes, and bank support per prompt.
Quality guardrails are greedy valid rate and sampled mean correctness@16.

The directional success pattern is a positive seed-mean primary delta with at
least two of three seed-paired deltas positive, without a material collapse in
sampled correctness. All individual seeds and null/negative results remain
reportable.

## Launch and recovery

The six jobs are submitted as one held cohort, audited for exact arm settings,
one-A100 placement, frozen source/data/protocol identity, and then released
together. A partial held cohort is cancelled. Resume is allowed only from that
job's own matched source snapshot and checkpointed bank state.
