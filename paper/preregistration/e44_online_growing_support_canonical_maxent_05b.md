# E44-OGS: online growing-support canonical MaxEnt at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Identifier note

An earlier, immutable protocol used the bare E44 label for
`signal_first_semantic_balance`. Its records and files remain unchanged.
This user-directed replacement direction is identified everywhere as
**E44-OGS** and uses disjoint `e44_ogs` run prefixes. Results from the two
protocols must never be pooled or described under an ambiguous bare E44 label.

## Question

For each training problem `x`, can Dr.GRPO maintain a growing bank
`B_x^+` containing only canonical outcomes that the current policy has
actually generated and the task validator has accepted? Does combining

1. an entropy-gradient estimate on the observed distribution over `B_x^+`;
   and
2. a one-time reward for each newly discovered verified outcome

improve pass@8 and verified mode coverage without sacrificing mean@8 or
greedy pass@1?

There are no discovery and exploitation phases. Bank scoring, discovery,
admission, and policy learning happen in every ordinary rollout update.

## Why this cohort excludes unrestricted MATH

Final-answer equivalence is not a proof or strategy verifier. On an ordinary
MATH problem, a correct boxed answer does not establish that decorative
intermediate equations were executed or even used. Consequently E44-OGS does
not assign canonical-strategy novelty to unrestricted mathematical prose.

The cohort uses the two landed tasks with executable outcome validators:

- graph coloring: the emitted coloring is parsed and checked against every
  graph edge and every fixed partial color;
- Countdown: the emitted expression is parsed as an arithmetic AST, exact
  one-time use of the supplied operands is enforced, and the AST is executed
  against the requested target.

A later MATH extension requires a constrained solution language whose exact
generated AST is executed and validated. Text signatures, method labels, and
post-hoc clustering are not admissible substitutes.

## Fail-closed admission contract

For each sampled response `y`, the learner independently replays the
ModeBench validator on the exact response decoded from the training
trajectory. Only the validator-bound helper
`validated_modebench_outcome_key(y, reference)` may propose a bank key.

The helper returns a key if and only if the same parsed outcome verifies:

- graph keys are the fully materialized coloring vector;
- Countdown keys are the normalized executable AST, with operands of
  commutative nodes sorted.

The learner asserts row-by-row equivalence between actor task reward positivity
and successful validator-bound key extraction. Any disagreement aborts the
update before bank admission. Thus a string that merely claims a canonical
form, an invalid program, a formatting alias, and an outcome different from
the one rewarded cannot enter `B_x^+`.

Each group is scored against an immutable pre-group snapshot. New verified
keys are committed only after all rows have been scored, so row order cannot
change novelty. Bank state and counts are part of the optimizer-resume state;
a treatment resume without the matching bank state fails closed.

No gold catalogue of valid outcomes populates the bank. The reference is used
only by the ordinary task validator. The support grows solely from
policy-generated, validator-positive outcomes.

## Frozen objective

Let `G=16`. For an active row `i`, let `a_i` be its validator-bound canonical
key when correct, and let `n_x(a)` be historical correct occurrence counts.
Let `m_x(a)` be counts among eligible rows in the current group. The support
for scoring is the union of the pre-group bank and current verified keys.

With pseudocount `lambda=1`, row `i` is scored from the leave-one-out
predictive distribution

`q_-i(a) = (n_x(a) + m_x(a) - 1[a=a_i] + lambda) / Z_i`.

Surprisal is clipped at `S=5`. Define

`h_i = min(-log q_-i(a_i), S)`,

`H_-i = sum_a q_-i(a) min(-log q_-i(a), S)`,

`A_i_ent = alpha * (h_i - H_-i)`,

with `alpha=0.10`. Wrong, inactive, or unparseable rows receive zero.

Let `N_x` be the set of current verified keys absent from the pre-group bank,
and let `m_x(a)` be the current multiplicity of key `a`. The set-level novelty
credit is

`A_i_new = beta * 1[a_i in N_x] / m_x(a_i)`,

with `beta=0.50`. Hence every newly discovered canonical class contributes
exactly one total `beta`, regardless of duplicates or row ordering.

The actor advantage is

`A_i_actor = A_i_task-DrGRPO + A_i_ent + A_i_new`.

Both canonical terms are detached and added exactly once after ordinary task
reward centering. Task reward itself is unchanged. This placement preserves a
discovery signal when every correct row in a group is new; putting a shared
bonus through Dr.GRPO centering could erase it.

This is an on-policy empirical canonical-entropy gradient estimator, not the
closed-form finite-policy entropy used by E16. Claims must use that precise
description.

## Matched cohort

- Model: local immutable
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Tasks:
  - graph coloring: 192 training prompts, 96 neutral evaluation prompts;
  - Countdown easy3: 384 training prompts, 128 neutral evaluation prompts.
- Arms:
  - `grpo`: ordinary Dr.GRPO;
  - `online_canonical_maxent`: the frozen objective above.
- Fresh paired seeds: `43, 44, 45`.
- Group size: `16`.
- Budget: ten complete prompt-pool passes.
- Learning rate: `2e-7`.
- One PPO epoch, `beta=0`, maximum norm `1`.
- Rollout temperature `1`, top-p `1`, maximum response length `192`.
- Both arms use the identical unrestricted `qwen_boxed` action format,
  verifier, data order, optimizer, actor synchronization, checkpoint cadence,
  and evaluation requests.

Run prefixes are:

- `gce44_ogs_canonical_maxent_05b_v1`;
- `cde44_ogs_canonical_maxent_05b_v1`.

## Evaluation and interpretation

Evaluate at initialization, every quarter pass, and ten passes. Every boundary
uses deterministic greedy pass@1 plus four fixed temperature-1, K=8 draws with
seeds `440100--440103`. Retain raw responses, rewards, and canonical keys.

Primary endpoints are neutral pass@8 and verified mode coverage@8. Secondary
endpoints are distinct-correct@8, mean@8, greedy pass@1, area under the
coverage curve, and time to breadth collapse.

Mandatory treatment telemetry includes entropy estimate, entropy/novelty and
combined advantage RMS, eligible fraction, new verified outcomes per group,
bank size before and after each group, total tracked prompts/outcomes,
canonicalization parity failures, response length, invalid fraction, and
no-EOS fraction.

The mechanism passes its exploratory gate only if, in both tasks:

1. mean terminal pass@8 and coverage@8 exceed paired Dr.GRPO;
2. at least two of three paired seeds improve both endpoints;
3. mean@8 and greedy pass@1 each decline by no more than `0.03`;
4. validator/key parity has zero failures; and
5. bank telemetry shows nonzero new admissions followed by persistent
   multi-outcome support.

Graph-only success is domain-specific evidence. No E44-OGS result supports a
claim about unrestricted MATH strategy canonicalization.
