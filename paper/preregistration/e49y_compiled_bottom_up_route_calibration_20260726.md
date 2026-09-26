# E49Y — compiled bottom-up 0.5B route calibration

**Status: PREREGISTERED BEFORE GENERATION — 2026-07-26**

## Motivation

E49T reproduced the internal E46 causal chain but failed its terminal
route-coverage gate.  E49U also found zero of ten training dual menus
bidirectionally executable by Qwen2.5-0.5B.  The first bottom-up calibration,
E49W, then failed before semantic auditing because all 19 generated menus
assigned the same action-ID combo to S1 and S2.  A frozen one-candidate
diagnostic reproduced the deterministic rejection
`strategy action combos must be distinct`.  No E49W menu reached either
auditor or policy training.

E49Y prospectively fixes only that representation error.  It does not weaken
answer validation, semantic route audits, distinctness, or the frozen E49T
canonicalizer.

## Frozen cohort and deterministic compilation

Reuse E49W's exact frozen 19-problem cohort and order: level-5 E47 training
problems with at least two ordinary `math_verify`-positive base-model
samples, ordered by decreasing positive count and problem ID.  For each
problem, expose the same first 12 validator-positive exemplars ordered by
length and sample ID, with no response truncation.

Qwen2.5-72B must return exactly S1 and S2.  Each route contains two to four
concrete, problem-specific operations, a concise answer-blind plan, a source
kind, and up to three cited exemplar IDs.  S1 must be observed.  S2 may be
observed or newly proposed.  A proposed route has no exemplar IDs.

A deterministic compiler concatenates S1's operations followed by S2's,
assigns contiguous action IDs, and gives each strategy only its own disjoint
ordered combo.  The compiler makes no mathematical decisions, sees no
reference answer, and may not add, remove, reorder, merge, or rewrite an
operation.  Malformed source claims, duplicate route IDs, overlong actions,
or any parser failure reject the proposal.

## Unchanged semantic and execution gates

Two independent temperature-zero 72B auditors receive the problem,
auditor-only reference answer, compiled menu, source claims, and cited
exemplars.  Both must literally execute every action, derive the reference
answer, find every decisive step present, find no answer leakage, verify
observed exemplar binding, find the actions concrete for a small model, and
judge S1/S2 genuinely distinct by named decisive operations.

Every double-audited menu is then forced route-by-route on the unchanged base
Qwen2.5-0.5B-Instruct model: eight samples per route, temperature 1, top-p 1,
and 1,024 output tokens.  A success requires both exact-answer validation and
unanimous assignment by the exact frozen, blinded-calibrated E49T
canonicalizer to the forced route, with no opposite-route assignment.
This calibration may use any single Ampere-or-newer GPU with bfloat16 support;
hardware placement is not an experimental factor.  The later matched policy
training remains one A100 per arm.

A route is minimally executable at one fully gated success in eight.  A
problem is bidirectionally executable only when both routes pass.  Advancement
requires at least ten such problems; the next matched toy may use only the
first ten passing candidates in the frozen order.  Failed routes are never
relabeled, merged, or credited from answer correctness alone.
