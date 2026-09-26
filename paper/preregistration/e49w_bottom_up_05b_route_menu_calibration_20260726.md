# E49W — bottom-up 0.5B-executable hard-MATH route calibration

**Status: PREREGISTERED BEFORE MENU GENERATION — 2026-07-26**

## Motivation

E49U showed that E49T's routes were mathematically valid and cleanly
separable by the frozen 72B gate, but none of the ten training problems had
both routes executable by the base Qwen2.5-0.5B model in eight forced
attempts.  The next toy must therefore construct routes bottom-up from paths
the small model can actually write, rather than weaken the answer validator,
semantic execution audit, or distinctness definition.

## Frozen source cohort

Use the 50 level-5 MATH training problems and 3,200 unchanged base-model
samples from E47.  Candidate problems are exactly those with at least two
ordinary `math_verify`-positive E47 samples, ordered by decreasing positive
count and then problem ID.  The frozen expected candidate count is 19.

For each candidate, Qwen2.5-72B receives only validator-positive natural
derivations.  To stay below the frozen context limit, the input is the first
12 responses after sorting by increasing character length, then sample ID;
responses longer than 6,000 characters are ineligible and no response is
truncated.  It must:

1. bind S1 to a mathematical path actually executed by at least one supplied
   response;
2. bind S2 either to a second genuinely distinct supplied path or to one
   newly proposed, concise route;
3. express both as finite, ordered, problem-specific action combos without
   including the final answer; and
4. identify the exemplar IDs used for every observed route.

Two independent temperature-zero 72B audits then literally execute both
routes using the auditor-only reference answer, verify sufficiency and
distinctness, check any claimed exemplar binding, and veto answer leakage,
hidden decisive steps, vague operations, or cosmetic route differences.
Both audits must pass.  Audit derivations and answers are never returned as
generation feedback.

## Base-model execution gate

For every double-audited candidate menu, force each route separately and
draw eight fresh Qwen2.5-0.5B-Instruct samples at temperature 1, top-p 1,
and at most 1,024 new tokens.  Success requires:

- ordinary exact-answer validation;
- unanimous assignment by the exact E49T frozen, blinded-calibrated 72B
  canonicalizer to the forced route; and
- no assignment to the opposite route.

A route is minimally executable at one fully gated success in eight.  A
problem is bidirectionally executable only when both routes pass.  The
calibration advances only if at least ten training problems pass.  The
future matched toy uses the deterministic first ten passing candidates in
the frozen candidate order.  Failed routes are never relabeled, merged, or
credited from answer correctness alone.

This is a calibration and data-construction gate, not a policy update.
