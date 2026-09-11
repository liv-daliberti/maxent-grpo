# E49U — 0.5B finite-menu route-capability calibration

**Status: PREREGISTERED BEFORE GENERATION — 2026-07-26**

## Motivation

E49T established that natural derivations restore nonzero answer-and-route
gated reward, but its first epoch was approaching completion without any
prompt-local canonical bank reaching support two.  The menus were certified
for mathematical validity and 72B route separability; neither certification
proved that Qwen2.5-0.5B could successfully execute both routes.

## Frozen cohort

Use all 20 E49T dual-menu problems: ten train and ten evaluation problems.
For each problem, force each of its two certified strategies separately and
draw eight responses from the unchanged Qwen2.5-0.5B-Instruct base checkpoint
at temperature 1.0, top-p 1.0, and at most 1,024 new tokens.  The prompt keeps
the exact certified menu but names the assigned strategy and exact action
combo.  This is a read-only capability audit, not a policy update or
controller phase.

The ordinary `math_verify` answer validator runs first, including the
pre-generation scalar-`solve(Eq)` compatibility repair that prevents valid
rational equations from being converted to false negatives. The exact
frozen E49T canonicalizer then applies its two independently permuted,
temperature-zero Qwen2.5-72B menu-route audits.  A route success requires both
the correct final answer and unanimous assignment to the forced strategy.
A declaration assigned to one route while executing another cannot pass.

## Decisions

For every problem-route pair report answer success and fully gated route
success out of eight.  A route is minimally executable when at least one of
eight samples passes the full gate.  A problem is bidirectionally executable
only when both certified routes meet that threshold.

The next matched toy iteration may use only dual problems proved
bidirectionally executable here.  If fewer than ten train problems qualify,
new easy-but-distinct routes must be constructed and certified before another
matched training launch.  No failed route may be relabeled, merged, or
credited from answer correctness alone.
