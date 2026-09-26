# E50C — 72B-teacher, 0.5B-executable hard-MATH route calibration

**Status: PREREGISTERED BEFORE E49AB/E50A/E49AC TERMINAL RESULTS — 2026-07-26**

## Contingent activation

E50C runs only if none of E49AA, E49AB, E50A, and E49AC yields ten
bidirectionally executable observed-route problems.  E49AC was added to this
activation gate before E49AB, E50A, or E49AC had terminal results.  Their
failures remain failures; E50C changes the source of candidate derivations
rather than weakening their route-distinctness decisions.

## Frozen problem cohort and teacher sampling

Use the exact 50 level-5 MATH problems frozen by E47, in its original order
and with source indices fixed by
`var/artifacts/e47_math_strategy_calibration_v1/manifest.json`.  For each
problem, draw sixteen independent Qwen2.5-72B-Instruct-AWQ solutions from the
frozen E49T endpoint in two batches of eight, temperature 1, top-p 1, and at
most 1,024 new tokens.  The prompt contains the problem but not the reference
answer, route names, prior responses, or a requested mathematical method.
Retain only solutions accepted by the corrected exact `math_verify`
validator; at least four positives are required.

No teacher response may be repaired, completed, or credited from prose
alone.  Full private responses and hashes are retained.

## Conservative route discovery

Run the exact calibrated E47W pairwise-veto canonicalizer, whose frozen
calibration has zero injected false-new errors, zero blinded analyst-same
false-new errors, and all semantic-regression gates passing.  Each problem
is processed as one group.  A candidate needs two emitted components with
at least two independently sampled teacher responses each.  Retain the two
largest components, ties by canonical key.

Convert only those observed teacher components to a finite S1/S2 action
menu.  Proposed third routes and hidden answer-bearing steps are forbidden.
Two independent temperature-zero audits by the frozen 72B endpoint must
literally execute both combos, bind each route to its cited full teacher
exemplars, independently derive the exact reference answer, find every
action concrete and sufficient, detect no answer leakage or hidden decisive
step, and unanimously classify S1/S2 as genuinely distinct rather than
routine algebra or paraphrase.

## 0.5B execution and natural-support gates

For every double-audited menu:

1. force Qwen2.5-0.5B-Instruct to execute S1 and S2 separately for sixteen
   independent samples per route;
2. require at least one response per route accepted by both the exact-answer
   validator and the frozen E49T finite-menu canonicalizer, with wrong-route
   responses receiving no credit; and
3. present the same menu without a forced declaration, explicitly state that
   IDs and listing order are non-preferential, and draw 64 independent
   samples at temperature 1.

An unforced response counts only when its exact answer and its exact executed
route both validate.  A problem passes natural support only when each route
has at least two counted unforced responses and at least eight total counted
responses.  This ensures the online E46 bank sees real two-route support
without a forced exploration phase.

Rank passing problems by decreasing smaller unforced route count, decreasing
total accepted unforced count, decreasing smaller teacher-component size,
then E47 problem order.  Advance only with ten passing problems and select
the deterministic first ten.  This calibration performs no policy update.
