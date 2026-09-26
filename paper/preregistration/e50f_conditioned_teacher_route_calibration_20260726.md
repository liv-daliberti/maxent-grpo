# E50F — conditioned-teacher, naturally supported hard-MATH routes

**Status: PREREGISTERED BEFORE E50C TERMINAL RESULTS — 2026-07-26**

At this freeze, E50C had completed teacher sampling for 2 of its 50 fixed
problems and had emitted no validation, clustering, menu, or 0.5B execution
result.

**Pre-terminal quarantine amendment — 2026-07-26:** the later prospective
E50F2 and E50F4 hard-MATH relation audits found false-new errors in the
open-ended 72B judge before E50C or E50F became terminal.  Neither E50C nor
E50F may therefore authorize training, even if its internal diagnostic
`pass` field is true.  E50F now runs after every terminal E50C outcome solely
to produce its frozen answer-blind proposal/execution corpus for E50G's
deterministic safe-signature gate.  No E50F generation, validation, route,
menu, 0.5B, or training outcome existed when this amendment was made.

**Pre-generation safe-pair amendment — 2026-07-26:** before E50F started, its
answer-blind proposal prompt was restricted to the twelve decisive-operation
pairs frozen by E50G.  The prompt exposes the finite engine vocabulary but no
problem answer, derivation, route chosen for a problem, validator feedback,
or prior sample.  Both engines must be independently proposed and executed.
This improves coverage of the conservative canonical support without letting
the 72B relation vote create a new identity; E50G still deterministically
rejects unknown, mixed, same, or non-allowlisted signatures and independently
audits mathematical soundness.

Because the old relation boundary is quarantined, E50F terminates
diagnostically with `pass=false` immediately after atomically writing and
hashing all 50 answer-blind proposal/execution records.  It does not spend
compute on its now-nonauthorizing relation, menu, or 0.5B stages.  E50G
revalidates the frozen executions and owns every trusted downstream gate;
there is no omitted authorizing check.

Generation uses an atomic per-problem partial journal.  A scheduler or
endpoint interruption may resume only missing problem orders with their
original fixed seeds; it may not regenerate, select among, or replace a
completed record.  The final corpus must contain exactly one record for every
frozen order 0 through 49.  Resume additionally requires an exact checkpoint
identity match over the script, protocol, E50C result, E47 cohort, endpoint,
and frozen canonicalizer inputs.

## Activation and isolated change

E50F originally activated only if E50C did not yield ten bidirectionally
executable, naturally supported problems.  Under the pre-terminal amendment
above it always runs after E50C, but its result remains diagnostic and cannot
authorize training.  It retains E50C's problem cohort, answer
validator, calibrated pairwise-veto judge, finite-menu audits, frozen 0.5B
model, forced-route gate, neutral-support gate, and deterministic ranking.
It changes only how the initial teacher derivations are elicited.

## Frozen cohort and answer-blind method proposal

Use the exact 50 level-5 E47 problems in frozen order.  For each problem,
make one structured Qwen2.5-72B request at temperature 0.7 and a fixed seed.
The request contains the problem but not its reference answer, E50C
responses, prior route names, or validator feedback.  It must propose
exactly two high-level mathematical methods, each as two to six concrete
ordered actions.  It must explain the decisive-operation difference and
must not state, encode, or hint at the final answer.

The methods must instantiate exactly one pair from E50G's frozen
decisive-operation allowlist and retain the corresponding engine phrases in
their labels or actions.  This is a finite proposal vocabulary, not evidence
that either route is sound or distinct.

The pair is rejected unless its two action sequences use substantively
different decisive operations rather than reordered algebra, notation,
verbosity, or a shared derivation with cosmetic changes.  A rejected pair
is not regenerated or replaced.

## Independent route execution

For each proposed route, draw eight independent Qwen2.5-72B derivations at
temperature 1.0.  Each execution request contains the original problem and
only the selected action sequence; it does not contain the reference answer,
the other route's derivations, or validator feedback.  Retain only complete
derivations accepted by the corrected exact `math_verify` answer validator.
The calibrated judge's frozen 4,000-character item limit applies before
relation auditing.  Each route requires at least two independently sampled
validator-positive executions within that bound.

Run the frozen calibrated E47W pairwise-veto relation contract twice, with
independent permutation seeds, on at most the first three retained executions
per route after ordering by full response SHA-256.  Every audit must find the
within-route exemplars semantically consistent and every
cross-route comparison genuinely different.  Missing IDs, ambiguity,
disagreement, format failure, or an equivalence decision rejects the
problem.  Proposed prose alone never establishes a route.

## Finite menu and execution gates

Convert only the two executed route components into an S1/S2 finite action
menu.  Two independent temperature-zero Qwen72 audits must literally
execute both combos, bind each combo to its cited full derivations,
independently derive the reference answer, find every action sound and
sufficient, find no hidden decisive step or answer leakage, and unanimously
classify the routes as genuinely distinct.

For every double-audited menu:

1. draw 16 forced Qwen2.5-0.5B-Instruct samples for each route and require at
   least one exact-answer-plus-requested-route success per route; and
2. draw 64 independent unforced samples with strategy IDs and listing order
   explicitly marked non-preferential.

An unforced response counts only when the exact answer validator and frozen
finite-menu route canonicalizer both accept it.  Natural support requires at
least two counted samples from each route and at least eight counted samples
in total.  Wrong-route and answer-only responses receive no credit.

Rank passing problems by decreasing smaller unforced-route count, decreasing
total accepted unforced count, decreasing smaller teacher execution count,
then E47 problem order.  Select the deterministic first ten.  E50F passes
only with ten passing problems and performs no policy update.
