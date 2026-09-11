# E50G — safe-signature conditioned-teacher route calibration

**Status: PREREGISTERED 2026-07-26 after E50F4 failed, before E50C or E50F
terminal output**

**E50H AMENDMENT PREREGISTERED AT E50F 12/50, BEFORE E50H GENERATION OR
ANY E50G OUTPUT — 2026-07-26.**  At that freeze, only three of the first
twelve E50F problems passed the deterministic pre-menu signature and
two-positive-executions-per-route filters.  The amendment adds the
answer-blind second-attempt corpus specified in
`e50h_second_conditioned_teacher_corpus_20260726.md`; it does not change any
menu, false-new, 0.5B, toy-training, or full-run gate.

**E50I CONTINGENCY AMENDMENT PREREGISTERED AT E50H 4/50, BEFORE THE
REMAINING E50H, ANY E50I, OR ANY E50G OUTPUT — 2026-07-26.**  At that
freeze, the completed E50F corpus contributed eight eligible problems and
the first four E50H records increased the observed union by only one.  The
bounded contingency in
`e50i_third_conditioned_teacher_contingency_20260726.md` is activated only
if the complete E50F/E50H union has fewer than twenty eligible problems.
It adds one answer-blind third pair only for still-ineligible problems, with
new fixed seeds and both prior label pairs excluded.  Twenty is a candidate
buffer before independent menu and 0.5B natural-support attrition; no E50G
pass threshold, validator, sampling count, ranking rule, or downstream gate
is changed.

**E50J CONTINGENCY AMENDMENT PREREGISTERED AT E50H 16/50, BEFORE THE
REMAINING E50H, ANY E50I/E50J, OR ANY E50G OUTPUT — 2026-07-26.**  The
first thirteen auditable second-attempt records contained four executable
pairs but only two new eligible problems; six failed execution support and
three failed the frozen signature gate.  The bounded contingency in
`e50j_fourth_conditioned_teacher_contingency_20260726.md` runs only if the
complete first-three-attempt union has fewer than thirty eligible problems.
It adds one answer-blind fourth-stage pair only for still-ineligible
problems.  Thirty is a candidate buffer before independent menu and 0.5B
natural-support attrition; no E50G or downstream pass condition changes.

**E50K FINAL CONTINGENCY AMENDMENT PREREGISTERED AT E50I 5/35, BEFORE THE
REMAINING E50I, ANY E50J/E50K, OR ANY E50G OUTPUT — 2026-07-26.**  The
complete first-two-attempt union was fifteen, and one of the first five
third attempts added a new eligible problem.  The final bounded contingency
in `e50k_fifth_conditioned_teacher_contingency_20260726.md` runs only if the
complete first-four-attempt union remains below thirty.  It adds one
answer-blind fifth-stage pair only for still-ineligible problems.  No
novelty, validation, ranking, sampling, training, or advancement gate
changes.

## Motivation and isolated change

The open-ended Qwen2.5-72B relation judge is not sufficiently calibrated to
define novelty.  On the frozen 18-pair sound manual set, E50F2 produced two
false-new decisions among six same-strategy pairs.  Pair-local finite-family
E50F4 improved this to one false-new, but also produced five false merges and
three unstable assignments.  Both remain failed and authorize no training.

E50G separates two decisions that earlier variants conflated:

1. Qwen72 may propose routes, execute them, and audit whether each route is
   sound, answer-correct, complete, concrete, nonleaking, and genuinely bound
   to cited derivations.
2. Qwen72 may not decide whether the pair creates a new canonical strategy.
   That boundary is a deterministic, fail-closed comparison of explicit
   decisive-operation signatures.

The signature gate recognizes only a frozen vocabulary and only the following
incompatible pairs:

- CRT construction versus finite constraint search;
- calculus versus a sharp inequality;
- Euclidean algorithm versus prime factorization;
- quadratic/root formula versus Vieta relations;
- polynomial expansion versus strategic-value evaluation;
- finite constraint search versus symbolic interval solving;
- finite constraint search versus a balance/convexity argument;
- factor localization versus a quadratic/root formula;
- vector geometry versus synthetic geometry;
- dynamic programming versus a closed-form count;
- complementary counting versus inclusion-exclusion; and
- incidence double counting versus unordered-pair counting.

Unknown, mixed, unstable, same-signature, and all other pairs are merged or
rejected.  Tree/product, tuple/falling-factorial, decimal/common-denominator,
and other representational changes cannot create support.

The deterministic gate was developed on the already unblinded E49R manual
set, where it has zero false-new errors on six sound same-strategy pairs and
one false merge among twelve sound distinct pairs.  This is explicitly
retrospective development evidence, not an independent calibration result.
E50G therefore adds prospective cross-rendering and runtime controls below.

As an additional explicitly retrospective whole-cohort diagnostic, the
unchanged frozen signature source was applied to all 150 E47
anchor-versus-exact-duplicate, format, and lexical-paraphrase comparisons:
zero were classified as new and zero changed signatures.  E50G verifies the
immutable audit artifact and source hash, but this diagnostic does not replace
the prospective controls below.

## Activation and frozen candidate corpus

E50G runs after terminal E50C, E50F, E50H, E50I, E50J, and E50K.  E50C's open-ended
novelty boundary was quarantined before it became terminal, and E50F/E50H/
E50I/E50J/E50K are corpus-only artifacts; none can authorize training.  E50G reuses
E50F's exact answer-blind conditioned proposals, E50H's separately
preregistered answer-blind second proposal per problem, and, only when the
frozen E50I trigger fires, one third proposal for each problem still
ineligible after those two attempts.  If the frozen E50J trigger fires, it
also reuses one fourth-stage proposal for each still-ineligible problem.
If the final E50K trigger fires, it reuses one fifth-stage proposal for
each problem still ineligible after all earlier attempts.
Each attempted pair has sixteen
independently sampled Qwen72 executions—eight per proposed route.  E50G
makes no proposal itself and uses no open-ended relation decision.

Before E50F generation began, its proposal prompt was restricted to these
same twelve frozen pairs.  This is only candidate elicitation: it supplies no
answer or problem-specific solution, and cannot establish support.  E50G
recomputes signatures, exact-validates independent executions, and applies
all soundness and natural-support gates below.

E50H uses new fixed seeds and asks for a different pair while exposing only
the first attempt's two method labels.  E50G verifies that exclusion field,
reconstructs the attempt-specific request, rejects an exact repeated label
pair or repeated unordered safe-signature pair, and otherwise applies the
same gates.  E50I exposes only the two earlier pairs' four method labels;
E50G reconstructs that exact request and rejects a third pair matching
either earlier label pair or unordered safe-signature pair.  E50G also
recomputes the complete first-two-attempt eligible union and requires exact
agreement with the E50I trigger record.  E50J analogously exposes only all
available earlier method labels.  E50G reconstructs that request, rejects a
repeat against every earlier label or signature pair, and requires exact
agreement with the first-three-attempt E50J trigger.  E50K repeats the same
answer-blind exclusion and exact trigger
recomputation over the first four attempts.  Among eligible attempts for one
problem, retain at most one by decreasing smaller exact-positive route
count, decreasing combined exact-positive count, then earlier attempt
index.  This rule was frozen before any E50H, E50I, E50J, E50K, or E50G output and never
observes a menu audit, 0.5B sample, or training outcome.

Use the exact 50 level-5 E47 cohort.  Re-run the corrected exact-answer
validator on every frozen execution, verify response hashes, enforce the
4,000-character bound, and require at least two independent positives for
each route.  The two proposal methods must form one frozen safe-signature
pair.

Before validation, reconstruct and hash the exact answer-blind proposal
request.  Require a terminal proposal response with a nonempty server ID,
the two-method/two-to-six-action schema, exactly two independent execution
response IDs, exactly eight indexed choices per route, terminal retained
executions, and byte-matching response hashes.  A declaration flag, route
label, or later prose audit cannot substitute for this provenance contract.

## Menu construction and prospective false-new controls

Convert each eligible pair to an S1/S2 menu using at most three independently
sampled, validator-positive executions per route.  Run the same two
temperature-zero Qwen72 menu audits as E50F.  Ignore only the audits'
uncalibrated pair-relation label.  Both audits must still unanimously verify
each route's literal soundness, completeness, exact answer, concrete actions,
absence of hidden decisive steps and answer leakage, and binding to its cited
full executions.

Recompute deterministic signatures from the final menu using only the
actions referenced by each combo.  A menu passes only if:

1. both final signatures equal their corresponding proposal signatures;
2. the final signatures remain a frozen safe pair;
3. comparing each proposal route with its independently generated menu
   rendering does not create a distinct pair; and
4. reversing route order leaves the decision unchanged.

Thus a 72B prose relation cannot manufacture canonical support.

The expensive per-problem menu generation and double audit use an atomic
partial journal.  Resume schedules only missing source indices and requires
an exact identity match over the E50G script/protocol, signature source and
audits, E50C/E50F/E50H/E50I/E50J/E50K inputs, endpoint, cohort, runtime controls, and
candidate-record and conditioned-record-contract hashes.  It cannot
regenerate or choose among a completed audit.

## Frozen 0.5B execution and natural-support gate

For every passing menu, use Qwen2.5-0.5B-Instruct at the frozen revision and
the E50C sampling contract:

- 16 forced samples for S1 and 16 for S2;
- 64 neutral, unforced samples;
- exact MATH answer validation before route assignment; and
- the frozen E49T two-pass finite-menu execution classifier.

A forced route requires at least one exact-answer-plus-requested-route
success.  Any forced response accepted as the wrong listed route fails the
entire calibration.  Natural support requires at least two accepted unforced
samples per route and eight accepted samples total.

Rank passing problems by decreasing smaller unforced route count, decreasing
total accepted count, decreasing smaller teacher execution count, then E47
problem order.  Select exactly ten.  E50G passes only with ten selected
problems, zero wrong-route forced accepts, zero cross-rendering false-new
controls, stable proposal-to-menu signatures, and terminal well-formed
evidence.  E50G performs no policy update.

Only a passing E50G may materialize a fresh matched three-epoch toy.  Full
384-train/MATH-500 training remains prohibited until that toy independently
passes the existing E46 mechanism and task-quality gates.
