# E50K — fifth conditioned-teacher contingency

**Status: PREREGISTERED AT E50I 5/35, BEFORE THE OTHER 30 E50I RESULTS,
ANY E50J OR E50K GENERATION, OR ANY E50G OUTPUT — 2026-07-26**

At this freeze, the complete E50F/E50H union contained fifteen eligible
problems. One of the first five E50I attempts added a new eligible problem.
No later E50I result, E50J result, E50G candidate/menu/0.5B artifact, route
probe, or policy update existed.

After E50J becomes terminal, reproduce the frozen pre-menu filters on every
available E50F/E50H/E50I/E50J attempt. If their union has at least 30
eligible problems, emit a corpus-only no-op artifact and make no request. If
it has fewer than 30, generate exactly one fifth-stage answer-blind proposal
only for each still-ineligible problem. This is the final proposal stage.
The threshold preserves the preregistered three-to-one candidate buffer for
the independently fixed target of ten before double soundness audits and
natural Qwen2.5-0.5B support; it does not relax any downstream gate.

The prompt exposes only the method labels from all available earlier
proposals and requires a different pair from the same twelve frozen safe
pairs. It exposes no reference answer, prior action text, executions,
validator result, signature decision, support count, menu audit, or 0.5B
outcome. It asks the proposer to choose only a pair for which each engine
independently supplies a complete problem-specific route. Execute each new
method in one independent eight-choice batch at temperature one, with a
complete derivation of at most 450 words, literal action use, and a boxed
answer. Use new fixed seeds and an atomic identity-locked journal.

E50G reconstructs the request and response-batch provenance, rejects a
repeated label pair or repeated unordered safe-signature pair against any
prior attempt, and applies the unchanged exact validator, deterministic
signature, two-positive, double soundness/binding audit, cross-rendering
false-new, forced-route, and 64-sample natural-support gates. It retains at
most one candidate per problem by decreasing smaller exact-positive route
count, decreasing total exact-positive count, then earlier attempt index.
E50K is always `pass=false`, selects no problem, performs no policy update,
and cannot authorize training.
