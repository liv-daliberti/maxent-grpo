# E50J — fourth conditioned-teacher contingency

**Status: PREREGISTERED AT E50H 16/50, BEFORE THE OTHER 34 E50H RESULTS,
ANY E50I OR E50J GENERATION, OR ANY E50G OUTPUT — 2026-07-26**

At this freeze, the complete first E50F corpus had eight problems satisfying
the deterministic safe-signature and two-exact-executions-per-route filters.
Among the first thirteen auditable E50H records, four pairs were executable,
but only two added support outside the first-attempt set. Six failed the
already-frozen exact execution-support gate and three failed the frozen safe
signature gate. No later E50H result, E50I result, E50G candidate/menu/0.5B
artifact, route probe, or policy update existed.

After E50I becomes terminal, reproduce the frozen pre-menu filters on E50F,
E50H, and any E50I attempts. If their union has at least 30 eligible
problems, emit a corpus-only no-op artifact and make no request. If it has
fewer than 30, generate exactly one fourth answer-blind proposal only for
each still-ineligible problem. The threshold provides a three-to-one
candidate buffer for the independently fixed target of ten before double
soundness audits and natural Qwen2.5-0.5B support; it does not relax any
downstream gate.

The fourth-stage prompt exposes only the method labels from all available
earlier proposals: normally six labels from three attempts, or four labels
if E50I was a no-op. It requires a different pair from the same twelve
frozen safe pairs and exposes no reference answer, prior action text,
executions, validator result, signature decision, support count, menu audit,
or 0.5B outcome. It explicitly asks the proposer to choose only a pair for
which each engine independently supplies a complete problem-specific route.
Execute each new method in one independent eight-choice batch at temperature
one, with a complete derivation of at most 450 words, literal action use, and
a boxed answer. Use new fixed seeds and an atomic identity-locked journal.

E50G reconstructs the fourth proposal request and response-batch provenance,
rejects a repeated label pair or repeated unordered safe-signature pair
against any prior attempt, and applies the same exact validator,
deterministic signature, two-positive, double soundness/binding audit,
cross-rendering false-new, forced-route, and 64-sample natural-support gates.
It retains at most one candidate per problem by decreasing smaller
exact-positive route count, decreasing total exact-positive count, then
earlier attempt index. E50J is always `pass=false`, selects no problem,
performs no policy update, and cannot authorize training.
