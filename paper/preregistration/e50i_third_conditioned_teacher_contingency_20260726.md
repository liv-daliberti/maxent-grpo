# E50I — third conditioned-teacher contingency

**Status: PREREGISTERED AT E50H 4/50, BEFORE THE OTHER 46 E50H RESULTS,
ANY E50I GENERATION, OR ANY E50G OUTPUT — 2026-07-26**

At this freeze, the complete first E50F corpus had eight problems satisfying
the deterministic safe-signature and two-exact-executions-per-route filters.
The first four E50H records had one eligible problem.  No E50H terminal
artifact, E50G candidate/menu/0.5B artifact, route probe, or policy update
existed.

After E50H becomes terminal, reproduce the frozen pre-menu filters on E50F
and E50H.  If their union has at least 20 eligible problems, emit a
corpus-only no-op artifact and make no request.  If it has fewer than 20,
generate exactly one third answer-blind proposal only for each still
ineligible problem.  The threshold provides a two-to-one candidate buffer
for the independently fixed target of ten before double soundness audits and
natural Qwen2.5-0.5B support; it does not relax any downstream gate.

The third prompt exposes only the four method labels from that problem's two
earlier proposals and requires a different pair from the same twelve frozen
safe pairs.  It exposes no reference answer, prior action text, executions,
validator result, signature decision, or support count.  Execute each new
method in one independent eight-choice batch at temperature one, with a
complete derivation of at most 450 words, literal action use, and a boxed
answer.  Use new fixed seeds and an atomic identity-locked journal.

E50G reconstructs the third proposal request and response-batch provenance,
rejects a repeated label pair or repeated unordered safe-signature pair, and
applies the same exact validator, deterministic signature, two-positive,
double soundness/binding audit, cross-rendering false-new, forced-route, and
64-sample natural-support gates.  It retains at most one candidate per
problem by decreasing smaller exact-positive route count, decreasing total
exact-positive count, then earlier attempt index.  E50I is always
`pass=false`, selects no problem, performs no policy update, and cannot
authorize training.
