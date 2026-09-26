# E50H — second conditioned-teacher corpus

**Status: PREREGISTERED AT E50F 12/50, BEFORE ANY E50H GENERATION OR E50G
OUTCOME — 2026-07-26**

At this freeze, the first twelve E50F problems had three pairs satisfying the
pre-menu deterministic signature and two-exact-executions-per-route gates.
No E50F terminal result, E50G candidate file, menu audit, 0.5B sample, route
probe, or policy update existed.  The observed interim yield makes ten final
naturally supported problems possible but fragile after later attrition.

## Isolated adaptive change

For each of the same 50 fixed level-5 E47 problems, make one second
answer-blind Qwen2.5-72B proposal at new fixed seeds.  Show only the two
method labels from that problem's frozen first E50F proposal and require a
different pair from the same twelve-pair safe vocabulary.  Do not show the
reference answer, first-pair actions or executions, validator feedback,
canonicalization decisions, or outcomes.

Execute each of the two new methods in one independent eight-choice batch at
temperature 1.  The request must require a complete derivation in at most 450
words, literal use of every prescribed action, no substitute decisive step,
and a boxed answer.  This addresses first-corpus length truncation without
increasing the 1,024-token limit or the frozen 4,000-character audit bound.

Use an atomic, identity-locked per-problem journal.  The terminal E50H
artifact is corpus-only and always `pass=false`; it cannot authorize
training.

## E50G integration

E50G reconstructs and verifies each attempt's own request hash and response
batch provenance, exact-validates terminal executions, and applies its
unchanged safe-signature boundary.  For each problem, retain at most one
pre-menu candidate by decreasing smaller route-positive count, decreasing
total route-positive count, then earlier attempt index.  This ranking is
frozen before E50H data and does not use a relation vote, menu audit, 0.5B
sample, or training outcome.

Every retained pair still needs both independent Qwen72 soundness/binding
audits, stable proposal-to-menu signatures, zero cross-rendering false-new
controls, forced execution of both routes by Qwen2.5-0.5B, and natural
unforced support of at least two samples per route and eight total.  All
matched-toy and full-run advancement gates are unchanged.
