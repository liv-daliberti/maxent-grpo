# E49E answer-bound singleton-repair amendment

**Status: FROZEN BEFORE ANY SINGLETON-REPAIR JUDGE REQUEST OR POLICY TRAINING — 2026-07-24**

This amendment repairs only rows for which the frozen E49E trace-bank audit
retains zero sound routes. It does not add an eligible novelty outcome and
does not alter the matched E46 Haarnoja or Dr.GRPO training methods.

## Motivation and scope

The strict E49E action executor correctly rejects plausible but wrong
answer-blind proposals. For example, both proposals for the stamp-growth toy
row started from the purchase price rather than the already doubled current
offer and derived `$20` instead of `$40`. Such rejection is desired, but every
training/evaluation prompt still needs one valid executable route contract.

For every row, the repair first recomputes the frozen original trace
certification and, when present, the separately frozen finite-kernel
augmentation from their immutable candidate banks and completed audits.
Existing sound routes are never regenerated. Only a row with no surviving
original or augmented route enters repair.

## Fixed singleton procedure

Each gap has at most two predetermined proposal slots:

1. `minimal_direct_route`, seed `492141`;
2. `independent_checked_route`, seed `492142`.

The Qwen2.5-72B proposer sees the problem, reference answer, and—when present
in the frozen training source—the gold derivation. Those fields are
auditor-only. The returned contract must contain exactly one strategy, two to
seven ordered actions, use every declared action once, remain
self-contained, and omit the reference answer, boxed answer, evaluated final
value, and worked numerical result. Local parsing, action-reference closure,
and an explicit answer-leakage filter run before auditing.

The first locally valid proposal is checked by the unchanged E49E
`literal_action_executor` and `adversarial_action_checker` with seeds `492111`
and `492112`. The frozen MATH verifier independently checks both derived
answers. The first double-sound route is retained; if neither fixed slot
passes, the stage fails. Completed proposal and audit outputs are cached
atomically and cannot be resampled.

Before that verifier call, the auditor's derived-answer surface may undergo
the frozen `audited_math_answer_surface_v1` normalization. This transformation
only strips an allowlisted unit suffix and normalizes mathematical typography
such as Unicode radicals; it cannot change numbers, variables, operators, or
mathematical values. Thus `4√2 cm` may be compared with `4\sqrt{2}`, while
`4√3`, `$20`, and `$40` remain different. The original completed traces are
re-evaluated under this conservative surface rule and are not resampled.

Every repaired row remains a singleton. It cannot satisfy the multi-route
support threshold, earn a first-discovery comparison against a second route,
or contribute a normalized-entropy dual observation.

## Reference-closed finalization

Existing E49E certification used the E49D pruning helper, which renumbered
surviving actions but did not rewrite explicit action IDs inside plan text.
Before any policy training, this repair recomputes every original certified
menu with reference-closed pruning: action definitions, combos, and plans are
renumbered together. The completed judge decisions and selected original
strategy IDs do not change.

Both the five known-invalid controls and three known-equivalent controls must
still pass against the immutable raw trace records before any repair request.
The exact raw records, finite-kernel augmentation records, E49D input, source
data, control manifests, endpoint, canonical E49E protocol, prior amendments,
this amendment, source snapshot, preflight, launcher, and Slurm wrapper are
frozen into the repair identity.

The repaired artifact remains pre-manual-audit. It cannot launch training
until every retained multi-route pair has a complete blinded manual decision
ledger and the frozen support, false-new, and prompt-length gates pass.
