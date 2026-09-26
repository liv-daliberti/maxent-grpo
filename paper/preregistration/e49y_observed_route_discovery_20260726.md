# E49Y — observed-route discovery on exact OAT hard MATH

**Status: PREREGISTERED BEFORE SAMPLING — 2026-07-26**

## Motivation

E49W failed because constructing S2 after observing only one natural method
still produced equivalent or unsound routes.  E49Y therefore forbids invented
routes.  Every dual option must first appear as a correct, naturally written
Qwen2.5-0.5B derivation.

## Frozen discovery cohort and sampling

Use all 211 level-4 or level-5 rows, in source order, from the exact
`var/data/math12k_384_math500/train` artifact.  Draw 64 independent natural
Qwen2.5-0.5B-Instruct responses per row at temperature 1, top-p 1, and at
most 1,024 new tokens.  Apply the corrected ordinary `math_verify` answer
validator.  A row enters route discovery only if at least four responses are
answer-valid.

For each eligible row, send at most 16 complete validator-positive responses
to the frozen E49T Qwen2.5-72B endpoint, chosen by increasing character
length then response hash.  No response is truncated and responses longer
than 6,000 characters are ineligible.

## Stable observed-route gate

Run two independent temperature-zero, order-permuted 72B partitions.  Each
must assign every supplied response to exactly one route cluster.  The two
partitions must induce the identical pairwise same-route relation; otherwise
the row fails closed.  At least two stable clusters must each contain at
least two naturally generated responses.

Retain the two clusters with the largest observed membership; ties use the
lexicographically first member hash.  Qwen2.5-72B converts only those two
observed clusters into a finite S1/S2 action menu.  S1 and S2 must both cite
their actual natural exemplars; proposed routes are forbidden.  Two further
independent temperature-zero audits must literally execute both action
combos, confirm every cited exemplar binding, derive the auditor-only
reference answer, find no hidden decisive step or answer leakage, and
unanimously judge the routes genuinely distinct.

## Base-model forced execution gate

Order double-audited candidates by decreasing size of the smaller retained
natural cluster, then decreasing combined retained membership, then original
source index.  Force each route separately and draw eight fresh base-model
responses with the same sampling settings.  A success requires exact-answer
validation and unanimous assignment by the exact frozen E49T canonicalizer
to the forced route.  Wrong-route execution does not count.

Advance only if at least ten problems are bidirectionally executable.  The
future matched toy uses the deterministic first ten passing problems in the
frozen candidate order.  No failed route is relabeled or credited from answer
correctness alone.  This stage performs no policy update.
