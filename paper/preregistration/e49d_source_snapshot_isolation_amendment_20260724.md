# E49D source-snapshot isolation amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

E49D's configuration preflight at 09:04 EDT created and hash-verified the
immutable training-source snapshot
`e49d_maximal_support_math_27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd`.
Its source files match the implementation hashes frozen in the E49D protocol.

At 10:37 EDT, a separate E50 launch legitimately changed the shared live
`src/` tree. No E49D training had launched. To prevent an unrelated concurrent
experiment from silently changing E49D, the E49D launcher now requires and
uses the already frozen source snapshot above instead of taking a new snapshot
from the mutable live tree. It fails if that snapshot is absent or its tree
hash differs.

The recorded core-contract identity is computed from that same frozen tree,
not from the mutable live `src/`. The terminal execution-audit Slurm job also
exports that tree as `OAT_ZERO_CAMPAIGN_SOURCE_ROOT`, and its Python auditor
imports the menu parser and runtime canonicalizer from that root. Thus
training, identity recording, and terminal re-audit all use the same source
bytes.

This is experiment isolation only. It does not alter E49D code, data,
proposal or audit behavior, policy inputs, runtime validation, reward, E46
controller, cohorts, schedule, or gates.
