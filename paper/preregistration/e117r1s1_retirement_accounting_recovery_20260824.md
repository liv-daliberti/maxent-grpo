# E117-R1-S1 retirement-accounting recovery

Frozen: 2026-08-24 after the first E117-R1 application stopped and before a
second replacement submission. No run artifact or endpoint was read.

The first E117-R1 application held and scientifically audited 12 lowprio
replacements, wrote provisional mapping files, canceled stale audit job
30873569 and zero-step originals 30873543--30873554, then checked terminal
accounting. Slurm reports a user cancellation state as
`CANCELLED by 363432`; the application compared that full string to the token
`CANCELLED` and treated the successful retirement as a failure. Its pre-release
exception handler canceled all 12 held replacements and removed the provisional
mapping files. All original and attempted-replacement cells have zero runtime
and no run directory.

Correct the parser to normalize a scheduler state at its first whitespace and
`+` suffix. Permit the replacement launcher to resume when the exact originals
and stale audit are already canceled at zero runtime. It must re-materialize and
held-audit a fresh 12-job replacement set, preserve byte-exact scientific
exports, skip the already-complete retirement action, release only after all
held audits pass, and attach a new audit dependency. On any new pre-release
failure, cancel the newly submitted replacements regardless of the original
retirement state. No scientific configuration, node, partition decision, seed,
or analysis rule changes.
