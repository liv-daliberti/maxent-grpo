# E117 Stage 1-S1 effective `cs` route amendment

Frozen: 2026-08-30 after the first Stage 1 submission transaction failed its
held-job audit and before any Stage 1 allocation, run directory, response, or
optimizer update existed.

The launcher requested partition `all`, account `allcs`, exact node202, and an
A5000 GRES for the first held job. The site `job_submit/lua` plugin normalized
that zero-runtime job's effective partition to `cs`, which is the native
partition of node202/node203. The prospective launcher expected the requested
alias `all`, failed closed, canceled the held job, and installed no release
ledger or audit job. No scientific export or resource changed.

For the replacement transaction, continue to request `all`/`allcs` but require
the held training record to expose effective partition `cs`. Continue to
require the exact registered node, A5000 GRES, one GPU, eight CPUs, 64 GiB,
seven-day limit, export hash, and start dependency. The CPU-only terminal audit
has no exact node constraint and remains effectively on `all`.

This amendment changes only validation of the scheduler's deterministic route
normalization. It changes no source, data, arm, seed, evaluation request,
endpoint, gate, resource envelope, causal contrast, or confirmation boundary.
