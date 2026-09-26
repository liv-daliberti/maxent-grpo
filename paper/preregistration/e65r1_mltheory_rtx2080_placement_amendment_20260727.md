# E65R1 MLTheory RTX 2080 placement amendment

Frozen 2026-07-27 13:16:32 EDT, before any placement mutation.

The corrected E65R1 cohort was released with Graph and MathIR restricted to
MLTheory/general A5000 nodes 105 and 202--204. Graph seed 43 (job 30127478)
allocated on node105 and is not changed. At this boundary the five remaining
MLTheory jobs were all pending, had zero elapsed runtime, zero restarts, no
allocation, and exit code `0:0`:

- Graph seed 44: 30127479
- Graph seed 45: 30127481
- MathIR seed 43: 30127489
- MathIR seed 44: 30127490
- MathIR seed 45: 30127491

Read-only inspection found nodes 915 and 917 idle in both `mltheory` and
`lowprio`, each with eight RTX 2080 GPUs, 48 CPUs, and about 385 GB RAM. A
one-GPU `srun --test-only` request using the campaign's 8 CPUs, 64 GB, and
24-hour limit was eligible for node915 at 18:57 EDT, earlier than the A5000
estimates for these pending jobs (2026-07-28 through 2026-08-01).

The five pending jobs may therefore be user-held, amended in place to
`account=mltheory`, `partition=lowprio`,
`nodelist=node915,node917`, and `gres=gpu:rtx_2080:1`, audited while held,
and released together. Their job identities, seeds, frozen source and
execution snapshots, model, data, optimizer, controller/actuator settings,
checkpoint cadence, RNG state, and protocol identity must remain unchanged.
The resolved node and GPU type are reported per seed; hardware placement is
not used for checkpoint or result selection.

The non-MLTheory Countdown and Python jobs remain on their registered RTX
3090 route: a contemporaneous test of idle general RTX 2080 nodes estimated a
later start than their current RTX 3090 queue. This amendment therefore
changes only the five zero-runtime MLTheory jobs listed above.
