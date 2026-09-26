# E118-R1 zero-step variant-label repair

Frozen: 2026-09-01 after E118 jobs 31006356--31006369 failed in zero to two
seconds and before any E118 optimizer step or scientific endpoint existed.
Jobs 31006370--31006376 were held at zero runtime.

The original launcher supplied the descriptive labels maxrl_compute_matched
and maxrl_verified_replay as OAT_ZERO_VARIANT, but ops/train.sh accepts only
registered plumbing variants. This is an operational launch error.

Replace both descriptive labels with the registered
verified_replay_semantic_maxent_verified_support_discovery plumbing variant.
All scientific arm-defining environment variables remain unchanged: both arms
use the MaxRL task objective, and only ReplayMaxRL activates the verified replay
gradient while MaxRL retains identical replay compute with compute-only mode.

Submit a fresh 20-cell grid to new output directories and a new ledger. Preserve
the original ledger and failed job records as provenance. Use the requested
12-hour scheduler limit on mltheory, node302, with one A100 per cell. Validate
all held job records before release. No E118 response or outcome may inform this
repair because no E118 training step occurred.
