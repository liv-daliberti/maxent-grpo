# E113-R3-S2: held Qwen partition normalization

**Frozen:** 2026-08-19 16:35 EDT, after the first R3 all-or-none submission
attempt failed its held-record audit and before any R3 job was released or ran.

## Observed held-only routing behavior

Qwen validation probe job 30790897 was submitted with `--partition=all`, as
shown in its immutable `SubmitLine`, but the cluster's submission routing
recorded `Partition=mltheory`. That partition does not contain the requested
A6000 node set. The launcher detected the mismatch while the job was held,
canceled it with zero runtime, wrote no R3 ledger, and created no output. The
probe is not a science cell or training attempt.

## Prospective repair

For each newly submitted held Qwen R3 job, immediately issue
`scontrol update Partition=all` before its expanded-record audit. Continue only
if the job remains `PENDING` for `JobHeldUser`, has zero runtime, and the full
record now contains `Partition=all` plus every frozen A6000 and experimental
field. Any failure cancels the entire held set before release. Falcon jobs do
not receive this normalization.

This corrects scheduler routing only. It does not change any objective,
resource quantity or type, model, domain, seed, path, horizon, query cap, or
scientific denominator. The final ledger must bind this amendment and identify
the canceled held validation probe separately from its 50 science records.
