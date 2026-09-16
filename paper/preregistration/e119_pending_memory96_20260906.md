# E119 pending memory increases — September 6, 2026 UTC

The user approved the preceding recommendation to raise exactly22 queued E119
jobs from64 to96GiB: five Countdown cells and17 Pantry cells. The reviewed list
is in `var/artifacts/e119_pending_memory_audit_20260906/recommendations.json`.
All15 other Countdown cells demonstrated64GiB memory pressure. Pantry MaxRLs46
had a confirmed67.2GiB non-cache working set; the remaining Pantry upgrades are
preventive extrapolations. Pantry ReplayDr.GRPOs44 had a separate severe stall
without a confirmed live memory diagnosis. MathIR/Python requests remain64GiB.

Apply to ordinary pending jobs only, under their existing IDs. Briefly hold each
pending job to prevent admission during its resource update, increase only
MinMemoryNode to98304MiB, verify, and release our hold. No running learner is
stopped, cancelled, or requeued; no checkpoint or scientific progress changes.
Preserve submitted arguments/exports, cells, seeds, models, objectives, datasets,
CPU/GPU resources, accounts, partitions, QOS, placement, limits, dependencies,
restart counts, and main/continuation ledgers. Larger requests may reduce
simultaneous admissions.96GiB addresses observed demands, not a guaranteed
full-run peak bound.

The exact plan, before/after scheduler records, mutation receipts, and final
verification are stored under `var/artifacts/e119_pending_memory96_20260906/`.
