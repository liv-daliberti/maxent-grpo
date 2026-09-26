# E121 independent integrity audit — 2026-09-09

All five registered seeds completed their 3,072 optimizer updates. The complete post-freeze telemetry population passes the stronger audit: all 3,406 frozen identities across 1,575 seed–prompt pairs have finite, aligned scores at every scheduled replay visit. No identity was excluded. There are 29,041 identity observations over 2,689 actual post-freeze updates per seed (384 through 3,072 inclusive).

| Seed | Frozen prompts | Frozen identities | Identity observations | Visits per identity |
| --- | ---: | ---: | ---: | ---: |
| 43 | 310 | 636 | 5,504 | 8–9 |
| 44 | 314 | 697 | 5,961 | 8–9 |
| 45 | 316 | 635 | 5,411 | 8–9 |
| 46 | 323 | 755 | 6,273 | 8–9 |
| 47 | 312 | 683 | 5,892 | 8–9 |

The independent population denominator comes from `online_canonical_tracked_prompts` and `online_canonical_tracked_outcomes`, which count the complete bank, rather than from the scored histories alone. Both totals are constant from the freeze onwards and equal their pre-freeze totals at update 383. The observed fingerprint population exactly matches these independent totals in every seed. The complete prompt cycle repeats with period equal to the bank's prompt count; every identity appears at every visit of its prompt. Each row also agrees with the logged number of available, banked, and acted-on modes. This establishes coverage of the full frozen population rather than only the identities that happened to appear in the metrics.

Membership fingerprints, explicit ordered outcome sets, per-identity fresh counts, and uniform target weights are constant after the freeze. Local bank sizes are unchanged before versus after fresh-group processing. The freeze flag first becomes active at update 384 and never reverts. The global scheduler remains active, with one group per update and no bootstrap truncation. Mean and sequence scores, fingerprints, weights, counts, and applied-gradient diagnostics are finite and aligned. The coefficient remains 0.10, objective scale 1/16, and backward scale 16. No applied positive replay-score gradient is recorded.

The final `trainer/step=3073` record is an end-of-training log event carrying forward all training fields from update 3072. Its optimizer/global counters remain 3072. Both auditors assert exact equality of those carried-forward fields and exclude this record from visits. Receipt/export names use 3073, but there were exactly 3072 optimizer updates. Treating this final log event as an additional visit would add five spurious prompt visits.

The frozen code creates 52-bit fingerprints in float64, but the common metric aggregator converts tensors to float32 (`learner/run.py:3123`). No within-run identity collision is detected: the unique fingerprint counts equal the independent full-bank cardinalities and the complete schedules match. The audit therefore uses the actually recorded numeric fingerprints without claiming that full 52-bit precision survived serialization.

Per-row token counts are not logged directly. The sequence/mean score ratio is within 0.000004 of a positive integer on every row; inferred lengths are constant by identity and their sum exactly matches each group's logged replay-token total. Across seeds, inferred lengths range from 4 to 100 tokens, reinforcing the need to report sequence-score changes separately from normalized scores.

Successful-run cleanup pruned all bank checkpoints. The audit consequently establishes population completeness through independently recorded bank cardinalities, complete cyclic schedules, and unchanged explicit memberships; it does not claim a surviving checkpoint-based bank reconstruction. Frozen source, telemetry, completed model exports, and content hashes remain available. No new runs or scientific changes were made for this audit.

Jobs 31040762–31040766 all have Slurm `COMPLETED`, exit code `0:0`, on node204. Each seed has one attempt directory, a continuous update sequence, one `[slurm] host=` startup marker, and no logged checkpoint-resume match. E121 began on September 8 after an explicitly authorized scheduling-only amendment released stale E120 dependency barriers before the entire E120-R1 cohort had completed. The original protocol, source snapshot, seeds, freeze rule, model, and compute budget were unchanged. That scheduling amendment is recorded in `var/artifacts/e121_dependency_release_20260908T180159.json`; the within-approved-node relocation is in `var/artifacts/e121_placement_update_20260908T180828.json`.

These checks establish complete fixed-exemplar score telemetry. They do not identify canonical-mode probabilities, numerically validate the theorem's lower bound, or estimate replay's causal effect without a no-replay comparison.

Reproduction:

```bash
python paper/audits/e121_20260909/audit_integrity.py
python ops/exp_scaling/audit_e121_fixed_bank_survival.py
PYTHONPATH=src python -m pytest -q tests/test_e121_fixed_bank_survival.py
```

The independent audit is in `integrity.json`. The strengthened operational auditor writes `var/artifacts/e121_fixed_bank_survival_audit.json` (schema v2). Fourteen focused tests passed, including regression cases for a never-observed frozen prompt/key, a missing scheduled update, invalid terminal duplication, nonfinite or misaligned row scores, changing memberships, and changing fresh counts. The old auditor used observed histories alone and counted the final carried-forward record; it also used an upper-middle element rather than the standard median for even populations. All three issues are corrected in the current auditor.
