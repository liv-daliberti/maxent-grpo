# Supplemental reported-comparator source audit — September 4, 2026

All **281 admitted source files and 356 requested checkpoints passed**: no conflicting duplicated `(step, draw_index)` keys, no identical duplicated keys, no missing sampled draws, and no source-binding issues. This supplemental audit closes the reported comparator inputs outside the [primary terminal audit](primary_terminal_integrity_20260904.md), including all 75 plain-GRPO initial and terminal checkpoints. It requires no further figure or numerical-result correction.

The [machine record](comparator_source_integrity_20260904.json) contains the complete source selection, registration/figure input hashes, per-file hashes, exact row numbers and row hashes, full reported metric dictionaries, and response-free prompt/request identity hashes. The [exact executed script](audit_comparator_source_integrity_20260904.py) is retained. The recorded completion time is `2026-09-04T21:22:09.484918+00:00`. The scan read and hashed **13,971,403,510 bytes**, using six independent source-reading workers.

## Reported-source coverage

| Cohort | Admitted sources | Checkpoints per source | Selection authority |
|---|---:|---|---|
| Plain GRPO | 75 | Initial step 0 and terminal step 3072 | E95 three-scale registrations plus the E114 Qwen3B seed extension; all runs contributing to the frozen baseline precheck |
| UCPO | 50 | Terminal step 3072 | Exact comparator jobs/seeds in `direct_comparator_endpoint_effects.json`, resolved through its registered ledger inputs |
| Sparse RLEP-Dr.GRPO | 47 | Terminal step 3072 | Exact completed comparator jobs/seeds in the same figure, respecting registered recoveries |
| Fixed semantic-only comparison | 50 | Terminal step 3072 | Qwen E83 with registered E85 Pantry replacements, and Falcon E86 |
| E112 exploratory treatment | 49 | Terminal step 3072 | Exact registered jobs and source hashes in the frozen E112-R1 result |
| E112 repaired replay comparison | 10 | Terminal step 3072 | Exact frozen E112-R1 registered jobs outside the primary 445-run audit |

Each requested checkpoint contains exactly four `fixed_seed_sampled_k_neutral` draws, indexed 0–3: **1,424 admitted sampled rows** in total. The scanner checks repeated keys against the entire reported metric dictionary and both prompt/reference and request-seed identities. It does not merely compare the two headline endpoints. All 59 admitted E112 sources match their frozen source hashes.

## Registered replacements and limits

Source selection uses the current registered job or the exact frozen E112 registered job. Explicitly superseded historical jobs are excluded using ledger replacement identities and repair-history amendment provenance; 7 admitted runs have excluded historical source paths. The record retains those excluded identities and amendment hashes. No outcome-based first/last-row selection, averaging of conflicting evaluations, new evaluation, or selection of a better-performing retry was used.

This is a retrospective source-integrity audit, not a revised preregistration. It covers actual reported inputs. Unfinished UCPO/RLEP trials and unused semantic-plus-replay branches are outside scope. Comparator trajectory checkpoints other than those expressly listed above are outside this supplemental scan; the separate [initial/AUC audit](initial_auc_source_integrity_20260904.md) retains its original trajectory scope. This check establishes uniqueness and source admissibility of recorded sampled evaluations; it does not independently recompute verifier outcomes from generated responses.

The previously identified Falcon Countdown ReplayDr.GRPO seed-59 terminal conflict remains excluded as documented in the primary audit. The companion initial/AUC audit's Qwen3B MathIR ReplayDr.GRPO seed-71 initial-checkpoint ambiguity also remains excluded from descriptive initial reporting, while its valid terminal remains admissible. This supplemental clean result does not reverse either correction.

See the [full evidence/story audit](story_evidence_audit_20260904.md) for interpretation and limitations, and the [paper revision validation record](paper_revision_validation_20260904.md) for final builds and focused checks.
