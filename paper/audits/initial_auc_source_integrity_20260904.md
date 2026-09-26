# Initial-checkpoint and AUC source-integrity audit

Audit date: 2026-09-04. This is a retrospective source audit, not a new
preregistration and not authorization to select a retry by its outcomes.

## Scope and reproducibility

The frozen result is `initial_auc_source_integrity_20260904.json`. It preserves
source file SHA-256 hashes, ledger identities, duplicate-row line numbers,
complete metric dictionaries, and separate response-free prompt/request hashes.
The accompanying `audit_initial_auc_source_integrity_20260904.py` is the exact
script executed. From the repository root, run:

```bash
python paper/audits/audit_initial_auc_source_integrity_20260904.py
```

It reads source data and writes `/tmp/paper_initial_auc_integrity.json`; it does
not change paper results. It requires `ujson`. Its discovery scan intentionally
includes all historical `debug_job*` sources to reveal collisions. This scan is
not the rule used to admit efficacy observations.

The scope was:

- All 150 E78/E79/E80-R1 core runs at step 0 (three models, five static domains,
  five seeds, control and replay).
- All 17 half-pass checkpoints, steps 0 through 3072 in increments of 192, for
  the 50 Qwen2.5-0.5B E78 control/replay runs.
- The same 17 checkpoints for all five Qwen2.5-0.5B RLEP Pantry runs.

There were no missing grid keys or malformed JSON rows. Sixteen run directories
contained repeated keys; three contained conflicting metrics when all historical
job sources were pooled. No repeated key changed the response-free prompt or
request identity. The other thirteen repeated-key run directories had identical
metric dictionaries on the audited grid.

## Supported claims and additional ambiguous initial checkpoint

All 50 Qwen2.5-0.5B core trajectories have unambiguous observations across the
entire audited grid. Their reported AUC summaries are unaffected. All 75 core
Dr.GRPO control initial evaluations are unambiguous. This audit does not include
the separate plain-GRPO initial-evaluation cohort.

One additional core initial checkpoint has a real within-source conflict:
Qwen2.5-3B MathIR ReplayDr seed 71, registered job `30277405`, source
`var/data/xdr_qwen25_3b_instruct_verified_first_replay_rehearsal_only_e80r1_qwen3b_aligned_mathir_replay_s71/debug_job30277405/eval_mode_coverage_draws.jsonl`.
Its SHA-256 is
`eb44925e3becc632b1665669d38334e348da9eb6eb1c47ce66ca4fb99577fee1`.

- Step 0, draw 0: lines 2 and 7 have mean@8 `0.05859375`; line 11 has
  `0.0576171875`.
- Step 0, draw 3: line 5 has mean@8 `0.0615234375`; line 14 has
  `0.060546875`.
- Pass@8, distinct@8, and the other metric fields agree for these repeated keys.

The E80-R1 completion scheduler amendment of August 21 authorizes operational
requeues and placement changes, but does not authorize selecting first/last
initial evaluations. Under a full-metric checkpoint policy, the ambiguous
initial checkpoint should be omitted from descriptive curves with a recorded
reason; the valid terminal endpoint remains admissible. A new full-grid
Qwen2.5-3B AUC calculation would need explicit handling of this missing initial
checkpoint. This finding does not invalidate the Qwen2.5-0.5B AUC.

The previously documented Falcon Countdown ReplayDr seed 59/job `30269051`
terminal conflict remains governed by
`paper/preregistration/e112r1_two_scale_49pair_terminal_integrity_amendment_20260828.md`.
That amendment excludes the run's efficacy values at all steps. Identical
initial repeats do not override that exclusion. The present initial/AUC audit
supplements the separate full terminal audit; it does not repeat it.

## RLEP Pantry recovery-source correction

The apparent Qwen2.5-0.5B RLEP Pantry initial conflicts arise from mixing failed,
superseded jobs with registered replacements in preserved run directories:

| Seed | Superseded jobs in ledger | Current registered job |
| --- | --- | --- |
| 43 | 30538126, 30579531 | 30688944 |
| 44 | 30538127, 30579532 | 30688945 |

The original jobs `30538126` and `30538127` each wrote repeated initial
four-draw evaluations before failing. Each current registered source has one
initial four-draw evaluation. The source-selection authority is explicit in
`var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json`, its `replaced_job_ids` and
`repair_history`, and these amendments:

- `paper/preregistration/e98r1_pantry_action_surface_repair_20260814.md`: repairs
  the incompatible offline Pantry action representation, preserves historical
  job identities, and restarts failed attempts without checkpoints.
- `paper/preregistration/e98r1_pantry_memory_recovery_20260818.md`: replaces the
  cancelled dependent attempts after the disposable smoke's memory failure;
  preserves run directories, frozen scientific settings, and recovery history.

Consequently, the valid current initial evaluations should remain. Selecting
sources by the current registered job identity is an authorized recovery rule,
not a choice between observed outcomes. Source hashes are:

| Source | SHA-256 |
| --- | --- |
| Both superseded 30538126/30538127 logs | e50acdd7492c57bcbed70d0a9124a060b2e755bf79a7e359e03c7834057d9497 |
| Current 30688944 log | ee2e59b7ed880abd72a3eefabcefb438bb199b84704b2f487cfef2aaec308afd |
| Current 30688945 log | 4026264e9940cc86d4d4c74af153a1c4497569ab8f2613fbd5b1218e16c62e39 |

The same source-binding rule applies to Falcon RLEP Pantry's registered
replacements, documented in `e100_sparse_rlep_dr_falcon1b_jobs.json` and
`paper/preregistration/e100_pantry_action_surface_repair_20260823.md`.

## Reader correction and validation

`ops/exp_scaling/plot_paper_aligned_domain_strips.py` now reads the current
registered job's evaluation source. It records superseded source hashes and
recovery-amendment hashes in provenance, rejects undocumented source jobs, and
never substitutes an old source when the current one is missing. Its terminal
summary uses the same source-bound reader. Conflicting repeated evaluations
within an admitted source still omit the entire nonterminal descriptive
checkpoint; a terminal conflict still fails closed. Identical repeats remain
admissible. This reader compares its reported pass@8/distinct@8 fields; the audit
above separately compares complete metric dictionaries.

Focused validation:

```bash
python -m pytest -q tests/test_paper_partial_block_reporting.py tests/test_paper_core_endpoint_integrity.py
```

Result: 22 tests passed. Regression cases cover recovery-source selection,
provenance, absent replacements, unknown source jobs, missing recovery
amendments, terminal summaries, and retained same-source conflict handling.

An additional run including `tests/test_paper_grid_contract.py` returned 23
passes and one unrelated failure: its one-axis-grid prohibition rejects the
existing `axis.grid(axis="x", ...)` in
`ops/exp_scaling/plot_e118_cross_domain_only_preview.py`. That file is outside
this correction and was not changed.

A subsequent read-only check used the corrected reader on all ten current
Qwen2.5-0.5B/Falcon RLEP Pantry runs. Every run retained all 33 available
quarter-pass checkpoints, including steps 0 and 3072, with one admitted current
source and no conflicting reported metrics. This check is broader than the
17-checkpoint frozen discovery scan for those RLEP runs.

Regenerate only the affected direct-baseline curve artifact:

```bash
python ops/exp_scaling/plot_paper_aligned_domain_strips.py --figure ucpo
```

This updates `paper/figures/direct_baseline_learning_curves_static_strip` in
PDF, PNG, and JSON forms. Refresh its corresponding workshop copy and rebuild
both PDFs afterward. No manuscript changes or figure regeneration were performed
as part of this delegated reader correction.
