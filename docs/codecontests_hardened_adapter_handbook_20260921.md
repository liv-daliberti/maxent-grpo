# Hardened real coding application: source and adapter handbook

The tasks, statements, C++ output checkers, and labeled Python programs come from CodeContests+ (revision `96c850540fade31d384a25766461e0da6b08f5fc`), with original released input suites joined from CodeContests-O (revision `1a765191567b429f633bbd1c6e67b5890dfaf267`). These are real Codeforces constructive programming tasks. Exactly five additional input fixtures were constructed from source-labeled incorrect programs before inspecting policy outputs; their provenance is explicit. The application must therefore be described as real contest tasks with source-reviewed supplemental verification inputs.

Use `build_constructive_code_hardened_20260921` as the adapter module. Its accepted configuration is:

```json
{
  "slate_root": "/absolute/path/to/hardened/slate",
  "problem_ids": ["1454_A", "988_A"],
  "image": "/absolute/path/to/python-3.10-slim-c1e4e6c01eb4.sqsh",
  "runtime_root": "/job/local/verified/runtime",
  "build_root": "/job/local/checkers",
  "launcher": "/job/local/sandbox-launcher",
  "scratch_root": "/job/local/scratch"
}
```

Concurrent jobs must have distinct writable checker/build, launcher, and scratch paths. A complete validated runtime may be reused read-only. Frozen execution must set `OAT_ZERO_SOURCE_ROOT`, `OAT_ZERO_TESTLIB_ROOT`, and `OAT_ZERO_SANDBOX_SOURCE` to the corresponding frozen copies. Module imports follow the frozen `ops` path. The loader verifies project-relative source identities and bytes, the exact testlib pin, dataset/task/checker/input/reference hashes, and the full 24-program source-replay bijection.

`load_tasks(config)` yields tasks with `task_id`, complete rendered `prompt`, `family`, `split`, and `verify(text)`. A result includes `accepted`, `canonical_key`, `hard_violations`, and a full receipt. An ordinary incorrect program receives reward zero. Infrastructure failures, integrity failures, and a program accepted on its first execution but inconsistent on independent full-suite repetition are hard violations. Every accepted generated program executes the full suite twice and must receive the identical canonical key.

## Immutable initial pilot slate

`var/data/constructive_code_hardened_initial_20260921` admits 21 of the historical 23 source-admitted candidates (24 were originally proposed). All 21 pass exactly 12/12 labeled positive and 12/12 labeled negative Python replays. The original false-negative sources `1073_A` and `1399_D` are quarantined; the earlier `1352_F` quarantine remains visible. The original order and source selection have not been replaced after policy screening.

- Manifest SHA256: `735d4d0c3e77aa45554e3cbf791367d8dd8f4ea085207e799b1c21741d9837bb`.
- Strict quality receipt SHA256: `1c2420bbba67f0b0dd586615fe6dab273aa1a7391f7df702645f6f203654a1f7`.
- Loader source SHA256: `1f05ef03ff2ad9b8b3836f5b1d16ff9a17742a004d362666f0a5f7a3a1c3f0b4`.

All eight previously screened multi-mode development tasks survive the strict source gate: `1454_A`, `1569_A`, `361_A`, `1408_A`, `988_A`, `1323_A`, `1380_A`, `1038_B`. Their post-hardening policy capability must be measured under the new verifier before drawing conclusions. The historical faulty reward results remain diagnostic records.

The fixed fixture file has SHA256 `3e7c34e2d8b2e3bb6a5d4b329525bffaeced7e8fb2791f2d09f257bca65513cb`. Exactly one input was appended to each of `1016_D`, `1360_G`, `1408_A`, `1323_A`, and `1038_B`. Every original effective input remains byte-identical and in its original order, and all prompts and output checkers remain unchanged. Supplemental fixtures pass the original input validators and independently coded source constraints. They are now reward inputs, so they are no longer independent post-training stress tests.

All tasks use the new suite ID `codecontests_source_hardened_20260921_v1`. The suite ID is part of mode identity; the five appended suites also have new content hashes. Use fresh replay banks and training checkpoints. Old and new mode counts or PCMD are not measurements on an identical support and cannot be directly compared.

## Larger pool and heldout boundary

The larger hardened pool is complete at `var/data/constructive_code_hardened_larger_20260921`: **44 strict-admitted tasks from 56 prospective candidates**, comprising 29 training, 2 validation, and 13 heldout tasks. All 44 pass 12/12 positives and 12/12 negatives. There are 12 visible quarantines: six retained from the earlier source gate and six newly excluded by the strict gate.

- Final manifest SHA256: `7cb281632b8e58be1bfa33d0400ac69c3470c0e998ee88c7d5c982cec0ee4986`.
- Final quality receipt SHA256: `4509672ce27833cec74dd6a8b82cff945cf3f246546826a870371478db5747fc`.
- The legacy 32/2/16 readiness flag remains false. `strict_larger_ready_source_gate` explicitly records the accepted 29/2/13 scope.
- The admitted split ledger is `split_manifest.json`; the original ledger is preserved as `pre_hardening_split_manifest.json`.

| Pool | Assignment | Ordered sequence | Unordered partition | Unordered set |
| --- | ---: | ---: | ---: | ---: |
| Training | 7 | 14 | 3 | 5 |
| Validation | 0 | 2 | 0 | 0 |
| Heldout | 7 | 4 | 0 | 2 |

Training covers four families; heldout covers three. The sole reserved heldout partition task failed the strict source-negative gate.

| Quarantined IDs | Source-admission reason |
| --- | --- |
| `1073_A`, `1399_D`, `1326_A` | Only 11/12 source positives accepted; respectively checker token-case incompatibility, fixed-runtime resource failure, and Python integer-string conversion limit. |
| `1332_B`, `1360_F`, `1559_B` | A source-labeled wrong program passed the released suite: only 11/12 negatives rejected. These heldout exclusions used source programs alone. |
| `472_A`, `1047_A`, `1088_A`, `1196_C` | Earlier source-negative rates below the original 0.9 gate; `1047_A` also had an input-validator discrepancy. |
| `1352_F` | Original output checker uses unsupported regex anchors and rejects a valid canonical witness. |
| `1497_C2` | Original checker does not compile against the pinned testlib. |

The two validation tasks were selected by a source-only deterministic statement-hash rule: `1554_D` and `534_A`. All tasks exposed to the initial policy screen remain in training/development. The original prospective heldout reservation remains unchanged: source failures are quarantines, not replacements selected after model outcomes. The protected older IDs `361_B`, `1294_C`, and `149_C` remain unused and outside the larger pool. No heldout policy calls are part of source materialization or reference auditing.

For the larger adapter, omitting `problem_ids` selects the manifest's admitted training pool. Validation requires the explicit two IDs. Heldout loading requires both an explicit list from `admitted_test_problem_ids` and `allow_heldout: true`; enabling that flag does not silently select heldout tasks.

The larger build reuses initial 24-program replay ledgers only after matching the complete verification contract and source hashes, with the origin manifest, quality receipt, task, admission, and replay hashes recorded. It executed fresh complete 12+12 panels for the remaining 27 source-admitted candidates. Split and source-manifest metadata changes are declared; no source outcome is relabeled. `independent_replay_reuse_audit.json` binds an independent check of all 23 reused panels, and `post_replay_pool_sealing` records the additional post-audit validation of canonical task-record hashes, adapter IDs, and complete execution contracts. The reproducible sealing command is `ops/seal_constructive_code_hardened_larger_20260921.py`; it changes only pool/provenance metadata, never task verdicts or verifier behavior.

## Interpretation

The canonical modes are verified output-witness behaviors across the fixed suite, with task-specific symmetries removed. They measure useful alternative satisfying outputs, not distinct algorithms or source-code strings. Renaming variables does not create a mode. Distinct mode coverage must be reported alongside pass@1 and pass@k; a gain in pass@k by itself does not isolate diversity from a gain in marginal success probability.

Passing 24 source controls per task is a finite audit, not a proof of program correctness for every legal input. The original finite suites demonstrably accepted some incorrect source and policy programs, motivating the versioned hardening and quarantine gate. Main evidence must come from fresh hardened baseline and paired training evaluations with identical prompts, decoding, compute accounting, and verifier versions. Old raw generations may be regraded for explicitly labeled diagnostics. The small 13-task heldout pool supports a pilot with limited precision, not a broad benchmark claim.
