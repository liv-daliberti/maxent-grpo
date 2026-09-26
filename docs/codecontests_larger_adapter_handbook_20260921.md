# CodeContests constructive data and adapter handoff

The source-audited larger pool is `var/data/constructive_code_larger_20260921`. Its manifest SHA256 is `becd9b0da6273662cc6369daac81161e90839f923bd112eee532d8134c4474c1`. The authoritative ID lists are in `split_manifest.json`:

| Role | Tasks | Selection |
|---|---:|---|
| Train | 32 | All 23 initially admitted tasks, plus nine admitted training reserves |
| Validation | 2 | `1554_D`, `534_A`: the two smallest normalized-statement SHA256s among the 11 admitted training reserves |
| Heldout | 16 | Source-audited prospective test reservations and structural replacements; no policy calls |

The original heldout IDs `361_B`, `1294_C`, `149_C` remain protected and unused. The 16 runnable heldout IDs are `1023_C`, `1093_B`, `1325_A`, `1332_B`, `1360_F`, `1430_A`, `1436_B`, `1450_A`, `1549_A`, `1559_B`, `1606_A`, `544_B`, `710_C`, `1092_A`, `1413_A`, `1520_C`. They cover seven assignment, six sequence, two unordered-set, and one anonymous-partition tasks. Both validation tasks are sequences; validation is a small development check, not a representative final benchmark.

The initial 23-task screen remains frozen separately at `var/data/constructive_code_wider_20260921`, manifest SHA256 `51e2d88f8a8e699a96da1af307e94936691b0e60fd2a89bc9cde8e497158cfb9`. Its capability outcomes do not filter the larger 32-task train pool. The eight-task learning pilot is explicitly capability-selected from that initial pool; it is a development experiment.

## Adapter configuration

Use `adapter_module: "build_constructive_code_wider_20260921"` with `PYTHONPATH` including the frozen `ops` and `src`. Imports respect `OAT_ZERO_SOURCE_ROOT`. Supply these adapter configuration fields:

```json
{
  "slate_root": "/absolute/path/to/frozen/constructive_code_larger_20260921",
  "problem_ids": ["1454_A", "1554_D", "534_A"],
  "allow_heldout": false,
  "image": "/absolute/path/to/python-3.10-slim-c1e4e6c01eb4.sqsh",
  "runtime_root": "/unique/job/work/runtime",
  "build_root": "/unique/job/work/checkers",
  "launcher": "/unique/job/work/launcher",
  "scratch_root": "/unique/job/work/scratch"
}
```

Every concurrent job or preflight needs separate writable launcher/build/scratch paths. A complete verified runtime may be reused read-only. Keep all such outputs outside frozen data.

For the trainer, set `train_ids` to the selected train list and `eval_ids` to the two validation IDs. Set `adapter_config.problem_ids` to their union. For the shared evaluator, set `task_ids` and `adapter_config.problem_ids` to the same explicit evaluation list. Omitting `problem_ids` on the larger pool loads exactly the 32 train tasks. Heldout IDs are rejected unless `allow_heldout` is explicitly true; final evaluation must also supply the explicit 16 heldout IDs. Setting that flag alone does not change the default train selection.

`load_tasks(config)` returns tasks with `task_id`, a complete rendered chat `prompt`, `family`, `split`, `metadata`, and `verify(text)`. Verification returns binary acceptance, a stable canonical key or null, hard violations, and the complete receipt. Every initially accepted program runs independently over the complete source suite twice; both acceptance and canonical behavior must match. Ordinary wrong-program rejection is reward zero; infrastructure, integrity, or unstable acceptance is hard failure.

Use `evaluate_real_domains_20260921.py --config REQUEST.json --output FRESH.json --preflight` to load frozen data, compile original checkers, verify the runtime, and bind exact prompts without loading a model. Freeze all new adapter modules together, including `constructive_code_wider_adapters_20260921.py`, `constructive_code_reserved_adapters_20260921.py`, and `constructive_code_holdout_extension_20260921.py`.

## Source admission and interpretation

The 50 admitted tasks come from 56 audited candidates. All six quarantines and original failures remain in the merged dataset: `1352_F` (released output-checker regex incompatibility), `1497_C2` (released output-checker compile error), and `472_A`, `1047_A`, `1088_A`, `1196_C` (source-reference rejection rate below the preregistered 0.9 gate). No task was replaced using model performance.

The tasks are human-authored Codeforces problems. Released checker and validator sources come from [CodeContests+](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus), pinned at `96c850540fade31d384a25766461e0da6b08f5fc`; original test inputs come from [CodeContests-O](https://huggingface.co/datasets/caijanfeng/CodeContests-O), pinned at `1a765191567b429f633bbd1c6e67b5890dfaf267`. Generated checker/test infrastructure should not be described as human-written. Source statements, original/effective hashes, faithful notation repairs, source rows, original suites, excluded invalid inputs, reference programs, and admission receipts are retained.

Each task is audited with 12 source-labeled correct and 12 source-labeled incorrect Python programs, original checker binaries, independent input constraints, and test-only semantic-equivalence probes. Admission requires TPR and TNR at least 0.9; this does not prove program correctness beyond a finite suite. `545_B` and `710_C` retain narrow, independently reviewed, hash-bound input-validator exceptions: broken generated regex parsers are replaced only for source-input admissibility by exact problem constraints. Original reward checker bytes are unchanged. All original source inputs for those two tasks are retained. For `710_C`, ordinary single-integer whitespace semantics are explicit; this is not a claimed byte-identical repair of strict `readLine` parsing.

Modes represent canonical execution witnesses across the fixed source suite, not algorithm classes. Formatting, anonymous partition labels, unordered witness serialization, and stated redundant quantities are quotiented. Named coordinates and sequence positions are preserved. For `1092_A`, character permutations are additionally quotiented: only frequencies of the named letters matter to every task constraint.

The largest heldout matrix task, `1520_C`, emits 4,156,363 integer cells across its 36 source inputs for one accepted program, before the independent stability rerun. Account for CPU verification inside any GPU allocation; do not infer full-study GPU cost from generation throughput alone.

## Separate stress diagnostic

The five-task fixture is `var/artifacts/codecontests_out_of_reward_stress_20260921.json`, SHA256 `3e7c34e2d8b2e3bb6a5d4b329525bffaeced7e8fb2791f2d09f257bca65513cb`. It contains independent, constructed counterexamples for `1016_D`, `1360_G`, `1408_A`, `1323_A`, `1038_B`. These are audit-only probes, outside the primary reward and metrics. Each is bound to unchanged source checker bytes, an accepted known-correct program, and a source-labeled incorrect program that passed the original finite suite but is rejected on the counterexample. No policy outputs informed probe selection.

The helper `ops/evaluate_constructive_code_stress_20260921.py` checks response/attempt sidecar hashes and request identities, selects only original reward-accepted programs for covered tasks, and evaluates each on the separate diagnostic suite. It uses the same sandbox and accepted-program stability rerun. Reference-control smoke passed all ten programs: five positives accepted, five negatives rejected, zero hard failures (`var/artifacts/codecontests_stress_helper_smoke_20260921.json`).

Run from the repository with a fresh diagnostic output and unique local work paths:

```bash
PYTHONPATH=ops:src var/seed_paper_eval/paper310/bin/python \
  ops/evaluate_constructive_code_stress_20260921.py \
  --evaluation /absolute/path/to/completed/evaluation.json \
  --stress-manifest var/artifacts/codecontests_out_of_reward_stress_20260921.json \
  --stress-manifest-sha256 3e7c34e2d8b2e3bb6a5d4b329525bffaeced7e8fb2791f2d09f257bca65513cb \
  --output /absolute/path/to/fresh/out_of_reward_stress.json \
  --runtime-root /unique/job/work/runtime \
  --launcher /unique/job/work/launcher \
  --build-root /unique/job/work/checkers \
  --scratch-root /unique/job/work/scratch
```

Use a current/frozen diagnostic code snapshot containing all three adapter modules; an older initial-pilot source snapshot lacks the later heldout module. The input evaluation's frozen task data and exact source checker/suite identities remain binding. Run the same fixed fixture on deeper base and both final arms. Report primary accepted count, covered accepted count, stress survival, and failures separately. Conditional stress survival can change because the set of reward-accepted programs changes; it is not a substitute for the primary paired metric.

## Reproduction

`build_constructive_code_wider_20260921.py --phase fetch|audit` materializes and audits `--cohort initial`, `reserved`, or `heldout-extension`. Each cohort needs a separate output directory. The original reservation and additive extension reservation are under `var/data`; aliases are split by normalized statement identity. `--phase resolve-validator` records the independently reviewed `545_B` exception; `resolve_constructive_input_validator_20260921.py --root RESERVED_ROOT` records the `710_C` exception. Both preserve pre-review failures. `merge_constructive_code_pools_20260921.py --root INITIAL --root RESERVED --root EXTENSION --output FRESH_LARGER_ROOT` copies validated task data, retains source manifests, and assigns the fixed 32/2/16 split.

The first completed 23×32 broad-screen stress run tested 14 of its 74 primary accepted programs. `1408_A` survived 1/3, `1323_A` 6/6, and `1038_B` 1/5; the other two covered tasks had no primary accepts. Six programs therefore demonstrably fail on additional valid inputs, at least 6/74 (8.1%) of all primary accepts. This is a targeted lower bound, not an unbiased error estimate. No hard violations occurred. Results and bound receipts are in `var/artifacts/real_domains_pilot_20260921/code_capability/out_of_reward_stress.json` and its `.attempts.jsonl` sidecar. Preserve this limitation when interpreting witness diversity and apply the same fixed diagnostic to the deeper base and both final arms.
