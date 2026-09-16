# Current ModeBench prompts

The main version for new Level 3 work is the **neutral-Python revision**, using **`python_level3_neutral_v1`** for Python training and inference. Countdown, Graph Coloring, MathIR and Pantry retain their admitted V3 data and registered wording. Other levels retain their registered interfaces. The shared Python interface is [`ops/modebench_current_contract.py`](../ops/modebench_current_contract.py). This prompt choice was made after inspecting the original/neutral inference and coarse-key comparisons on September 11, 2026.

The exact system prompt is:

> Solve the executable constraint problem carefully. You may reason briefly, but end with exactly one final answer inside \boxed{}. Construct one allowed lambda expression. Output exactly the boxed lambda.

It removes the small-divisor example and dispatch suggestion. The default renderer preserves the supplied task text and matches every one of the 32 stored neutral Level 3 Python prompt pairs exactly. The current dataset calibration separately revises the task text to put legal syntax restrictions first; legal outputs, verifier, syntax constraints and response cap remain unchanged.

For new inference, import `make_messages` and `profile_metadata` from `modebench_current_contract` (with `ops` and `src` on the Python path). Save the returned metadata with the run. `registered_hints_v1` is the explicit historical condition.

For new Python Level 3 training, use `neutral_python_training_environment(native_environment, campaign="e122")` from [`ops/modebench_current_training.py`](../ops/modebench_current_training.py); select `campaign="e124"` for the registered 7B recipe. Supply the complete native recipe, including its registered Python template. This binds neutral wording, admitted train/eval data, and both tested immutable runtime paths. It authenticates the runtime inventory and fresh admission before returning a configuration. Keep the returned `OAT_ZERO_SOURCE_ROOT` and `OAT_ZERO_OPS_SNAPSHOT_ROOT` when submitting the job. The E124 systems-qualification gate still applies.

For inference, use `neutral_python_dataset()` from [`ops/modebench_current_data.py`](../ops/modebench_current_data.py) and `make_messages` from the prompt contract. The selected native template is `qwen_level3_python_factors_neutral_v1`; `qwen_level3_python_factors` aliases the same neutral renderer. The historical `qwen_level2_python_factors` template remains versioned.

Use a new run directory and prompt-version identity. A frozen source snapshot must contain the selected template; changing a working-tree default does not modify already-submitted jobs or their snapshots.

## Dataset calibration and admission

The active [V5 calibration](../var/artifacts/modebench_level3_neutral_v5/registration.json) retains the measured Level-1 Qwen2.5-0.5B targets of 0.2109375 pass@1 and 0.76953125 pass@8, with absolute tolerances 0.04 and 0.08. The reference keeps its original hinted prompt.

All 21,248 development responses are complete. The once-selected mixture has weights `[0, 2, 17, 1]` across the four registered numeric strata; its selected development split achieves pass@1 0.236572265625 and pass@8 0.71875. Both forecast gates and both selected-subset gates pass. The new dataset is `var/data/modebench_level3_matched_neutral_v5`, with 384 train / 128 dev / 128 eval rows per domain; non-Python domains are byte-identical to admitted V3. Dataset construction was recovered with the same recipe and seeds after a seed-program certification failure; the original partial construction and error are retained. All 1,024 reconstructed train/eval seed-program certifications passed with unchanged verifier and timeout.

Fresh confirmation **passed** on 2026-09-12 UTC: **pass@1 0.223388671875 and pass@8 0.705078125**. Both original tolerances pass, and all 4,096 responses reproduced exactly under independent external regrading. The [admission](../var/artifacts/modebench_level3_neutral_v5/admission.json) binds the full receipt and dataset identity. This is an observed numerical match to the fixed historical reference, not statistical equivalence: the point differences are +0.012451171875 / -0.064453125, with prompt-bootstrap 95% intervals [-0.022216796875, 0.0498046875] / [-0.1171875, -0.013671875].

Array 31254312 used one independent draw per GPU, retaining all original effective request/child seeds; three fake-engine tests verified serial/sharded receipt equivalence and tamper rejection. Job 31254334 performed exact regrading and initiated the registered 22-job held migration. The original failed confirmation remains a separate negative result. Admission removes the neutral-difficulty block; normal E122 resource gates and E124's separate systems-qualification block remain in force.

## Registered training state

The September 11 registration audit found no training in the registered Python Level 3 cohort: E122 has 20 held jobs with zero elapsed time, no restarts and no output directories; E124 has two such held jobs; E123's 20 Python jobs remain prospective. Other Level 3 domains have already started training. The reported coarse-key sensitivity used Level 2-trained checkpoints evaluated on Level 3.

The [original prompt registration](../artifacts/modebench_level3_neutral_default_20260911/registration.json), frozen snapshots and [first migration](../var/artifacts/python_level3_neutral_migration_v5_20260911/committed.json) remain preserved. That first migration created E122 jobs 31254498–31254517 and E124 jobs 31254518–31254519. Four released E122 jobs then failed before training because the frozen CLI did not register the neutral template; they were held to stop the retry loop. No scientific run directories or optimizer steps existed in any of those 22 jobs.

The [runtime repair](../var/artifacts/python_level3_cli_recovery_20260912/committed.json) is committed. It changes only argument admission in new immutable successors: the CLI accepts the neutral template, the Python validator accepts its registered syntax, and other-domain mismatches are rejected. Four native tests pass using the training launcher's native-library path. Every source file except `src/oat_drgrpo/args.py` is byte-identical to its parent runtime. Data, renderer, verifier, objectives, seeds and budgets remain bound to their existing versions.

All 22 prior jobs are verified cancelled. Current IDs are **31259067–31259086** for E122 and **31259087–31259088** for E124. The four seed-43 E122 treatments, **31259067–31259070**, have completed initial evaluation and produced positive training steps with finite loss/gradient metrics and zero restarts. The [retained optimizer evidence](../var/artifacts/python_level3_cli_recovery_20260912/FINAL_STATUS.json) records drgrpo 32, replay_drgrpo 28, maxrl 11, replay_maxrl 32. Their later seeds remain in the held queue. The two E124 successors remain held under the separate failed systems-qualification gate.

Current state lives in the repair directory's `e122_jobs.json`, `e124_transaction.json`, and `e122_status.json`. CPU controller **31259141** is running with a clean 18-running / 11-completed / 71-held status after the finite expansions. The [storage-aware continuation](../ops/exp_scaling/resume_python_level3_cli_storage_20260912.py) reads the migrated queue and preserves the ordinary E122 cap of four. Explicit finite capacity expansions are separately registered; they include all pending inference writers in their storage budgets. Existing paper cohorts retain their recorded historical prompts and datasets.

## Interpretation

The existing difficulty-matching certificate measured the hinted interface. The completed V5 calibration measures the neutral Python interface against fixed historical Level 1 targets. Its passing gates establish an observed numerical match under these explicitly different interfaces; they do not establish a causal comparison of difficulty alone or statistical equivalence.

Retain the original-wording result alongside the new default. The prompt choice is prospective for training, but it is informed by existing evaluation outcomes. Original divisor-vector keys remain the primary outcome identity, and unordered factor-pair keys remain a sensitivity analysis. Neither key identifies the algorithm implemented by a Python program.

## Historical recalibration record (September 11)

The migration now includes revising the Python Level 3 dataset under the neutral system prompt. Its fixed measured Level 1 target is pass@1 0.2109375 and pass@8 0.76953125, retaining the 0.04/0.08 tolerances. The historical target used its registered hints; this is a fixed-reference match, not a same-interface comparison.

The incomplete numeric-only pilot is preserved under `var/artifacts/modebench_level3_neutral_v1`. It produced almost no legal programs and was retired before fitting or confirmation. Revision v2 puts the legal syntax restrictions first in the task text, without a solution example or algorithm hint. Its neutral system prompt, verifier and exact mode-count histograms remain unchanged. Four fresh development pools are running as array31252912; CPU31252941 continues a passing development fit through fresh held-out confirmation. The registration is `var/artifacts/modebench_level3_neutral_v2/registration.json`. No neutral difficulty match or training migration is claimed yet.

At the user's request, four additional non-Python E122 jobs (31158685–31158688) were released. All eight E122 jobs were confirmed running, while all20 Python jobs remained held. The finite burst reserved2513GiB against about3376GiB free, including every pending inference task. Its evidence is `var/artifacts/e122_nonpython_burst_20260911/post_start_verification.json`. This is a finite expansion; existing subsequent-release safeguards remain in force. E124's earlier failed systems qualification remains a separate release block.


## Historical failed fixed-candidate confirmation (2026-09-12 UTC)

At the user's request, a separate fresh confirmation is now running in array
31253481, with verifier-audit job 31253494 dependent on completion. Registration:
`var/artifacts/modebench_level3_neutral_fixed_confirmation_20260912/registration.json`
(SHA-256 `b64a06692f5023fb230deb71eb2993836e2393c8ab2002dfade22524c25f3a0a`).
All 384 training, 128 development and 128 confirmation rows were generated and
frozen before sampling; historical data and all then-present pilot cases were
excluded. Every split preserves the exact Level-2 solution-count histogram;
the four other domains remain byte-identical to historical Level-3 V3.

This is one prospectively fixed 17:3 common-factor-three/all-even numerical
mixture with neutral proper-divisor wording. It is a direct independent check,
separate from the V2 full-development fit (which did not pass). It does not
claim a passing fitted development recipe or create automatic admission or
training releases. The 128 full confirmation problems receive four independent
groups of eight responses (4,096 total), followed by external regrading. The
fixed reference remains pass@1=.2109375 and pass@8=.76953125, with .04/.08 absolute
tolerances. The recipe and confirmation subset will not be changed in response
to these outcomes. Results belong to this new version only.


Fresh confirmation completed at 2026-09-12T01:46:26Z. All 4,096 saved responses
reproduced exactly in the unchanged external verifier on local reconciliation;
the initial cluster audit failure and diagnostic attempts are preserved. The
final report is `var/artifacts/modebench_level3_neutral_fixed_confirmation_20260912/confirmation_report.json`.
Observed pass@1=.194580078125 satisfies its tolerance, but pass@8=.583984375
falls below .76953125 by .185546875, exceeding the .08 tolerance. This candidate
is not admitted and has not migrated or released Python training jobs. These
confirmation outcomes must not be used to change this candidate's recipe or
select a different confirmation subset.


Full prospective V5 development is now registered at
`var/artifacts/modebench_level3_neutral_v5/registration.json` (SHA-256
`13d74095bbb3e73c271c2c5bf6498d1d0db3924501ecab2a009a69b33e5bbd1a`)
and submitted as array 31253768. It scores 166 fresh problems in each of four
predefined numerical strata with four independent groups of eight responses
(21,248 responses). The fit uses only these new development outcomes, checks
both support-weighted split forecasts, and scores one deterministic selected
development subset. All earlier confirmation cases are excluded. A passing
fit proceeds to a new 128-problem held-out confirmation; only a passing fresh
admission can trigger the prepared replacement of 22 still-held Python jobs.
The continuation retains all original release caps and qualification gates and
does not release Python training automatically. Its source and migration code
are separately pinned in `automatic_continuation_registration.json`.
