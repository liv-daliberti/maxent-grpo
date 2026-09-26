# Amendment 2: extend to the completed scientific census

This amendment is frozen after inspection of the initial 135-block conditional-concentration analysis and before reading newly admitted response samples or computing concentration effects for the completed cohort. The first analysis remains a retrospective secondary analysis. This extension is explicitly after results; it is not preregistration or an independent confirmatory replication.

## Reason and fixed population

A parallel manuscript completion refresh replaced the dated training-curve census while the first conditional analysis was running. The first sample cache had already bound the earlier snapshot, with 74 Level-1 Dr.GRPO replay pairs and 67 MaxRL replay pairs. The current primary paper now has 75 MaxRL pairs, including five paired seeds in every 3B domain. The difference reflects later source admission/completion, not collision-effect-based selection.

Use EVERY cell of the newly frozen completed-cohort snapshot, including unfavorable, undefined, and conflicted records. Preserve the first snapshot, sample cache, collection receipt, code receipts, and initial 135-block result unchanged. Report both cohorts and the reason for extension. Do not silently describe the 67-pair initial analysis as the current complete primary cohort.

## Prespecified source changes

The metadata comparison identifies 13 newly available step-3072 checkpoints: 11 Level-1 Qwen3B MaxRL/ReplayMaxRL endpoints and two Level-2 Pantry endpoints. An additional Level-1 MathIR MaxRL seed72 endpoint was already cached but now receives terminal admission. The added Level-1 paired seeds are Countdown72,73; MathIR72,73; Pantry70,71,73,74. Level-2 Pantry adds a MaxRL pair at seed43. Its ReplayDr.GRPO seed44 endpoint is independently available without adding a paired seed.

Withdraw the previously available Level-2 Pantry ReplayMaxRL seed46 initial checkpoint from the completed-census analysis: the new census authorizes another source that conflicts at step0. Preserve the old-census measurement in the initial cache and explicitly report the changed integrity status. Do not select one conflicting attempt by its outcomes.

Retain cached compact checkpoint payloads only when the complete frozen raw-draw metric, metadata, and origin records are exactly equal between snapshots. Re-ingest every newly available or changed complete checkpoint through exact frozen prefix hashes and the existing raw-response normalizer. No new model calls, training, checkpoint selection, grading, or changed outcome definitions are permitted. The 13 new checkpoint origins occupy approximately 640 MB of frozen prefixes; no full historical rescan is needed.

## Sampling and analysis rules

Retain amendment 1 and the original estimator/eligibility rules unchanged. Audit the nominal child-stream mapping and actual neutral request metadata on new samples. Empty historical request-seed lists and exact singleton [draw_parent_seed] lists are compatible with neutral n8 sampling when all option IDs are null. Other options or request seeds require explicit unsupported status. Do not infer 32 independent streams: vLLM n-sampling uses parent seed plus sample index, ordinarily yielding 11 distinct nominal child seeds across these four adjacent parent seeds. The parent seed is also reused across prompts. Distinct nominal streams do not prove iid observations; missing historical dependency identity remains a stated assumption.

Choose nominal-stream representatives only by the already fixed earliest draw/sample rule. Duplicate-key disagreements are provenance diagnostics, not grounds for outcome-selected removal. No change to thresholds, orientations, statistics, uncertainty rules, or included domains is permitted in response to the initial effects.

## Outputs

Create a separate verified_samples_completed_cohort.jsonl.gz and collection_receipt_completed_cohort.json. Preserve conditional_concentration_20260911_initial.json before any updated result replaces the main result path. Bind the completed snapshot, source-prefix identities, old and new cache hashes, amendment text, source code, and all added/removed checkpoint records. The extension program performs only source validation and reconstruction of existing P,D,M metrics; the parent analyzer computes concentration effects in a separate explicit invocation.
