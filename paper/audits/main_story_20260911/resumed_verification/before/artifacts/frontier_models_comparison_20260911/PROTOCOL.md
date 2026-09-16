# Hosted model comparison protocol

This extends the completed `frontier_modebench_gpt56sol_20260911` run. Each new model receives the identical frozen 1,920 evaluation prompts (128 prompts in each of five domains and three levels), with eight stateless requests per prompt. No training, tools, demonstrations, or conversation history are added. Original system and user strings, graders, and the existing formatting-only secondary normalizer are frozen before collecting new benchmark results.

The seven original-protocol deployments are gpt-5.6-sol (reference), gpt-5.4, grok-4.3, DeepSeek-V4-Pro, FW-Kimi-K3, claude-opus-5, and claude-opus-4-8. Each cohort targets all 15,360 draws (107,520 intended draws overall). Both Claude deployments use the same original frozen prompts. Live completion and audit status are reported in STATUS.md; enrollment in this protocol does not imply completion. Separate changed-prompt Opus 5 diagnostic cohorts are excluded from these seven cohorts and their main comparison.

All new deployments accept the requested medium reasoning control in a non-benchmark arithmetic preflight. GPT uses Responses `reasoning.effort`; Claude uses native Messages `thinking.type=adaptive` and `output_config.effort`; Chat deployments use `reasoning_effort`. DeepSeek's extra `thinking` request argument is rejected by this Azure endpoint, so it is omitted; the accepted medium request exposes reasoning_content. Provider labels do not establish equal compute. The native requested output limit is 8,192, and returned usage is preserved in its original schema because reasoning-token accounting differs. Unreported temperature/top_p defaults are not inferred. No token or monetary comparisons should assume equivalent accounting.

Kimi's native `n=8` arithmetic preflight returned HTTP 200 but only its first choice answered the supplied question; the remaining choices contained unrelated long outputs, six ending at the length limit. The complete anomalous receipt is retained in `preflight_kimi_n8_medium.json`. This probe is excluded from evaluation. The benchmark uses single-choice requests, matching the reference experiment, with paced requests below the observed 100 requests/minute quota. No grouped-generation results are included.

Preflight bodies, native responses, non-secret response headers, and usage are saved separately from benchmark draws. API credentials are read through hidden terminal input and are never saved in artifacts. Benchmark attempts, requests, raw outputs, grading receipts, errors, code snapshots, dataset identities and summaries are retained in model-specific directories. API failures are retried and remain excluded from model accuracy; received truncated outputs remain sampled draws.

Interpretation: This tests concentration in hosted inference. It does not identify how training produced the distribution, establish inaccessible modes have zero probability, or isolate a causal effect of parameter scale. Level labels were calibrated elsewhere and may not be monotonically harder for each hosted model. Prompt directives and varying support sizes constrain level comparisons. Countdown's declared expression-library count is not the verifier's full accepted support, so a finite uniform-support reference is omitted there.

API references (checked 2026-09-11):
- https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry
- https://platform.claude.com/docs/en/build-with-claude/effort
- https://developers.openai.com/api/docs/models/gpt-5.4
- https://docs.x.ai/developers/models/grok-4.3
- https://api-docs.deepseek.com/guides/thinking_mode/
- https://docs.fireworks.ai/api-reference/post-chatcompletions

Actual Azure preflight receipts take precedence over generic provider documentation about accepted arguments.

DeepSeek protocol recovery (2026-09-11): two native HTTP-200 receipts returned no final answer or stop reason, nonempty reasoning, and usage:null. They remain saved as provider protocol failures, are excluded from accuracy, and are retried without altering the request. A separately versioned recovery runner recognizes only this exact malformed shape; other unexpected native shapes still stop collection. Original source, manifest, existing grades, and raw receipts remain unchanged. Unknown usage is not imputed as zero. Runtime concurrency may increase to192 and transport timeout to600seconds, with unchanged8192-token native generation budget and medium reasoning profile. Details and source hashes are in the DeepSeek run’s PROVIDER_PROTOCOL_RECOVERY.md and provider_protocol_adapter.json.
