# Evaluation diagnosis before expansion

The initial pilot does not establish that Re:Max is worse. The subsequently corrected 128-update QA diagnostic gives positive same-HF discovery and topic-balance differences on its fixed eight training and 32 development documents, while retaining a weak fresh-learning signal and single-seed limitation. The coding decrease in expected distinct @32 is small, concentrated in a few tasks, and compatible with finite-sample variation. The evaluation metrics and frozen checkpoint identities check out. We did find measurable numerical differences between HF training/scoring and vLLM evaluation that should be controlled before interpreting small treatment effects. No larger study or reserved-test evaluation was run for this diagnosis.

## What the existing evidence establishes

The independent final audit binds all three endpoints within each domain to the same model revision, prompts, response budget, sampling settings, verifier, source/data bytes and selected task IDs. Each trained endpoint loads its own sealed checkpoint 32 adapter. The MaxRL and Re:Max names and weight hashes are not swapped. See `independent_hardened_final_v2/summary.json`, SHA-256 `7f69750aeb09c2e938e093ed8eda550566c6203369205060d98cafd7bdc04430`, under `var/artifacts/real_domains_pilot_20260921/`.

The QA MaxRL training arm has zero fresh and replay gradients and unchanged adapter weights. Its vLLM endpoint is **token-for-token identical to base for all 6,144 requests**. Re:Max differs on 90 requests. This is a useful zero-adapter control; it is not a numerical equivalence test for nonzero adapters.

Stopping does not explain the trained coding decline. Base, MaxRL and Re:Max have respectively 5, 5 and 7 length-limited responses among 2,688 attempts. All are rejected and all belong to untrained development tasks. Every QA response contains two tokens and stops normally. These counts and response-file hashes are retained in `evaluation_diagnosis_v1/existing_endpoint_decoding.json`.

Training's HF generation uses temperature 1, top-p 1, unrestricted top-k, repetition penalty 1, both Qwen EOS IDs 151645/151643, and the tokenizer-supported vocabulary. The vLLM evaluator uses equivalent sampling settings and vocabulary masking. The installed vLLM 0.8.4 generation-config update imports EOS IDs without replacing the explicitly supplied sampling parameters. Qwen's packaged temperature/repetition defaults therefore do not silently explain the result. HF and vLLM use different RNG implementations and request seeding; sampled outputs across these engines are not expected to match by seed alone.

Relevant code: [HF generation](../ops/train_real_domains_pilot_20260921.py), `generate_samples`; [endpoint evaluator](../ops/evaluate_real_domains_20260921.py), `verify_lora_checkpoint` and `run`; [independent auditor](../ops/summarize_real_domains_pilot_20260921.py), `audit_endpoints`.

## Confirmed precision difference and its measured size

All 392 saved adapter tensors in each inspected QA/coding Re:Max checkpoint are FP32. PEFT's native forward path casts the adapter input to its parameter dtype. vLLM's default LoRA dtype follows its BF16 base model, and it casts the saved adapter matrices to BF16. The rank 16, alpha 32 scaling factor is two in both implementations; no scaling mismatch was found. Matching file hashes does not imply matching floating-point forward results.

We scored exactly the same saved token sequences from the union of discovered replay exemplars: 25 coding exemplars across seven trained tasks and 20 QA exemplars across 15 trained documents. Base and checkpoints 16/32 for both arms were evaluated using HF native adapters, HF adapters explicitly cast to BF16, and vLLM BF16 adapters. Configs, source snapshots, checkpoint seals and adapter bytes, exact row coverage, token echoes and score sums are independently bound by [the comparison script](../ops/compare_real_domains_likelihood_20260921.py).

The following numbers are changes in **mean per-token log probability**, averaging exemplars equally. They are not native mode probabilities.

| Domain and engine | MaxRL32 minus base | Re:Max32 minus base | Re:Max32 minus MaxRL32 |
| --- | ---: | ---: | ---: |
| Coding, HF native | +0.00302221 | +0.00388416 | +0.00086195 |
| Coding, HF BF16 adapters | +0.00293981 | +0.00438215 | +0.00144234 |
| Coding, vLLM BF16 adapters | +0.00320649 | +0.00432610 | +0.00111961 |
| QA, HF native | 0 | +0.01091056 | +0.01091056 |
| QA, HF BF16 adapters | 0 | +0.01091332 | +0.01091332 |
| QA, vLLM BF16 adapters | 0 | +0.02575464 | +0.02575464 |

Across coding exemplars, the mean absolute difference between the vLLM and HF-native **Re:Max-minus-MaxRL contrast** is 0.00160504, larger than the small signed mean treatment contrast. Casting HF adapters to BF16 alone changes this contrast by mean absolute 0.00146837; residual vLLM-versus-HF-BF16 disagreement is 0.00169369. These are numerical sensitivities, not estimates of metric bias.

For QA, vLLM-versus-HF-native treatment-contrast disagreement is mean absolute 0.04337791 and maximum 0.34953532. HF adapter casting alone has mean absolute 0.02506876 even though its signed mean nearly cancels. Residual vLLM-versus-HF-BF16 disagreement is 0.04340283. The largest difference concerns the competing `earn`/`acq` answers to `sata_news_0778526c7854b130`. Adapter precision is therefore not the sole source of cross-engine variation. Attention kernels, batch shape and prefill-versus-incremental decoding remain plausible contributors; this panel does not isolate them.

vLLM `prompt_logprobs` normalize before sampling-time vocabulary processors, so the comparison correctly uses HF **raw full-vocabulary** probabilities. The HF masked-minus-raw token log-probability difference is at most 3.82e-6 for coding and exactly zero for QA on this panel. That normalization distinction is immaterial to the observed engine differences here. The vLLM diagnostic disables prefix caching, verifies the full prompt-plus-response token echo, and discards the single generated token required by the API. It does not add a task evaluation. Its 11 CPU tests cover offset alignment, EOS inclusion, missing tokens and malformed scores.

Machine-readable comparisons are `evaluation_diagnosis_v1/code_engine_parity.json` and `qa_engine_parity.json`. Their `status=pass` means binding/arithmetic passed, **not** that HF and vLLM are numerically equivalent. They include every exemplar's checkpoint contrasts and all input/source hashes. Raw diagnostic token probabilities remain in the separately frozen `diagnosis_{code,qa}_{likelihood,vllm}_v1/diagnostic.json` files.

## Metrics and uncertainty

Expected distinct @k sums `1 - choose(N - n_m, k) / choose(N, k)` over accepted modes, where N includes failures. PCMD conditions on accepted samples and computes `1 - sum n_m(n_m-1) / [C(C-1)]`, requiring C≥30. Thus an acceptance change, rare-mode sampling fluctuation or probability redistribution can lower expected distinct while conditional PCMD rises. This is not an algebraic contradiction. Coding mode keys identify canonical executable witnesses across the suite, not algorithms. QA mode keys are native annotated topics, including overlapping taxonomy categories.

Only three coding tasks meet the PCMD threshold in both arms. A small conditional increase there does not establish a whole-cohort diversity benefit. The trained-task expected-distinct@32 decrease of 0.09086 is concentrated in 1038_B, 988_A, 1323_A and 1454_A; other tasks offset some of it. The untrained-development mean decrease of 0.01454 is almost entirely 244_A moving from two accepted responses to one, with only one observed mode in each arm. No untrained coding task exhibits multiple modes in either trained endpoint, so this portion of the result is not evidence of mode collapse.

The separate dynamics analysis recomputes metrics from all original attempts. A paired request-index resampling sensitivity analysis with 4,000 draws gives trained coding expected-distinct difference percentiles [-0.21547,+0.05140] and untrained [-0.05924,+0.02550]. These are conditional sampling diagnostics, not training-seed confidence intervals; they cannot recover unseen modes and may reflect rare-mode plug-in bias. One training seed still provides no estimate of variation across training seeds. See `training_dynamics_diagnosis_v1/diagnosis.json` for the complete decomposition.

## Original diagnostic plan, followed by completed checks below

Use the same HF implementation and native adapter dtype for a fresh, matched base/MaxRL/Re:Max evaluation on the already-used development cohort, with fixed prompts, existing verifier and matched sampling budget. This removes the avoidable engine/precision difference from the treatment comparison. Preserve the current vLLM results as a separate endpoint protocol; do not replace or pool their response files.

Before further training, complete the independent check of padded old-policy scoring versus trimmed new-policy scoring. If that check finds a systematic discrepancy, correct it in a new source version and require initial importance ratios near one under the exact same scoring procedure. This evaluator audit does not itself establish that issue or its magnitude.

For QA, exact teacher-forced scores of every listed answer on development prompts would additionally distinguish redistribution among annotated topics from sampling variation. Count only the evaluated response forms; scoring one `letter + EOS` string is not automatically exhaustive probability for a mode that also admits surrounding whitespace.

The positive average Re:Max bank-exemplar likelihood changes in both engines argue against an ignored or sign-reversed replay term. They do not show that every rare mode gains mass, that an entire mode gains mass beyond its one exemplar, or that held-out diversity improves. The defensible diagnosis is a weak, noisy pilot effect with a measurable evaluation precision difference—not a demonstrated failure of the method and not a reason to launch a larger study yet.


## Independent review of the same-HF exact-format QA follow-up

The completed `diagnosis_qa_exact_format_v1` follow-up scores all six listed letters followed by each of the two actual stopping EOS tokens, for all 48 previously evaluated prompts and the base/MaxRL32/Re:Max32 models. The independent review verifies all **576 unique two-token forms and 1,728 native-precision scores** against the source records, original rendered prompts, tokenization, train/dev splits and original checkpoint32 seals. Every source-positive native topic is represented; incorrect topics remain in the enumerated total mass. The cohort is exactly the original 16 trained and 32 dev documents, with no reserved test included. First-token probabilities agree across the two EOS variants, as required by a consistent causal prefix.

No actual-result bug was found. All enumerated probability sums lie between 0.9869836469 and 0.9999999285; none exceeds one. The reducer leaves the remaining mass unresolved. It does not normalize the listed valid topics into a probability-one distribution. Let p_m be the enumerated probability of correct mode m and r the total unresolved probability, including potentially incorrect strings. Monotonicity gives the discovery lower bound `sum_m [1-(1-p_m)^k]`. Its derivative with respect to each mode mass is at most k, so allocating at most r additional mass increases the sum by at most kr; the known source support gives the further cap. Taking treatment lower minus control upper, and treatment upper minus control lower, gives valid difference bounds for these fixed probabilities.

The independently recomputed Re:Max-minus-MaxRL expected-distinct@32 interval is **[-0.006268927, -0.004046002] on the 16 trained documents**, and **[-0.009454676, +0.030871774] on the 32 dev documents**. Expected distinct@8 increases on the trained documents, with interval [+0.006762181, +0.007717916]. The budget dependence is real under this scored distribution: an improvement at eight draws does not imply improvement at 32. Dev diversity remains unresolved, and these results support no broad positive claim.

These are bounds over unenumerated output mass under the fixed HF teacher-forced floating-point calculation. They are not training-seed uncertainty intervals or certified error bounds against every cache/batch/kernel implementation. They use the population discovery expression for independent repeated draws; the original endpoint estimator instead uses without-replacement subsampling of its finite response pool. Those are related estimands, but their numerical values should not be equated directly. `pcmd_conditioned_on_enumerated_correct_format` is explicitly restricted to the enumerated correct format and is not claimed to bound full-policy PCMD.

The review receipt is `evaluation_diagnosis_v1/qa_exact_format_independent_review.json`, SHA-256 `84a71a68efb2d52ad3d68229b725cf4919270c2245bbc0d7304cfa7b6630113d`; its exact CPU review script is preserved beside it. The three probability-bound property tests pass independently. That initial reducer assumed disjoint forms and complete known support in its input; the review checked those assumptions exhaustively for the completed receipt. The subsequent version explicitly rejects duplicate response forms. Native-label and complete-support checks remain part of the independent source audit before a new config is accepted.


## Why replay can improve balance and still reduce discovery

The full 16-document original QA panel supports a more specific mechanism than a single aggregate diversity number. Ten documents finish with only one discovered topic in their replay bank. On these documents, Re:Max-minus-MaxRL expected-distinct@32 lies in [−0.01004146, −0.00649998]. Five documents have multiple discovered topics: their expected-distinct@32 is effectively saturated, with a difference in [−0.00000090, +0.00001872], while their population PCMD improves by [+0.03804327, +0.03812256]. The remaining document has an empty bank and a small positive difference. These groups exhaust the original cohort; they are exploratory groups defined by post-training bank contents, not alternative primary denominators.

Fourteen source-positive topics are absent from both original banks. Five lose full native-topic probability under Re:Max even if every unenumerated output is assigned favorably when computing the difference bound. For example, an undiscovered `grain` topic falls from enumerated probability 0.00522011 to 0.00407012 while the bank contains the dominant `wheat` topic. Replay can increase a discovered answer's probability while reducing the probability of a valid answer that never entered its bank. Conversely, balancing already-discovered topics can improve conditional PCMD when discovery at 32 draws is already near its ceiling. This explains the differing metric directions without an evaluator or arithmetic failure; it does not establish a general causal effect across training seeds.

The complete decomposition, including every document and source hash, is `evaluation_diagnosis_v1/qa_original_exact_mode_bank_decomposition.json`. The added population-PCMD bounds retain all known native topics, including zero enumerated mass, and allow unresolved mass to be incorrect or assigned to any correct topic. Concentrating that mass on the largest topic minimizes conditional diversity; distributing it to equalize the smallest topic masses maximizes it. No known correct mass yields an undefined bound rather than a zero diversity claim.

## Corrected training integrity check

A separate versioned trainer now trims old-policy scoring to exactly the attention width used by the live scorer and requires microbatch size one. The original runs and reports remain intact. The new independent wrapper pins the corrected trainer, original audit helper, source snapshots and defaults; checks raw response/token bindings and strict verifier labels; reconstructs replay banks from an empty start; and checks checkpoint seals, paired initialization and the gradient accounting. It additionally requires exactly zero old/new token-log-probability difference extrema and clipping fraction at every update. These are checks before each update, not claims that the optimizer leaves the policy unchanged.

The corrected eight-update coding check passes all gates with seven mixed-reward groups in each arm. This establishes that the correction works while fresh gradients are active; eight updates do not establish an efficacy result. The corrected QA comparison passes all gates for 128 updates and 2,048 fresh rollouts per arm. MaxRL has four mixed groups and Re:Max five; all remaining groups are entirely successful. Both final banks contain 16 native topics across eight documents. MaxRL has no replay gradient, while Re:Max applies replay on all 128 updates. The same-HF endpoint covers these eight trained documents and all 32 development documents, with six checkpoint seals and a base control; its final 128-update contrast was fixed before scores were available.

The corrected training receipts are `corrected_training_audit_v2/code_pair.json` (SHA-256 `5b60d800dadefef30dfd636843d8a05a79eaff16bb4708592ac4bbf4c6a3ca20`) and `qa_pair.json` (SHA-256 `5a5303b9a510db41aa0a219f0ff07e80cc16de4fbaaa4d2a6ea9c86c0717028e`). Separate scheduler receipts verify completed jobs and count 0.35 GPU hours for the coding pair and 0.762222 GPU hours for the QA pair. The low number of mixed groups remains a limitation even after selecting three documents that had mixed rewards in the original baseline screen. Reserved test documents were not used.


## Completed corrected QA endpoint review

The fixed final-128 contrast passes the independent review. All 40 source questions, 480 distinct letter/EOS forms and 3,360 native-adapter scores match the frozen request. Every native gold topic, original prompt token sequence, train/dev split and all six checkpoint seals agree with the source and training receipts. The review independently recomputes accuracy and discovery bounds, and uses a separate bisection implementation to verify the population-PCMD upper bound. It checks all four predeclared cohorts and all three fixed checkpoint steps, without selecting a favorable checkpoint after scoring.

| Fixed cohort | Questions | Re:Max minus MaxRL ED@32 bounds | Population PCMD difference bounds |
| --- | ---: | ---: | ---: |
| All trained documents, primary diagnostic | 8 | [+0.185292, +0.186057] | [+0.256933, +0.256978] |
| All development documents, primary diagnostic | 32 | [+0.018335, +0.020225] | [+0.018163, +0.018271] |
| Predeclared reward-learning documents, secondary | 3 | [+0.486066, +0.487904] | [+0.428991, +0.429098] |
| Predeclared retention controls, secondary | 5 | [+0.004828, +0.004949] | [+0.153698, +0.153706] |

Final success-probability differences remain bounded across zero: [−0.00000822, +0.00002499] on training documents and [−0.00000781, +0.00005152] on development documents. The same-HF trained MaxRL endpoint has ED@32 in [1.813943, 1.814708], versus [1.998931, 2] for base and essentially 2 for Re:Max. Both final banks contain the complete two-topic support of every training question. The result therefore supports balancing and retention of discovered modes in this selected diagnostic; it does not test protection of undiscovered training modes.

The final128 endpoint remains the primary contrast. The fixed32 and64 contrasts are retained in full: trained ED@32 differences are respectively [+0.150111, +0.153729] and [+0.180237, +0.181009]; development differences are [+0.001841, +0.007157] and [+0.020855, +0.025511]. The complete artifact also preserves all success, ED@8 and PCMD results for every cohort and checkpoint.

These intervals bound unresolved output strings under the **native HF teacher-forced numerical policy**. They are not training-seed confidence intervals or enclosures of all cached-generation, attention-kernel or batch-shape variants. The lowest enumerated mass across the entire seven-checkpoint panel is 0.515809 in a base-model reward-learning question; that missing mass remains explicit. Final128 mean residual mass is much smaller: 0.00002466/0.00000854 for MaxRL/Re:Max on training documents, and 0.00004940/0.00000993 on development documents. No probability renormalization is used to obtain the positive final difference bounds.

The planned mixed-reward learning gate still fails: only four MaxRL and five Re:Max groups were mixed, on only two of the three baseline-selected reward-learning questions. This is a positive mechanistic diagnostic with a fully discovered finite support and one training seed, not a confirmatory generalization result. Reserved test data and the unsubmitted larger experiments remain unused.

The endpoint audit is `corrected_training_audit_v2/qa_exact_format_results_review.json`, SHA-256 `9d8c1525513e7673e7259f7ffe8e0a99752606e8a39ecc371e58685ae5c8a03b`; its exact review source is saved beside it. It binds the frozen endpoint summary, SHA-256 `93c21582a9f062cf3aa7e500da3f5b3dc1773e43d08d3fc997d5888fd9c3deb6`, and the corrected paired-training/source/request audits. No original evidence was overwritten. The final independent accounting review checks all 12 completed diagnostic/correction jobs, including the superseded probe: 1.420833 additional GPU hours and **5.210278 GPU hours cumulatively**, against the original 200-hour cap. Its receipt is `corrected_training_audit_v2/diagnosis_accounting_independent_review.json`, SHA-256 `d0eacf018f4e5ebe2bb7c42bf3a17a436b269198dd54a90f6f6ae6f1405d671d`.
