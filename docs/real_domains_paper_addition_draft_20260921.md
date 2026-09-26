# Real applications: reviewable methods note

**Draft dated 2026-09-21. This note is not included in the paper or its appendix.** It records the application definitions, completed QA and hardened coding pilots, and prospective evaluation plan. The final combined independent audit passes for both pilots. The larger coding cohort has completed its sealed source audit. Dataset readiness and successful execution do not establish a Re:Max advantage.

## Human contest problems with executable witness modes

We study constructive competitive programming: a model writes a Python program that must produce an admissible witness for every input in a frozen verification suite. The problem statements originate in human programming contests. The actual evaluation assets come from [CodeContests+](https://arxiv.org/abs/2506.05817), supplemented by the pinned [CodeContests-O overlay](https://huggingface.co/datasets/caijanfeng/CodeContests-O/tree/1a765191567b429f633bbd1c6e67b5890dfaf267) and our documented source-derived counterexample inputs. CodeContests+ uses generated tests and custom checkers, so the accurate description is **human-authored problems with generated and supplemental verification assets**. It would be inaccurate to call the entire evaluation pipeline nonsynthetic. The pinned CodeContests+ release is [ByteDance-Seed/Code-Contests-Plus](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus/tree/96c850540fade31d384a25766461e0da6b08f5fc).

A mode is a canonical signature of a program's accepted output witnesses across the fixed suite. Canonicalization removes irrelevant presentation choices, such as ordering an unordered set or renaming anonymous groups, while preserving the task's witness identity. Different keys need not correspond to different algorithms; different algorithms can share a key. We therefore measure **verified executable-witness diversity**, rather than algorithmic or reasoning diversity. These signatures depend on the frozen suite, and counts from different suites are not interchangeable.

Admission requires all 12 known positive reference programs to pass and all 12 known negative reference programs to fail. The reference identities, source bytes, input suites, released checker, canonicalizer, supporting code and testlib are hash bound. Accepted policy programs undergo a second full-suite check, with stable verification and canonical mode required. These checks establish agreement with a finite reference panel and finite test suite; they are not a proof of correctness on every legal input or a guarantee of zero future false acceptance.

The historical development suite exposed a material limitation: a targeted stress test rejected 6 of 14 inspected programs that it had initially accepted, across five covered tasks. That targeted fraction is not an unbiased estimate of the suite's overall false-positive rate. The earlier paired coding runs were canceled and cannot support the corrected comparison. The hardened suite appends five fixed source-audit probes, preserves original prompts, checker and original input order, and reapplies the strict reference gate. These probes came from source auditing, not from tailoring new tests to policy-generated failures. The initial hardened cohort contains **21 admitted problems**; two further candidates fail the all-positive gate. The final larger cohort contains **29 train, 2 validation and 13 held-out test problems**. Its independent closure audit checks 528 positive and 528 negative reference replays across the 44 admitted tasks. Of 50 candidates reaching the hardened audit, 27 receive fresh panels and 23 reuse exact sealed initial panels, including the two initial quarantines. Reuse requires equal executable contracts, source bytes, original input records and raw replay ledgers; an explicit post-reuse audit additionally verifies adapter, problem identity and canonical task hashes. Six further prospective candidates had already failed earlier admission, leaving 12 exclusions overall. The original 32/2/16 quota was not reached and is not claimed.

All primary coding endpoints—base, MaxRL and Re:Max—were freshly sampled with the same hardened verifier and prompt definitions. CPU regrading of historical samples serves only development diagnosis and training-task selection. Replay banks begin empty, with no historical samples or reference programs seeded into them. The primary result must not mix historical and hardened rewards, mode keys, or training trajectories.

[Large Language Monkeys](https://arxiv.org/abs/2407.21787), already cited as `brown2024monkeys`, motivates repeated program sampling and execution-based evaluation. Its Section 4.2.2 also discusses CodeContests false negatives when several outputs are correct. It is neither our dataset source nor an experimental arm. Our temperature 1.0, top-p 1.0 protocol is not an exact reproduction of its Appendix A.2 temperature 0.6, top-p 0.95 protocol; task selection, prompting, model and verification also differ.

## Real news articles with several accepted topic answers

The noncoding task uses the NEWS component of [SATA-Bench](https://arxiv.org/abs/2506.00643), derived from [Reuters-21578](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection). Reuters supplies real 1987 newswire documents and human-assigned topic annotations. SATA supplies the processed multiple-choice menus and sampled taxonomy distractors, followed by correction and consensus filtering by three human annotators. Our source is the [canonical Hugging Face release at a fixed revision](https://huggingface.co/datasets/sata-bench/sata-bench/tree/ba43a7ab537adfa3498e3a160a6d1eafbefc95c1), not the inconsistent helper JSON in its code repository. The SATA dataset card specifies CC-BY-NC-4.0; the UCI source reports CC-BY-4.0. A software repository's MIT license does not replace the dataset's license.

Our instruction explicitly asks the model to **choose ONE correct listed topic**, noting that other options may also be correct. This is a task adaptation: SATA's official task asks for all correct options, which defines one correct set. Our reported acceptance is consequently not official SATA select-all accuracy. We retain the released passage, question, positive topics and distractors without changing labels or adding generated content. The six-option menus have two to five accepted native topics.

A fixed structural filter removes five of the 248 NEWS rows because the question field contains extra nonwhitespace material. The remaining **243 documents are frozen as 128 train, 32 dev and 83 test** before model experiments. Normalized-content and character-shingle checks precede splitting; an independent comparison against the original Reuters archive matches every source NEWS passage and finds no original Reuters article identifier crossing retained splits. Multiple original matches for some passages are recorded, rather than silently collapsed into a claim of exact unique provenance. Source inspections are assistant-assisted audits; they are not new human annotations.

Each retained question has a persisted option permutation, with a deterministic rotation balancing gold-label positions across its split. Training gold-position counts for A–F are 49, 49, 48, 48, 49 and 49. Option order is identical across methods and repeated samples. The verifier strips surrounding whitespace and otherwise accepts only one uppercase menu letter whose underlying topic is source-positive. It assigns the canonical key `reuters_topic:<native topic>`; incorrect topics and malformed responses receive reward zero. There is no LLM judge.

The modes are native taxonomy tags, not arbitrary answer letters. Some tags overlap or are hierarchical, such as grain and corn; they are meaningful distinct labels but not independent lines of reasoning. Known support means all released positive topics in the given menu. It does not mean that the source annotators exhaustively identified every conceivable relevant topic. This application tests discovery and retention of accepted document tags under repeated sampling.

## Matched training and completed pilots

Within each domain, MaxRL and Re:Max use the same base model, initial LoRA parameters, prompt schedule, fresh-sample budget, generation settings and verifier. The pilot uses rank-16 LoRA, AdamW learning rate 1e-5, and 16 fresh rollouts per update. MaxRL uses the production fresh-rollout objective. Re:Max additionally applies the production verified likelihood-replay term, with uniform weight over a prompt's discovered modes and one first-verified policy exemplar per mode. Each bank holds at most 16 modes per prompt. One global scheduler selects a nonempty prompt bank per update by round-robin traversal of sorted task IDs. The raw replay coefficient is 0.1, corresponding to the production effective coefficient 0.005859375. Both arms begin with empty banks. The matched control applies zero replay gradient. Original prompt/response token IDs, raw text, verdicts, bank updates and final checkpoints are retained for audit.

The completed QA comparison uses Qwen2.5-7B-Instruct, one paired training seed and 32 updates per arm on 16 training documents: 512 fresh training rollouts per arm. External endpoints then draw 128 responses for every one of the 16 trained documents and all 32 dev documents. The 83 test documents remain model-untouched. The independent endpoint audit checks the same frozen runner, prompts, sampling settings and base/checkpoint identities across all three endpoints.

Every training group was saturated in both arms: 30 groups were entirely correct and two entirely incorrect, with no mixed-reward group. The resulting fresh MaxRL advantage was zero; its trainable weights stayed unchanged. Re:Max had a nonzero replay gradient and changed its weights. Base and MaxRL endpoint aggregates agree. This is a useful mechanism and ceiling smoke test, not evidence that the method improves a nontrivial learning trajectory.

| Fixed evaluation stratum | Method | Acceptance | Expected distinct @8 | Expected distinct @32 | PCMD | Annotated-topic coverage | PCMD-eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All 32 dev documents | MaxRL / base | 99.8291% | 1.19562 | 1.31250 | 0.07405 | 57.1354% | 32/32 |
| All 32 dev documents | Re:Max | 99.8535% | 1.20575 | 1.33805 | 0.07339 | 57.1354% | 32/32 |
| All 16 trained documents | MaxRL / base | 93.8477% | 1.24611 | 1.27743 | 0.15231 | 63.5417% | 15/16 |
| All 16 trained documents | Re:Max | 93.8477% | 1.25324 | 1.27744 | 0.16272 | 63.5417% | 15/16 |

Changes are small and mixed: dev expected distinct @32 increases while dev PCMD decreases slightly; trained PCMD increases while trained expected distinct @32 is effectively unchanged. Neither stratum gains observed annotated-topic coverage. The final replay bank contains 20 modes across 15 training documents; the five documents with multiple bank modes form a post-treatment exploratory subset, not the primary comparison denominator. Replay cannot be credited with retaining modes that were never discovered. A single paired seed cannot establish an efficacy advantage, extinction recovery, or reproducibility across training seeds.

## Completed hardened coding pilot

The coding comparison uses Qwen2.5-Coder-7B-Instruct, revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`, with one paired seed and 32 updates per arm. Eight training problems were chosen using development capability evidence; this is disclosed development selection. Both arms receive 512 fresh training rollouts. Independent base, MaxRL and Re:Max endpoints then sample **all 21 admitted initial-development problems**, 128 responses per problem, with the same hardened verifier, rendered prompts, sampling settings and evaluation seed. All 21 remain the complete pilot endpoint cohort. The 13 untrained development problems are distinct from the separate 13 reserved larger-study test problems; no held-out generalization result is claimed here.

Both coding arms have 27 mixed-reward groups, five all-wrong groups and no all-correct group. Recomputing binary MaxRL advantages and checking positive actual gradients confirms active fresh updates, including in the MaxRL control whose replay gradient is zero. Re:Max applies replay gradients on all 32 updates. Initial LoRA hashes match; raw-token bindings, empty-start bank reconstruction and both final checkpoint seals pass. The final banks contain 21 discovered modes over seven prompts for MaxRL and 19 over seven for Re:Max. Those discovery counts are descriptive and do not favor Re:Max. The two-problem, eight-samples-per-problem built-in evaluation is only an execution diagnostic; the following results use the full external endpoints.

| Fixed evaluation stratum | Method | Acceptance | Expected distinct @8 | Expected distinct @32 | PCMD | PCMD-eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| All 21 development problems | Base | 8.7054% | 0.47640 | 1.00916 | 0.54773 | 3/21 |
| All 21 development problems | MaxRL | 9.4122% | 0.50567 | 1.04595 | 0.53601 | 3/21 |
| All 21 development problems | Re:Max | 9.4494% | 0.49826 | 1.00233 | 0.54129 | 3/21 |
| All 8 trained development problems | Base | 20.1172% | 1.09597 | 2.31059 | 0.54773 | 3/8 |
| All 8 trained development problems | MaxRL | 21.3867% | 1.16666 | 2.42477 | 0.53601 | 3/8 |
| All 8 trained development problems | Re:Max | 21.6797% | 1.15609 | 2.33391 | 0.54129 | 3/8 |
| All 13 untrained development problems | Base | 1.6827% | 0.09513 | 0.20828 | — | 0/13 |
| All 13 untrained development problems | MaxRL | 2.0433% | 0.09891 | 0.19745 | — | 0/13 |
| All 13 untrained development problems | Re:Max | 1.9231% | 0.09344 | 0.18290 | — | 0/13 |

Across all 21 problems, Re:Max accepts 254 of 2,688 samples versus MaxRL's 253, while expected distinct @32 decreases from 1.04595 to 1.00233. On the trained eight problems, acceptance is 222/1,024 versus 219/1,024, while expected distinct @32 decreases from 2.42477 to 2.33391. On the untrained development problems, acceptance is 32/1,664 versus 34/1,664 and expected distinct @32 also decreases. Conditional PCMD increases slightly on the same three eligible trained problems; zero untrained-development problems qualify. This small eligible denominator must not stand in for the complete cohort. These are mixed single-seed pilot outcomes, **not evidence of a Re:Max efficacy advantage**. The pilot establishes executable comparisons and multiple verified witness modes, while leaving seed-level effects and reserved-test performance to the larger study.

Mean coding update time is about 63.45 seconds per arm, with peak allocated GPU memory about 23.05 GiB. The complete two-domain pilot effort, including failed/canceled allocations and the separate resume check, consumed 3.78944 allocated GPU-hours according to terminal scheduler receipts. This is within the 200 GPU-hour overall cap; larger studies remain prospective.

## Prospective larger-study reporting

The main study should fix all protocols before evaluating the reserved test sets, use at least three paired training seeds, and treat held-out performance as primary. Trained-prompt support and bank retention are separate mechanistic diagnostics. Any development capability screen and any selection based on it must be disclosed; held-out admission must depend only on source and verifier quality, not on model success. The combined pilots and follow-up remain subject to the original 200 allocated-GPU-hour ceiling, including failed and canceled allocations.

Always report acceptance and expected distinct valid modes at matched budgets, including @8 and @32, over every task in the fixed evaluation cohort. For N responses and accepted-mode counts n_m, expected distinct @k is the sum over modes of `1 - choose(N - n_m, k) / choose(N, k)`; unsuccessful samples remain in N. PCMD is the probability that two accepted samples drawn without replacement have different modes: `1 - sum_m n_m(n_m - 1) / [C(C - 1)]`, where C is the accepted count. Require at least **30 accepted samples per task**, compare methods on the **common eligible intersection**, and display eligible and total task counts. This threshold applies to accepted samples, not to known support or to a requirement of 30 eligible tasks.

For QA, also report observed native-topic coverage divided by known annotated support over all tasks. For code, do not invent a finite total support from observed witness keys. Report paired differences and variation across training seeds; prompt-level bootstrap uncertainty may supplement, but not substitute for, seed variation. Failure-heavy tasks remain visible through acceptance and expected distinct counts even when excluded from conditional PCMD. Predefine a final result that is informative if effects are null or negative.

## Evidence and bibliography for integration review

Both result tables are derived from `var/artifacts/real_domains_pilot_20260921/independent_hardened_final_v2/summary.json`, which independently audits the completed QA `qa_endpoint_{base,maxrl,remax}_v2` endpoints, hardened coding `code_hardened_base` and `code_hardened_endpoint_{maxrl,remax}` endpoints, and both paired training histories. Its readiness gate passes with terminal accounting and QA source provenance. Earlier reports remain preserved; historical coding evidence and the first final audit's analysis-schema failures are superseded, not silently overwritten. The independent hardened initial-data audit is `code_hardening_independent_source_review_v2.json`; the final larger-cohort closure audit is `code_hardening_larger_independent_closure_v1.json`, in the same artifact directory. The original-news comparison is `var/artifacts/noncoding_multi_answer_sata_20260921/provenance_audit/news_source_comparison.json`, schema v3. Data and audit contracts are documented in `docs/noncoding_multi_answer_sata_protocol_20260921.md`, `docs/real_domains_pilot_20260921.md`, and `docs/real_domains_final_audit_runbook_20260921.md`.

Keep the existing `brown2024monkeys` entry. The following are proposed bibliography additions only; this draft does not modify the paper bibliography. Reuters uses the UCI repository's recommended citation year, 1987; its later repository donation date is not substituted for that year.

```bibtex
@misc{wang2025codecontestsplus,
  title = {{CodeContests+}: High-Quality Test Case Generation for Competitive Programming},
  author = {Wang, Zihan and Liu, Siyao and Sun, Yang and Li, Hongyan and Shen, Kai},
  year = {2025},
  eprint = {2506.05817},
  archivePrefix = {arXiv},
  primaryClass = {cs.SE},
  url = {https://arxiv.org/abs/2506.05817}
}

@misc{xu2025satabench,
  title = {{SATA-BENCH}: Select All That Apply Benchmark for Multiple Choice Questions},
  author = {Xu, Weijie and Cui, Shixian and Fang, Xi and Xue, Chi and Eckman, Stephanie and Reddy, Chandan K.},
  year = {2025},
  eprint = {2506.00643},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2506.00643}
}

@misc{lewis1987reuters21578,
  author = {Lewis, David},
  title = {{Reuters-21578 Text Categorization Collection}},
  year = {1987},
  publisher = {UCI Machine Learning Repository},
  doi = {10.24432/C52G6M},
  url = {https://doi.org/10.24432/C52G6M},
  note = {Dataset}
}
```
