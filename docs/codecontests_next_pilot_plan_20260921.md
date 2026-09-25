# Next pilot: broaden constructive-code coverage, then test learning

Status: prospective recommendation after the completed capability pilot. No new GPU jobs or paper edits are part of this note.

## Decision

Keep CodeContests+ for one broader, bounded feasibility screen. The previous three development problems are too narrow to reject the application. They establish that the execution pipeline can recover stable alternatives on one real contest problem; they do not test Re:Max versus MaxRL.

The next question is whether there are enough well-specified, reliably checked, learnable human-authored problems with meaningful alternative outputs. Only after answering that question should we spend on a paired learning pilot.

Evidence: [completed pilot report](../var/artifacts/codecontests_pilot_20260921/report.md). Qwen2.5-Coder-7B-Instruct produced 4/64 accepted programs and one mode on 359_B, 26/64 and five modes on 988_A, and 0/64 on 1399_D. All generations stopped normally; the longest was 323 tokens under a 1,024-token limit. Raising that limit is not the first intervention supported by these results. The scheduler charged 377 seconds, or 0.1047222 GPU-hours.

## 1. Admit a wider development slate on CPU

Start from the [24-candidate shortlist](codecontests_wider_screen_shortlist_20260921.md): 22 new problems plus two existing development anchors. The list spans ordered sequences, unordered subsets, matrices, and partitions/allocations. Particularly useful candidates include 1454_A and 1513_A (permutations), 1016_D and 1360_G (matrices), 244_A (anchored allocations), and 1102_B (partitions).

The local index has 395 rows but only 375 distinct normalized statements. It is metadata, not a ready 395-problem benchmark. Some rows have unique substantive answers; 83 contain image tags. Even image-free statements can have flattened equations or exponents. Admit tasks using source and checker evidence, not an assumption that every reference-independent checker implies multiple answers.

Before sampling a task:

1. Materialize only its pinned source records and tests. Check the rendered statement against the source, document any faithful formatting repairs, and validate that inputs satisfy its constraints.
2. Compile and test the released checker using known accepted and rejected source programs. Check execution time, memory and stdout limits. Large program output is different from long generated program text.
3. Establish that distinct substantive answers exist and that the fixed suite can expose them. Audit canonicalization against examples of both equivalent and different answers; use distinct accepted source/reference constructions when available. References are verifier evidence only, never initial replay entries.
4. Remove formatting, set order and arbitrary partition-label variation. Preserve genuinely observable choices such as selected student indices or assignments to named children. Do not call every alternative witness a different algorithm.
5. Freeze task manifests, input suites, prompt text, canonicalizers and failure rules before model sampling. If a candidate fails admission, record the reason and select a reserve before freezing the slate.

Target 24 admitted development tasks, including the two continuity anchors. If the shortlist shrinks, extend source review before running the screen; do not silently replace disappointing model results afterward. Report new-task results separately from the already observed anchors.

Preserve the original held-out problems (361_B, 1294_C and 149_C). Before any wider model screen, reserve additional final-test candidates using statement/checker/reference audits alone. Deduplicate and split by normalized statement, not just contest ID. Every model-screened problem stays training/development material. Splitting inputs of one statement does not create independent held-out problems.

## 2. Fixed broad sampling, then independent deeper sampling

Use the same pinned Qwen2.5-Coder-7B-Instruct, prompt style, T=1, top-p=1 and 1,024-token cap as the completed pilot. Hold those choices fixed so this stage measures the effect of a broader task slate. Recheck every initially accepted program for identical acceptance and canonical behavior.

Broad screen: 24 tasks x 32 samples = 768 programs. Thirty-two draws are still noisy, but at a true 10% success rate the probability of observing no success is about 3.4%, compared with 18.5% for sixteen draws.

Deep screen: 12 tasks x 128 fresh samples = 1,536 programs. Predeclare selection before the broad screen: eight promising tasks balanced across output families, two borderline tasks, and two randomly chosen zero-success tasks if that stratum exists. Define deterministic substitutions if a stratum is too small. A broad result of at least four accepted samples and two stable modes is a useful priority signal, not a universal exclusion threshold. Keep some high-accuracy tasks to measure retention as well as improvement.

Do not pool adaptively selected deep-screen results into an unbiased estimate for all CodeContests problems. Publish the entire admission/screening funnel and per-task outcomes, including failures. The target population is explicitly constructive contest problems with the stated structural eligibility rules.

Report correctness, stable distinct witnesses at a common sampling budget, checker/runtime failures, generation truncation, and representative outputs. Report PCMD only when its accepted-sample threshold is met, with eligible/total denominators. Low PCMD eligibility is not zero diversity: 988_A had 26 successes, below the current threshold of 30. Fixed 128-draw evaluation improves eligibility on moderate-success tasks without selecting a variable number of draws until success.

The broad-plus-deep screen has a 3.5 allocated-GPU-hour ceiling. This is an allowance, not a measured duration or a promise of completion. The earlier observed 380 generated tokens/second is encouraging, but prompt length, program length and checker cost may change. Charge all allocated GPUs, including time they wait for verification. Stop at the ceiling and report completed work.

## 3. Conditional production timing and paired learning pilot

Proceed only if the deeper screen provides at least eight reliable development problems across at least three output families, with multiple stable canonical witnesses and enough accepted samples to populate replay. This is an engineering gate for a tiny pilot, not sufficient breadth for the final application study. Retain sparse-success tasks in the report rather than claiming they are unsolvable.

First allow at most one GPU-hour to measure the actual 7B training path: gradients, optimizer memory, actor synchronization, on-policy rollout, verification and replay loss with a populated policy-generated bank. Do not estimate training cost from inference throughput. Do not assume adapter-based training or CPU offload works before testing the selected implementation. If the selected configuration cannot run within this timing allowance, stop and revise the training plan.

Then run MaxRL and current Re:Max with one matched seed and 32 real online updates per arm, on a fixed small development slate. Re:Max must be the current uniform verified-mode replay method: one policy-generated exemplar per prompt-local mode. Start both arms from the same checkpoint and empty banks. Match prompts, fresh rollout count, decoding and update schedule; account for replay's extra computation separately. Do not reuse an older multi-term experimental objective under the Re:Max name.

Allow at most six GPU-hours across both training arms. Measure accepted samples, useful reward variation within groups, replay support and mode retention at baseline and fixed checkpoints. Use one additional GPU-hour for fixed development evaluation, with 128 draws per selected task where feasible under the frozen manifest. Set any sample-count fallback before launching, not after seeing method differences.

The paired pilot tests that learning and replay execute correctly, retain a useful signal and fit the budget. A null result after 32 updates is not evidence that Re:Max fails. Broken gradients, unstable mode keys, empty verified banks, an unusable reward signal or excessive runtime are reasons to stop or revise.

## Budget and transition to the non-synthetic application

| Stage | Allocated GPU-hour ceiling |
| --- | ---: |
| Completed capability pilot (measured) | 0.1047222 |
| Wider and deeper capability screen | 3.5 |
| Actual learner timing | 1.0 |
| Paired training micro-pilot | 6.0 |
| Fixed development evaluation | 1.0 |
| Total through this plan | 11.6047222 |

These are total GPU-hours, not wall-clock hours on a multi-GPU job. No unmeasured training budget is guaranteed. Before a full study, use measured throughput and all-in costs to fit training, evaluation, seeds and contingency within the user's 200-GPU-hour total. About 188 GPU-hours would remain if every proposed stage reached its ceiling.

A final application should expand beyond the eight-task engineering slate: aim for 32–64 training problems and 16–32 structurally admitted, previously unscreened held-out problems across several families, with multiple matched seeds if timing supports them. These counts are targets requiring further curation, not currently validated assets. If that breadth cannot be assembled, describe a small case study honestly or change application; do not manufacture additional questions by splitting test inputs.

Freeze the held-out list before model screening and never replace a held-out problem for poor baseline or treatment performance. Use all held-out tasks for correctness and unconditional expected distinct valid witnesses at a fixed sampling budget. Give success-conditioned diversity separately, including eligibility. Corroborate whole-suite mode hashes with per-input witness differences on fresh, valid probe inputs: a hash can change because of only one edge case. Inspect whether gains survive removal of purely representational variation.

Pass@k on the same binary verifier is a correctness/coverage measure. Under independent sampling its population value is 1-(1-p)^k, and the finite-sample estimator depends only on the number of accepted programs. It cannot by itself show that mode diversity caused a practical benefit. A stronger claim about useful solution portfolios requires a separately frozen downstream criterion; this pilot should claim verified output diversity and correctness only.

## When to change course

Change course if a functioning, audited wider screen yields too few reliably solvable multi-answer tasks, if apparent diversity is mostly serialization or anonymous-label variation, or if measured training cost makes a credible comparison exceed 200 GPU-hours. The three-task pilot alone does not justify that decision. Do not keep escalating model size without a bounded test and a cost projection.

The intended result is an application to human-authored constructive programming problems, with CodeContests+'s generated tests and checkers explicitly disclosed. This is non-synthetic task provenance, not production software evidence. Cite [Large Language Monkeys](https://arxiv.org/abs/2407.21787) for repeated-sampling evaluation and the multiple-valid-output issue, and [CodeContests+](https://arxiv.org/abs/2506.05817) for the verification infrastructure. See the [citation note](codecontests_pilot_citations_20260921.md). Keep the appendix unchanged until the paired study has interpretable, reproducible evidence.
