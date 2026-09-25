# Real-domain Re:Max versus MaxRL pilots

This is a development protocol, not a claim of an established method advantage.
The two applications are constructive programming and multi-topic classification
of real Reuters news. The combined pilot and proposed follow-up must fit within
200 allocated GPU-hours. Scheduler allocation time, including failed jobs and
model loading, is the accounting unit.

## What is real, and what a mode means

**Coding.** Problems are human-authored Codeforces competition questions. We use
released CodeContests+ output checkers with CodeContests-O finite input suites.
Those verification assets include generated material; this is a real-task
application, not an entirely human-produced benchmark. Acceptance means passing
the fixed, audited suite, not a proof of correctness on every valid input.
A mode is a canonical valid output-witness signature across that suite. It is
not an algorithm, source-code style, or proof of semantic program equivalence.
Canonicalization removes serialization, order where irrelevant, and anonymous
label permutations while preserving choices that affect the requested witness.
Accepted samples must pass a second execution with the same canonical signature.

**QA.** SATA-Bench NEWS supplies real Reuters articles and their annotated topic
labels. We explicitly adapt its select-all format to: choose ONE correct listed
topic. Original positive and negative choice texts are retained and shuffled by
a fixed rule. A mode is the native Reuters topic identifier, not its menu letter.
Some topics are hierarchical or overlapping, so the endpoint is annotated-topic
coverage, not diversity of independent reasoning. Correctness follows the
released labels. The fixed set contains 128 train, 32 development, and 83 test
articles; test responses are not generated during pilot selection. See the
separate QA protocol for provenance, exclusions, licenses, and duplicate checks.

## Development sequence

1. Audit sources, constraints, checker behavior, canonicalization, split groups,
   and the exact rendered prompts before model sampling. Keep rejected tasks and
   reasons in the source admission ledger.
2. Measure base-model capability on a fixed development slate. The wider initial
   coding screen contains every one of the 23 admitted questions, with 32 samples
   per question. Do not select a favorable replacement after seeing an outcome.
   Follow with deeper fresh sampling to distinguish absent support from rare
   modes; every outcome-dependent training selection must be recorded explicitly.
3. Verify the actual training path with a short GPU run, then run matched MaxRL
   and Re:Max pilots. Freeze the model, data, source, prompt order, seeds, optimizer,
   rollout budget, and output cap. Start both replay banks empty. Bank only first
   verified policy exemplars; do not seed banks with reference answers or programs.
4. Evaluate sealed checkpoints using fresh samples and the same evaluation setup
   for base, MaxRL, and Re:Max. Separate trained prompts from unseen development
   prompts. Keep the final test split untouched until the larger protocol is fixed.
5. Use measured speed, memory, accuracy, mode support, and verifier failures to
   decide whether the larger experiment is ready. A working pipeline is not
   evidence that Re:Max wins. A single seed is a feasibility result.

## Matched optimization and measurements

The shared pilot trainer calls the production MaxRL advantage and Re:Max replay
implementation. Both arms generate 16 fresh responses per update and compute the
same replay path; its applied gradient is zero for the MaxRL control. Re:Max uses
one exemplar per retained prompt-mode with uniform mode weights. LoRA rank is 16,
AdamW learning rate is 1e-5, and sampling is temperature 1 with full support over
tokenizer-renderable tokens. Behavior log probabilities and optimization use the
same policy and token masks. Original sampled tokens, prompt hashes, decisions,
mode keys, bank transitions, and checkpoint hashes are retained.

Report accuracy and expected distinct valid modes at fixed sample budgets
together. For QA also report coverage of the known annotated support. Report
conditional pairwise mode diversity only for prompts with at least 30 accepted
samples and always give the eligible denominator; compare methods on a common
eligible set as well. Pass@k alone does not demonstrate a diversity gain.
For coding, also report truncations, execution failures, suite coverage limits,
and accepted-sample stability. Audit failures invalidate a run rather than
becoming ordinary zero rewards.

The QA 32-update feasibility pair is a ceiling/zero-gradient setting: all fresh
groups were wholly correct or wholly wrong. This is a useful replay mechanism
check but cannot establish a broad advantage in learning reward. The fresh
checkpoint comparison must determine whether topic coverage improves without
losing accuracy.

## Reproduction and evidence

- `ops/prepare_real_domains_run_20260921.py`: freeze and submit bounded jobs.
- `ops/train_real_domains_pilot_20260921.py`: shared production-backed trainer.
- `ops/evaluate_real_domains_20260921.py`: fixed-budget raw-sample evaluation.
- `ops/summarize_real_domains_pilot_20260921.py`: independent receipt audit.
- `ops/account_real_domains_pilot_20260921.py`: scheduler GPU-hour accounting.
- `var/artifacts/real_domains_pilot_20260921/`: requests, immutable job bundles,
  outputs, checkpoints, and accounting receipts.

Large Language Monkeys motivates repeated sampling on real verifiable tasks; it
is not the dataset name and is not a separate experimental arm. Cite it alongside
the actual dataset and verifier sources when describing this application:
[Large Language Monkeys](https://arxiv.org/abs/2407.21787),
[CodeContests+](https://arxiv.org/abs/2506.05817),
[SATA-Bench](https://arxiv.org/abs/2506.00643), and the
[Reuters collection](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection).

## Coding verifier correction discovered during the pilot

The first 23-task screen is historical diagnostic evidence only. Independent
counterexamples fixed from source-program audits, before model sampling, reject
six of its 74 reward-accepted generated programs. Among the five covered tasks,
14 generated programs were accepted by the original suites and six failed the
additional probes. This targeted sample does not estimate an overall false
positive rate. The two newly submitted coding training jobs were cancelled;
their partial checkpoints must not be used as a corrected comparison.

A separate hardened data version appends those exact five source-audit probes
while preserving original inputs and output checker bytes. Admission now requires
12/12 source-positive and 12/12 source-negative decisions. Incompatible tasks are
quarantined with their receipts. This is a regression check on known reference
programs, not proof of complete verification. Original data, generated samples,
failed allocations and source ledgers remain intact.

Old raw generations may be CPU-regraded for diagnosis and training-subset
selection, with explicit old/new verifier identities and denominators. The
primary corrected comparison will use a fresh base evaluation and two fresh
training runs under the same hardened verifier. Banks start empty because adding
probes changes the witness signatures and hence the mode space. Do not compare
old and new mode counts as if their definitions were identical.
