# ConstructiveCode: a realistic fifth ModeBench domain

Status: v1 executable admission failed; ineligible before model sampling; not a paper result  
Date: 2026-07-29

## Admission outcome

The source audit, four-family adapters, pinned official `testlib`, networkless
Landlock/seccomp boundary, and compute-node throughput audit are complete. The
full v1 overlay replay executed 794 labelled programs over 13,898 candidate-test
pairs; throughput passed at 0.068 s median and 0.203 s p95.

The v1 row is nevertheless ineligible. The frozen materializer treated both
`py2` and `py3` as Python while the admission worker was explicitly Python
3.10; 242 of 794 selected replay programs were not Python-3-labelled. Released-
checker equivalence also failed the frozen gate: only 988A passed every
criterion, while 482A, 1153B, and 149C had registered rate or identity failures.
No model sample was taken. A Python-3-only slate is allowed only as a separately
preregistered v2 and cannot repair or replace v1 in the stopped cohort.

## Decision

Pursue a small, audited `ConstructiveCode` pilot from the hash-aligned
intersection of
[CodeContests+ Verified](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus)
multiple-answer rows and
[CodeContests-O](https://huggingface.co/datasets/caijanfeng/CodeContests-O).
CodeContests+ supplies the per-problem TPR/TNR eligibility record, labelled
solution pool, and semantic output checker. CodeContests-O supplies a newer
feedback-hardened test suite and explicitly reuses the CodeContests+ checkers.
Treat the latter as an optional test-suite overlay until problem identity,
checker behavior, and local submission replay agree exactly; otherwise fall
back to the CodeContests+ 5x tests for that row.

Do not add it to the benchmark table or claim a fifth ModeBench domain until
the executable-identity and base-model viability gates below pass. The current
route-successor campaign remains the immediate compute priority.

The key reason for this choice is scientific, not cosmetic. Ordinary code
benchmarks verify that a program implements one specified input/output
function. Different correct source programs are then surface or implementation
variants, not different semantic outcomes. Multiple-answer constructive
programming tasks instead require a generated program to emit one of many valid
witnesses. The official checker validates the executed witness, and the same
parsed witness can provide the mode key.

## Why this is larger than the current synthetic domains

CodeContests+ contains 11,690 competitive-programming problems, test-case
generators, input validators, output checkers, and more than 13 million correct
and incorrect submissions. Its Verified subset retains problems whose
generated tests achieve both true-positive and true-negative rates above 0.9
against held submissions. The released project is CC-BY-4.0.

CodeContests-O contains 11,682 problems, averages 40.19 tests per problem, and
reports aggregate TPR 89.37% and TNR 90.89% after feedback-driven refinement
against roughly 11 million solutions. It publishes generators, tests,
checkers, and iteration history under Apache-2.0. Those aggregate rates do not
replace the per-problem CodeContests+ Verified threshold or our own replay
audit.

The proposed domain would contain real programming statements and require the
policy to generate a complete Python program. The program would run on frozen
hidden inputs in an isolated container. Tasks would include constructive
families such as:

- topological orderings and constrained permutations;
- matchings, assignments, and decompositions;
- schedules and packings;
- paths, orientations, and graph labelings.

These are not all interchangeable. Each admitted family needs an explicit
semantic schema and canonicalization audit.

## Count-only source audit

The pinned source audit is complete at
`var/artifacts/constructive_code_source_overlap_audit.json`. It uses
CodeContests+ revision
`96c850540fade31d384a25766461e0da6b08f5fc` and CodeContests-O revision
`1a765191567b429f633bbd1c6e67b5890dfaf267`.

The audit found:

- 11,690 CodeContests+ 5x rows, with unique source IDs and unique
  `{id}. {title}` overlay names;
- 11,683 physical CodeContests-O rows: 11,682 unique nonempty problem names
  plus one empty-name record, reconciling the physical count with the dataset
  card's 11,682-problem claim;
- 7,558 unique exact name joins, of which 7,556 have both normalized statement
  and checker hash equality;
- 6,036 rows at the inclusive CodeContests+ Verified threshold;
- 469 Verified rows under the conservative multiple-answer lower bound
  (checker reads `inf` and `ouf` but not reference stream `ans`);
- 424 of those with at least two labelled correct Python submissions; and
- 395 with a unique cross-source name plus exact normalized statement and
  checker alignment.

The 395-row set is a feasibility pool, not an admitted benchmark. Its key list
is sealed by SHA-256 in the aggregate artifact. The audit projected only the
declared data columns plus Parquet metadata and did not materialize submission
code, generated tests, corner cases, generators, commands, or iteration
histories. The source file sizes in the artifact are logical repository sizes,
not measured HTTP transfer volume.

First-failure counts are also frozen: 5,654 rows miss the Verified threshold,
5,567 Verified rows use a reference-dependent or statically ambiguous checker,
45 conservative-checker rows lack two correct Python submissions, and 29 lack
an exact overlay name. Checker execution, witness schema review, local replay,
and isolation remain fail-closed prerequisites.

## Why this domain comes before other skill domains

`ConstructiveCode` is the strongest answer to the paper's external-validity
gap because both parts of the claim are executable: the model must synthesize a
program, and the program must emit one of multiple checker-accepted witnesses.
It is therefore more realistic than adding another small hand-built constraint
family while preserving ModeBench's correctness-and-identity contract.

The best lower-cost fallback is an `IPCPlan` domain drawn from International
Planning Competition PDDL tasks and validated with VAL. A mode could be the
canonical executed state/action trajectory of a valid plan. This is a sound
benchmark object and spans logistics, rovers, scheduling, and resource
management, but it overlaps MathIR's action-trajectory abstraction and is less
persuasive as evidence beyond synthetic symbolic tasks. Pursue it if the
constructive-code capacity or execution-cost gate fails, not as a post-outcome
replacement in the same experiment.

Free-form recipe generation remains deferred. Real cooking success depends on ingredient
chemistry, equipment, timing, sensory quality, and human preference that we
cannot verify by executing text. A separate `PantryPlan` track now tests exact
nutrition, inventory, allergen, and quantity constraints with the
ingredient-support set as its mode key; see
`pantry_plan_modebench_extension_plan.md`. It establishes a verified
ingredient-formulation domain, not that different recipes make the same dish.
Ingredient prose, recipe embeddings, and LLM judgments remain ineligible mode keys.

Repository repair, browser use, and API workflows are useful correctness
transfer tracks, but patch text and action traces are not automatically
different semantic outcomes. They need a separate validator-bound identity
study before being called ModeBench domains.

## Benchmark contract

For problem statement \(x\), generated program \(p\), frozen input suite
\(I_x=(i_1,\ldots,i_m)\), and trusted checker \(C_x\):

1. execute \(p(i_j)\) under the frozen runtime and resource limits;
2. parse the actual emitted witness \(w_j\);
3. require \(C_x(i_j,w_j)=1\) for every primary test;
4. canonicalize the accepted witness with a versioned, task-family-specific
   map \(K_x(w_j)\); and
5. return the tuple
   \[
   V(p,x)=\bigl(K_x(w_1),\ldots,K_x(w_m)\bigr).
   \]

Compilation failure, exception, timeout, malformed output, checker failure, or
canonicalizer disagreement returns no key and reward zero.

The mode is therefore the behavior of the generated program over a fixed input
suite, not its source text, AST, variable names, comments, or formatting.
Whitespace aliases collapse. Ordering is normalized only when the task itself
declares that ordering irrelevant. Two accepted programs that emit different
valid witness tuples occupy different modes.

## Source curation gate

Start from CodeContests+ Verified, join CodeContests-O by stable problem
identity, and retain only rows that satisfy all of:

- a custom multiple-answer output checker is present;
- normalized problem ID and statement hashes agree across sources before any
  CodeContests-O test-suite overlay is admitted;
- the problem is non-interactive, self-contained, and requires no image or
  network access;
- the statement, checker, generator, source-specific tests, and join record
  have frozen hashes;
- at least two held accepted Python submissions execute successfully under the
  local runtime;
- those submissions produce at least two different canonical witness keys on
  the development probes;
- median and tail execution costs fit the online-RL reward budget;
- the task maps to a reviewed witness schema rather than normalized raw text;
- train, development, and evaluation problem IDs are disjoint; and
- CodeContests+, CodeContests-O, original-problem, and testlib attribution and
  redistribution metadata are retained.

The full datasets are much too large to mirror casually: the hosted
CodeContests+ tree is approximately 951 GB and CodeContests-O approximately
325 GB. Stream metadata first, materialize only the selected rows, and record
every rejected or unmatched row with a machine-readable reason.

## Executable-identity gate

Before sampling a policy:

- run the released checker against at least 100 known-correct and 100
  known-incorrect submissions per retained task when available, under both
  the CodeContests+ 5x tests and any proposed CodeContests-O overlay;
- require zero false accepts by the wrapper relative to the released checker;
- require the wrapper and chosen test suite to preserve the per-problem
  CodeContests+ Verified true-positive/true-negative threshold;
- test whitespace, line-ending, and harmless source-refactor invariance;
- test that semantically unordered witness components collapse exactly when
  specified;
- certify at least two distinct accepted behavior keys per prompt; and
- independently replay every published key from raw program, input, and
  emitted-output artifacts.

Any task with ambiguous identity, checker nondeterminism, or unstable runtime is
excluded before the experimental split is frozen.

The initial implementation lives in
`src/oat_drgrpo/constructive_code.py`. It freezes four semantic families:
ordered integer sequences, unordered integer sets, position-to-value
assignments, and unordered partitions. Released-checker decisions bind the
exact input and output bytes by SHA-256 before canonicalization. Formatting
aliases collapse; sequence order remains semantic; set order, assignment-pair
order, group order, and member order collapse only under the corresponding
schema. Duplicate elements, duplicate keys, cross-group duplicates, rejected
checker decisions, and evidence/output hash mismatches fail closed.

`ops/audit_constructive_code_checker_equivalence.py` defines the replay-ledger
gate. For every frozen task and suite it requires the registered correct and
incorrect replay counts, wrapper/released-checker decision equality, the
per-task TPR/TNR floor, behavior keys only for accepted executions, and at
least two distinct accepted behavior keys. The gate is implemented but cannot
pass until selected source payloads are replayed.

The bounded 16-task schema-review slate is materialized by
`ops/materialize_constructive_code_review_slate.py`. It contains four proposed
tasks per witness family, at most 100 hash-selected correct and 100
hash-selected incorrect Python programs per task, and ordered final
CodeContests-O input probes. It intentionally excludes overlay reference
outputs and iteration histories, unselected submissions, and CodeContests+ 5x
tests. Its manifest therefore remains `pending_executable_equivalence`; the
overlay package authorizes isolation development, not task admission.

## Isolation contract

Generated programs must not execute in the trainer process.

The original registration selected the cluster's Apptainer runtime. Compute
jobs `30183856` and `30183859` failed before candidate execution because the
installation has neither usable user namespaces nor the setuid starter. The
failure logs remain in `var/artifacts/logs/`; no checker replay, policy sample,
or model-dependent decision was made.

### Pre-sampling isolation amendment (2026-07-29)

Replace only the unavailable container mechanism with the following frozen
kernel-enforced runner:

- raw SquashFS image
  `python-3.10-slim-c1e4e6c01eb4.sqsh`, 44,068,864 bytes, SHA-256
  `6d036dfa4a6e216d71e2ddae4cb673c0ff588d2c8ff8af3ed2ab7b3fed309437`,
  derived from OCI base digest
  `sha256:c1e4e6c01eb489c422288b2de34b0761ca316f7a2d98e2c33f47659a73ed108a`;
- verify the image before and after a fresh worker-local `unsquashfs`
  extraction and verify the pinned loader, Python 3.10.20 binary, and
  `libpython` hashes;
- require Landlock ABI 5 or newer, handling every available filesystem right
  and TCP bind/connect right, with read/execute access only to the extracted
  runtime and read/write access only to one fresh candidate directory;
- apply `no_new_privs` and seccomp denials for all socket calls, process
  creation and cross-process controls, namespace/mount operations, kernel and
  privilege interfaces, device creation, and System V IPC;
- close inherited descriptors other than standard input/output/error, clean
  the environment, and execute the image's own loader and Python with isolated
  mode, pinned standard `site` initialization, no bytecode writes, fixed hash
  seed, locale, timezone, and input order; and
- enforce per-candidate CPU, address-space, file-size/output, descriptor,
  process, source-size, and wall-clock limits. The trusted parent owns input
  and captured-output files outside the Landlock-writable directory.

The deterministic isolation smoke uses 1 CPU second, 3 wall seconds, 128 MiB
address space, 64 KiB per output stream, 32 descriptors, and a 256 KiB source
limit. It must show zero boundary/resource violations over adversarial probes
and 24 benign launches, with median launch latency at most 0.25 seconds and
p95 at most 0.50 seconds. Full checker throughput remains a separate gate.
This amendment changes infrastructure only and was frozen before any model
sampling.

The execution contract therefore retains:

- a hash-pinned image and Python version;
- no network;
- read-only base filesystem;
- a fresh writable temporary directory per candidate;
- explicit wall-clock, CPU, memory, process, output-size, and file-count limits;
- no host credentials or repository write mount;
- deterministic locale, hash seed, and input order; and
- fail-closed worker restart and audit logging.

The trusted checker and canonicalizer run outside the candidate container over
captured input/output files.

### Released-checker replay ladder

Before any policy sampling, run a diagnostic overlay replay on all 16 review
tasks using the first five `py3` correct and first five `py3` incorrect exact
code hashes per task. The diagnostic uses every ordered CodeContests-O input,
the pinned released checker and testlib build, and the amended kernel sandbox.
It may identify implementation errors but does not admit or remove a task by
itself.

The first diagnostic exposed two uniform language-runtime mismatches: the
largest valid `1208_C` witness is more than 4 MiB, and known-correct Python
programs can exhaust the source's base-language time limit. Before the
admission replay, freeze a 16 MiB captured-output limit and a 3x Python CPU
multiplier for every task; wall time is the resulting CPU limit plus two
seconds. The trusted released-checker timeout remains two seconds. The
isolation smoke's 64 KiB output attack remains unchanged. This is a uniform
language-compatibility repair, not a per-task or result-dependent acceptance
change.

After repairing only source-independent runner or adapter errors, freeze the
admission replay at the first 20 `py3` correct and 20 `py3` incorrect hashes
per task. Each admitted task must have exact wrapper/released-checker decision
agreement, TPR and TNR at least 0.90, and at least two distinct accepted
ordered-suite behavior keys. A task that lacks two semantic keys in this
source-ordered sample is ineligible for the multi-mode row; do not replace its
sample or tune its identity relation. CodeContests+ 5x inputs remain a second
required suite before final split admission.

The eventual 20+20 overlay run is retained only as a v1 postmortem because v1
had already failed its Python-runtime contract. All 640 replay decisions
completed. Eleven of 16 tasks passed; the audit reported five violations over
four tasks: missing behavior diversity for `1051_B`, TNR and diversity for
`1153_B`, TNR for `1408_A`, and TPR for `1516_C`. The four tasks later named in
the separately frozen v2 protocol all passed this postmortem overlay run, but
that outcome did not admit v1 or select, replace, or tune any v2 task.

ConstructiveCode v2 is registered separately in
`paper/preregistration/constructive_code_executable_slate_v2_20260729.md`.
It freezes one task from each family before source materialization, accepts
only explicit Python-3 labels, excludes every v1 code hash, and selects exactly
100 correct plus 100 incorrect hashes per task. Every selected program must
pass the complete CodeContests-O and CodeContests+ 5x replay gates. The v2
audit additionally requires identical submission sets across suites, direct
task/suite/checker hash identity, zero timeout/isolation/output-bound
violations, and the registered per-execution median and p95 latency bounds.
Passing v2 authorizes split construction and development-only coder viability
sampling; it does not authorize training.

## Base-model viability gate

Use development problems only; do not inspect the frozen evaluation split.

1. Sample the Apache-2.0
   [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
   at the intended group size and token budget.
2. Require enough verified positives to carry an online semantic advantage:
   at least 25% of development prompts have one accepted sample in 16, and at
   least 10% have two distinct accepted behavior keys in 64 samples.
3. If the 0.5B coder fails, perform one preregistered capacity retry with the
   Apache-2.0 1.5B coder. If that also fails, stop; do not curate prompts around
   sampled model successes.

The capacity ladder is fixed before evaluation and selects the smallest viable
model. A coder-model experiment would be reported as an external ModeBench
extension, not silently pooled with the Qwen2.5-0.5B-Instruct estimates.

## Experimental ladder

### Gate A: deterministic smoke

- 16 development prompts across at least four witness families;
- one matched Dr.GRPO arm and one online verified MaxEnt arm;
- seed 43, one pass;
- zero infrastructure, checker, identity, resume, or isolation violations.

### Gate B: compute-matched screen

- 64 training and 64 development prompts;
- seed 43, six passes;
- fixed group size, optimizer steps, token budget, and evaluation draws;
- treatment differs only by the existing online verified MaxEnt mechanism;
- no route library or separated-support repair in the first test.

Advance only if treatment-control `distinct@8` is positive in at least three
of four witness families, no family loses more than .03 `pass@8`, and the
median reward-worker cost stays within the registered compute envelope.

### Gate C: confirmatory extension

- freeze at least 192/96 train/evaluation prompts if source coverage permits;
- seeds 43, 44, 45, 46, and 47;
- 12 passes and the paper's fixed checkpoint policy;
- report greedy pass, `mean@8`, `pass@8`, `distinct@8`, excess multiplicity,
  compile failure, timeout, invalid-output rate, bank support, discoveries,
  controller state, and replay activity;
- require every execution, cadence, resume, equivalence, and separation audit
  to pass.

The primary interpretation gate is positive terminal and fixed-checkpoint AUC
for `distinct@8` in at least three witness families, with no family losing more
than .03 terminal `pass@8`. Peak or best-checkpoint selection is not evidence.

## Candidate comparison

| Candidate | What it establishes | ModeBench fit | Decision |
|---|---|---|---|
| CodeContests+ Verified, multiple-answer rows | Per-problem quality filter, labelled solutions, and semantic output checkers | Strong: executed accepted witnesses can be semantic keys | Primary eligibility and identity source |
| CodeContests-O | Feedback-hardened tests over the same problem family; reuses CodeContests+ checkers | Strong after exact ID/hash alignment and local per-task replay | Preferred test-suite overlay, not a replacement for the Verified filter |
| IPC PDDL benchmarks with VAL | Established executable planning across many domains | Strong trajectory identity, but close to MathIR and still symbolic | Lower-cost fallback if code feasibility fails |
| ICPC Problem Package Format problems with output validators | Human-authored contest packages with standardized validation | Strong, but requires per-package licensing and manual collection | Provenance-focused fallback |
| BigCodeBench-Hard | Practical functions, complex instructions, and diverse library calls | Correctness realism, but usually one specified behavior and no native mode key | Optional correctness-only realism track |
| SWE-bench Verified | Real repository issues and containerized tests | Strong external validity, but patch identity is not semantic outcome identity and inner-loop RL is expensive | Separate future endpoint track |
| DS-1000 | Realistic data-science code across major Python libraries | Useful transfer evaluation, but correct outputs are generally functionally equivalent | Optional correctness-only transfer |

## Paper integration rule

If all gates pass, add `ConstructiveCode` as a separately frozen fifth
ModeBench environment and update the benchmark table, limitations, and
confirmatory comparison. State the coder-model change explicitly if the
capacity gate selects it.

If correctness verification passes but semantic canonicalization does not, the
tasks may be used only as a coding-realism transfer track. Do not report source
ASTs, edit diffs, compiler traces, or normalized output text as solution modes.

If the base-model viability gate fails, record the negative feasibility result
and stop. The existing paper remains ModeBench plus online verified MaxEnt,
with the terminal separated-support causal result and endpoint-only MATH
boundary unchanged.

The superseding MATH-free design is recorded in
[`clean_05b_maxent_vs_drgrpo_eight_environment_plan.md`](clean_05b_maxent_vs_drgrpo_eight_environment_plan.md).
It retains compute-matched objective-equivalent Dr.GRPO, the frozen E58
verified MaxEnt treatment, five seeds, and 12 passes. That cohort stopped
before experiment ID because ConstructiveCode v1 and AntMaze were ineligible;
it does not promote E69 route replay or substitute a larger model.

## Recorded stopping point

1. Completed: pinned source overlap audit and bounded four-family review slate.
2. Completed: task adapters, adversarial tests, official checker builds, and
   hash-pinned networkless compute-node isolation.
3. Passed: execution throughput.
4. Failed: v1 runtime-language contract and released-checker equivalence.
5. Not materialized: CodeContests+ 5x suite for the selected v1 tasks.
6. Not frozen: benchmark train/development/evaluation split.
7. Not run: Qwen2.5-Coder-0.5B viability or any training job.

A future v2 starts from a new preregistration, explicitly selects Python-3-only
programs, and reruns both overlay and CodeContests+ 5x gates before policy
sampling.

## Sources

- [CodeContests+ dataset and license](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus)
- [CodeContests+ paper](https://arxiv.org/abs/2506.05817)
- [CodeContests-O dataset and license](https://huggingface.co/datasets/caijanfeng/CodeContests-O)
- [CodeContests-O paper](https://arxiv.org/abs/2601.13682)
- [VAL PDDL plan validator](https://nms.kcl.ac.uk/planning/software/val.html)
- [IPC 2023 benchmark domains](https://ipc2023-learning.github.io/)
- [ICPC Problem Package Format](https://icpc.io/problem-package-format/)
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [DS-1000](https://ds1000-code-gen.github.io/)
- [Qwen2.5-Coder family](https://qwenlm.github.io/blog/qwen2.5-coder-family/)
