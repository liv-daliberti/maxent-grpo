# Wider CodeContests development screen: metadata shortlist

This is a read-only review of existing metadata, prepared 2026-09-21. No model generations, new downloads, task-data edits, or training were performed. The only new artifact is this note. Difficulty labels below are implementation judgments from the statements, not verified Codeforces ratings.

## Recommendation

Widen the human-authored constructive-programming pool before changing application. There are enough plausible tasks to test whether the earlier narrow pilot was representative. First establish that generation, code extraction, execution, and output limits behave correctly on known simple problems. More tasks cannot repair a broken evaluation pipeline.

Use the 24 candidates below as a prospective screening pool spanning four output families. Start with the 12 first-tranche rows, which include two previously inspected development anchors. Preserve the original held-out tasks `361_B`, `1294_C`, and `149_C`; none is in this shortlist. Freeze a new problem-level development/evaluation split before observing any Re:Max-vs-MaxRL differences. A pilot selected for discoverable modes should be reported as a feasibility screen; the application result needs held-out problem coverage and all exclusions disclosed.

These are human-authored contest problems rather than generated toy tasks. They remain competition exercises, not production user workloads. CodeContests+ tests and checkers are generated infrastructure, so avoid describing the entire dataset as non-synthetic.

## Pool accounting and provenance

Source: `var/artifacts/constructive_code_candidate_source_index.json`, SHA-256 `00ae7293af83314b8c49a68d867ac07a4c4b81b580fd2174494c88fbf35a2602`.

- 395 rows, all Codeforces; 375 distinct normalized statement hashes. Twenty pairs are alternate contest IDs for the same statement. Split and deduplicate by statement hash, not ID alone.
- 83 statements contain image tags; 312 do not. There are 170 image-free rows whose recorded source TPR is 1.0 and TNR is at least .95. These counts do **not** certify multiple semantic outputs or trustworthy checkers.
- The original filter required a checker to read `inf` and `ouf` without `ans`, at least two correct Python submissions, source TPR/TNR >= .9, and matching normalized statement/checker hashes across the two sources. Reference-independent checking does not imply multiple answers.
- The metadata index excludes submission code, tests, generators, and per-submission results. It is not a runnable 395-task dataset. New candidates need targeted task materialization, input validation, positive/negative replay, and checker/canonicalizer agreement checks.
- Pinned sources recorded in the index: `caijanfeng/CodeContests-O` at `1a765191567b429f633bbd1c6e67b5890dfaf267`; `ByteDance-Seed/Code-Contests-Plus` at `96c850540fade31d384a25766461e0da6b08f5fc`.

All 24 shortlisted rows have **no image tag**. This is only a textual screen: flattened exponents and formatting issues still require review. The quality column is the dataset's recorded TPR/TNR, rounded to three decimals; Py is its correct-Python-submission count, not distinct-mode count or independent evidence of correctness. All 24 statements were read; their released checker logic was inspected, including the two existing anchors. No new checker was compiled or run for this note.

## First tranche: 12 candidates

Family S = ordered sequences, strings, or position-bearing tuples; U = unordered subsets/multisets; M = matrices with fixed coordinates; P = partitions/allocations.

| ID and title | Family | Meaningful output and exact identity | Quality; Py | Main limitation or admission check |
| --- | --- | --- | --- | --- |
| **1454_A — Special Permutation** | S | A derangement; retain the integer permutation in position order. | 1.000/1.000; 126 | Very easy capacity check; n=2 has only one answer. Multiple cycle constructions at larger n may still reflect simple tie-breaking. |
| **1513_A — Array and Peaks** | S | A permutation with exactly k peaks; retain sequence or canonical impossible verdict. | 1.000/1.000; 112 | Straightforward construction; impossible and n=1 cases contribute no within-input diversity. |
| **1569_A — Balanced Substring** | S | A nonempty interval [l,r] with equal a/b counts; retain the ordered endpoint pair. | 1.000/.958; 143 | Different locations are different witnesses even if substring text is equal. Bounds small (n <= 50); good inexpensive verifier. |
| **1352_B — Same Parity Summands** | U | k positive same-parity summands totaling n; sort the multiset, preserving multiplicity. | 1.000/.949; 1478 | Different summand orders must collapse. Recorded TNR is below .95, requiring careful negative replay. |
| **1016_D — Vasya And The Matrix** | M | Nonnegative matrix realizing specified row/column XORs; retain coordinate-indexed cells. | 1.000/1.000; 80 | Input-specific row/column constraints make alternatives concrete. Plain text flattens exponents (`109`, `2·109`); verify faithful source rendering before new prompts. |
| **1360_G — A/B Matrix** | M | Binary matrix with exactly a ones per row and b per column; retain fixed binary rows. | 1.000/1.000; 270 | Common construction is short; many alternatives are row/column symmetries. Checker is line-aware and expects compact binary rows. |
| **361_A — Levko and Table** | M | Integer square matrix whose row and column sums equal k; retain fixed cells. | 1.000/.990; 445 | Very easy construction; allowed entries include negatives. Matrix symmetries and arbitrary free values limit an algorithm-diversity interpretation. |
| **244_A — Dividing Orange** | P | Allocate numbered items into equal-sized groups, each assigned to a specified child and containing that child's anchor. Sort within each group, **retain child order**. | 1.000/.959; 231 | Strong concrete allocation semantics. Sorting the outer groups would erase real child assignments; output up to 900 indices. |
| **1102_B — Array K-Coloring** | P | Partition indices into k nonempty groups, no repeated array value within a group. Sort member indices and then groups. | 1.000/1.000; 347 | Remove arbitrary color labels; up to 5,000 positions. Adapter already exists, but this note does not newly admit the task. |
| **1408_A — Circle Coloring** | S | Choose one of three supplied values per fixed position so adjacent values differ cyclically; retain chosen value vector. | 1.000/.989; 641 | Good fixed-position alternatives; values are supplied input data, not anonymous color labels. Existing adapter must preserve actual values. |
| **988_A — Diverse Team** *(existing development anchor)* | U | A k-element student-index set with distinct ratings; sort indices. | 1.000/1.000; 1102 | Different students with equal ratings remain distinct teams. Useful pipeline continuity; avoid claiming every new team is a new strategy. |
| **1399_D — Binary String To Subsequences** *(existing development anchor)* | P | Minimum partition of character positions into alternating subsequences; sort indices within groups and quotient group labels. | 1.000/1.000; 535 | More demanding than the simplest candidates. Input suites previously reviewed; maximum output can reach 200,000 labels. Different index partitions may produce identical substring texts. |

## Second tranche: 12 reserves

| ID and title | Family | Meaningful output and exact identity | Quality; Py | Main limitation or admission check |
| --- | --- | --- | --- | --- |
| **1323_A — Even Subset Sum Problem** | U | Any nonempty even-sum subset of input indices; sort indices. | .970/1.000; 939 | Very easy, but source checker has bespoke line parsing and permissive stringstream parsing; test accepted spellings against the canonicalizer. |
| **1073_A — Diverse Substring** | S | Any substring whose largest character count is at most half its length; retain substring **text**, not a guessed occurrence index. | .990/1.000; 607 | Repeated occurrences of identical text are the same returned answer. Line/whitespace parser needs replay; source TPR below 1. |
| **1380_A — Three Indices** | S | A triple i<j<k whose middle permutation value exceeds both others; retain ordered triple. | .990/.968; 969 | Output is tiny, but checker performs a nested triple existence search, including for YES outputs. Benchmark CPU cost before admission; n can be 1,000. |
| **1095_C — Powers Of Two** | U | k powers of two summing to n; sort the multiset. | .990/.968; 422 | True alternatives exist after order removal, but some inputs have a unique multiset. k can reach 200,000, stressing stdout and canonicalization limits. |
| **1352_G — Special Permutation** | S | Permutation with every adjacent absolute difference in [2,4]; retain sequence. | 1.000/1.000; 665 | Slightly trickier construction; reversal symmetry is an easy source of modes. n<=3 is impossible. |
| **482_A — Diverse Permutation** | S | Permutation with exactly k distinct adjacent absolute differences; retain sequence. | 1.000/.990; 207 | Existing adapter. `483_C` is a duplicate-statement alias: keep only one. Flattened `105` bound should be source-checked; potentially large output. |
| **1339_B — Sorted Adjacent Differences** | S | Reorder supplied numbers so adjacent absolute differences are nondecreasing; retain value sequence. | 1.000/1.000; 975 | Duplicate values must not create identity through source occurrence labels. Total array length can be 100,000; audit runtime/stdout limits. |
| **1371_D — Grid-00100** | M | Binary square grid with k ones and minimum row/column imbalance; retain grid, with the required optimum score treated as redundant metadata. | 1.000/1.000; 431 | Checker computes optimum and grid score directly. Larger output (up to 90,000 cells); symmetric placements count as different witnesses. |
| **1038_B — Non-Coprime Partition** | P | Partition 1..n into two nonempty groups whose sums have gcd>1. Sort each group and then the two groups. | 1.000/1.000; 395 | Easy witness construction. Checker verdict spelling is exactly `Yes`/`No`; n can reach 45,000. |
| **1051_B — Relatively Prime Pairs** | P | Perfect pairing of all integers in [l,r], each pair coprime. Sort inside each pair and then pairs. | 1.000/.988; 438 | Existing pair-partition adapter. Must preserve up-to-10^18 integer values exactly; interval size can reach 300,000. Small intervals may have only one pairing. |
| **545_B — Equidistant String** | S | Binary string equally distant from two supplied strings; retain output bits by position. | 1.000/.989; 341 | Genuine choices in which differing positions match each input. Plain text flattens the length bound; line-oriented checker needs whitespace tests. |
| **1352_F — Binary String Reconstruction** | S | Binary string with specified counts of 00, mixed, and 11 adjacent pairs; retain exact string. | 1.000/1.000; 628 | Short outputs (<=301 characters/case). Released checker uses anchored pattern `^[01]+$`; smoke-test compatibility with pinned testlib before admission rather than trusting the rate field. |

## Why these are not the original filter's false positives

Examples of genuine alternatives after canonicalization, used only to reason about semantics (not proposed synthetic benchmark additions): for 1095C, n=10,k=4 permits multisets [1,1,4,4] and [2,2,2,4]. For 1352B, n=12,k=3 permits [2,2,8], [2,4,6], and [4,4,4]. For 244A with two children, two items per child, and anchors 1 and 2, allocations ({1,3},{2,4}) and ({1,4},{2,3}) differ even after removing within-child order. For 1016D, 2x2 zero row/column XORs allow both all-zero and all-one matrices. These establish possible semantic multiplicity, not observed support under the model or stored suites.

Exclude the following tempting rows despite reference-independent checkers:

- **1015_A Points in Segments:** the set of uncovered points is uniquely determined; only list order varies.
- **1154_A Restoring Three Numbers:** the recovered unordered triple is uniquely determined by the pair sums and total; variable permutation is not a new mode.
- **749_A Bachgold Problem:** a maximum-cardinality prime decomposition is all twos, with one three when n is odd; its sorted multiset is unique.
- **987_A Infinity Gauntlet:** the set of missing stones is uniquely determined; list order is immaterial.

`359_B` is not included in the wider pool because its source statement has a missing equation image; the separate verified prompt overlay already handles its narrow diagnostic use. Do not silently count a repaired image task as an unchanged image-free source problem.

## Admission and interpretation limits

The task is to generate a program from an existing human statement; the program's signature is its canonical accepted output across a fixed input suite. One test input is not a new independent training question. Splitting cases from one statement between train and test cannot establish generalization to unseen problems. Twenty-four metadata candidates may shrink substantially after checker validation and baseline support measurement; this note does not claim 24 runnable tasks or a validated 200-GPU-hour training recipe.

Canonicalization should remove only declared representational freedom: set order, irrelevant summand order, anonymous group labels, and formatting. Fixed index choices, child assignments, matrix coordinates, and string bits remain observable outcomes. These identities support output-diversity claims. Many alternatives are simple tie-breaks or mathematical symmetries; an algorithmic-strategy-diversity claim would require additional evidence.

Large **executed stdout** is separate from long **generated program text**. Candidates with 100,000+ output entries may require larger execution-output capture/hash limits while still needing only a short Python program. Check the relevant runtime, memory, stdout, and canonicalization bounds before interpreting zero reward as model incapacity. Short-output reserves such as 1569A or 1380A can help distinguish those failure modes, although 1380A has a potentially expensive checker.

Report every screened problem's pass rate, format/execution failures, truncation rate, and correct-mode support. If a functioning harness and a broader easy-task screen still yield too few problems with multiple verified modes, change course then; the present metadata alone does not justify abandoning this application or launching a full training sweep.
