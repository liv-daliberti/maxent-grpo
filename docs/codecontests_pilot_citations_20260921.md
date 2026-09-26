# CodeContests pilot: citations and interpretation

Prepared on 2026-09-21 before interpreting the new pilot. This is a prospective source note, not an experiment result or a paper edit.

## Connection to the cited literature

The existing bibliography key `brown2024monkeys` refers to Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini, *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*, arXiv:2407.21787 (2024; verified against v3).

- [Section 2](https://arxiv.org/html/2407.21787v3#S2) motivates independently sampled Python programs, execution-based correctness, and pass@k.
- [Section 4.2.2](https://arxiv.org/html/2407.21787v3#S4.SS2.SSS2) identifies CodeContests false negatives: multiple valid outputs are judged against one reference output; malformed generated inputs cause another failure.
- [Appendix A.2](https://arxiv.org/html/2407.21787v3#A1.SS2) uses 140 test problems without image tags, 10,000 samples/problem, temperature 0.6, top-p 0.95, two training examples, and a 1,024-token limit. It concatenates public, private, and generated tests.

Our proposed T=1.0, top-p=1.0 development pilot is inspired by this repeated-sampling setting, **not an exact reproduction**. It has different tasks, models, prompts, sample counts, and checking. Pass@k describes problem coverage; it does not establish diversity among correct outcomes.

[CodeContests+](https://arxiv.org/pdf/2506.05817), Section 4.3 and Appendix B, supplies specialized output checkers for problems with multiple valid answers. Its problems originate in competitive programming; its additional tests and custom checkers are generated with an LLM agent. Describe the application as **human-authored contest problems with generated evaluation infrastructure**, not wholly non-synthetic data. Checker release alone is insufficient validation: the paper discusses residual checker errors.

Verified bibliography metadata from the [primary arXiv record](https://arxiv.org/abs/2506.05817):

```bibtex
@misc{wang2025codecontestsplus,
  title         = {{CodeContests+}: High-Quality Test Case Generation for Competitive Programming},
  author        = {Wang, Zihan and Liu, Siyao and Sun, Yang and Li, Hongyan and Shen, Kai},
  year          = {2025},
  eprint        = {2506.05817},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2506.05817}
}
```

## Prospective paragraph

> Motivated by execution-verified repeated sampling in Large Language Monkeys (Brown et al., 2024), we will test whether mode-balanced replay improves the coverage and diversity of correct programs on human-authored constructive programming problems. We will use audited CodeContests+ checkers (Wang et al., 2025), which can accept distinct valid outputs for the same input. Modes will be canonical execution-output signatures on a fixed test suite. The pilot will first establish whether the initial model discovers multiple verified signatures; it will not by itself establish a Re:Max advantage.

## Development task review

The review below inspected only the three development statements and their checker/adapter metadata; it did not inspect held-out evaluation statements or solutions.

| Task | Valid output objects | Interpretation of the present mode key |
| --- | --- | --- |
| [359B: Permutation](https://codeforces.com/problemset/problem/359/B) | A permutation of 1 through 2n meeting an absolute-difference identity. | The ordered sequence is retained. This includes different pair orderings and other mathematical symmetries; distinct keys are different accepted witnesses, not necessarily distinct algorithms. |
| [988A: Diverse Team](https://codeforces.com/problemset/problem/988/A) | A set of k student indices with distinct ratings, or an impossibility verdict. | Sorting indices correctly removes output-order variation. Choosing different students with equal ratings remains a distinct team, even when the selected rating set is identical. |
| [1399D: Binary String To Subsequences](https://codeforces.com/problemset/problem/1399/D) | A partition of character positions into the minimum number of alternating subsequences. | Sorting member indices and groups removes arbitrary group labels. Different assignments of positions remain distinct even if their subsequence strings coincide. |

Concrete small witnesses demonstrate the distinction without modifying the benchmark: for 988A's first public example, teams {1,2,5}, {2,3,5}, and {2,4,5} differ in student identity but all select ratings {12,13,15}. For 1399D with string `0011`, partitions {{1,3},{2,4}} and {{1,4},{2,3}} both use two `01` subsequences but assign positions differently. These examples concern output semantics, not observed pilot generations.

### Missing statement image in 359B

The frozen local `359_b/task.json` still contains `<image>` in place of the defining equation, and the shared evaluator's `_prompt` inserts the statement unchanged. This creates an incomplete prompt. Do not interpret an unrepaired run as a fair task-capability measurement.

The equation was independently recovered and visually read from the [original Codeforces image](https://espresso.codeforces.com/b54693338584d5268d5ec3ab8c4f8e90b87dea39.png), linked by the original problem statement:

```text
sum_{i=1}^n |a_{2i-1} - a_{2i}| - |sum_{i=1}^n (a_{2i-1} - a_{2i})| = 2k
```

Image SHA-256: `7386a0480bd5b1bcfdca7729b5a041419b3c3641ce8096ee127fe239efa09060`.

The separate `codecontests_pilot_prompt_overlay_20260921.json` records this transcription and original/repaired statement hashes. Apply it only to the new pilot, preserving all frozen artifacts. This text rendering is a documented input repair, and is another difference from Monkeys' image-exclusion protocol.

### Parser and validity limits

Inspection of `src/oat_drgrpo/constructive_code_adapters.py`, the three released checkers, and pinned `third_party/testlib/testlib.h` found no immediate disagreement affecting accepted outputs. Although the 359B and 1399D adapters do not themselves require end-of-output, testlib's `InStream::quit` checks for extra output before returning acceptance. Canonicalization is gated by that released-checker decision, so trailing garbage cannot create an accepted mode. 988A additionally enforces end-of-output explicitly. The 1399D checker recomputes the optimum number of groups and checks alternation; group label permutations are removed by the adapter.

The released 1399D checker permits t up to 200,000 while the statement restricts t to 20,000. A direct read-only audit of all 52 `plus_5x_inputs.jsonl.gz` and 24 `overlay_inputs.jsonl.gz` development inputs checked t, n, binary-string lengths/content, total n, and input exhaustion; no statement-bound violations were found. Thus the broader checker bound is not an observed problem in these suites.

An execution signature establishes behavior on the fixed suite, not full program equivalence or general correctness on every legal input. Different witnesses can arise from tie-breaking within one algorithm. Report the result as verified output diversity; stronger claims about algorithmic strategy need additional evidence. A successful three-task development pilot is not sufficient alone for a convincing held-out application comparison.
