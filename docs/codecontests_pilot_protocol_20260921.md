# CodeContests+ capability pilot, 2026-09-21

Prospective development protocol, frozen before Coder-7B sampling. This is
a new pilot; it neither resumes nor replaces the stopped July experiment.
No appendix or paper result is authorized by this document.

## Scope and sources

The user authorized a pilot before deciding whether to add an application
to the appendix. CodeContests+ is the dataset; Large Language Monkeys
(Brown et al., 2024, arXiv:2407.21787) motivates repeated-sampling evaluation
and the need to accept multiple correct outputs. This is not a reproduction
of its 10,000-sample CodeContests experiment. Cite CodeContests+ (Wang et al.,
2025, arXiv:2506.05817) for the generated tests and custom checkers.

Use the three previously designated development problems 359_B, 988_A and
1399_D. Never load evaluation problems 361_B, 1294_C or 149_C for sampling.
Keep the v6-audited per-problem suites and semantic canonicalizers. Recheck
dataset/checker hashes and representative known submissions before accepting
model results. The old source snapshot directories were deleted by repository
cleanup; freeze a new dependency snapshot and record its distinct identity.

359_B's frozen dataset statement contains an image placeholder in place of
its defining equation. Before any generation, replace only that placeholder
with a faithful transcription of the original Codeforces equation image,
using `codecontests_pilot_prompt_overlay_20260921.json`. Record both statement
hashes and the image provenance. This repairs a missing part of the public
question; it supplies neither a solution nor hidden-test information.
988_A and 1399_D retain their original statements. Do not present the repaired
359_B result as a pure model-size comparison with the July pilot.

## Frozen capability stage

- Model: Qwen/Qwen2.5-Coder-7B-Instruct, revision
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- Zero-shot complete Python 3 programs; no retrieval, references, feedback,
  continuation, repair, SFT or training during this stage.
- 64 independent candidates per development task, 192 total.
- Seed: 77101 + 10000 * task_index + sample_index.
- Temperature 1, top-p 1, no top-k truncation, maximum 1024 output tokens,
  8192-token context; generation batches of 16.
- Preserve exact raw completions and executed source incrementally. Strip
  only the existing exact complete bare/Python Markdown fence convention.
- Execute through the pinned Python 3.10.20 Landlock/seccomp sandbox and
  hash-checked released testlib checker, with no network and unchanged
  per-task resource limits. Programs never execute in the trainer process.
- Each accepted key is the canonical tuple of executed valid witnesses.
  Program strings, AST hashes and formatting are not mode keys. Re-execute
  every accepted model program independently on the full suite and require
  identical acceptance and mode key; retain both receipts and treat any
  instability as an audit failure.

## Metrics and decisions

Report all three tasks and all 192 outcomes. Estimate pass@1, pass@8 and
pass@32 from the 64 draws with the without-replacement combinatorial
estimator. Also show prefix coverage, unique accepted mode count and the
distribution over accepted keys.

PCMD is `1 - sum_c n_c*(n_c-1)/(m*(m-1))`, where m is the accepted count.
Report it only when m >= 30, with eligibility explicit. Never substitute
zero for missing diversity. This small development slate gives a capability
diagnostic, not a held-out generalization estimate.

The capability decision passes only if aggregate accepted fraction is at
least 10%, at least two of the three tasks each yield at least two accepted
modes, all requests have terminal records, and there are no hard checker,
canonicalization, isolation, resource-bound or protocol violations.
Separate audit failure from insufficient capability. A failed stage stops
online training under this proposal, with all failures retained.

A passing capability stage permits a separately specified paired MaxRL /
current Re:Max engineering smoke within the remaining pilot allocation.
It does not establish a treatment benefit or authorize the six-run study.
Any paired smoke must start both arms from the same original checkpoint,
use policy-generated verified banks, and apply only uniform replay as the
treatment difference; the historical E58 multi-term trainer is ineligible.

## Compute and execution

First generation/checking job: one A6000, at most one allocated GPU-hour,
8 CPUs, 64 GiB requested host memory, no automatic requeue. This hardware
choice reflects available cluster capacity; report actual device and GPU
hours and do not relabel them as measured A100 hours.

The complete development pilot retains a hard ceiling of 12 aggregate
GPU-hours across capability, any paired smoke, and failed attempts. Network
model download and CPU-only preparation are separate from GPU use. Record
scheduler allocation/elapsed time, generation and verification wall times,
token counts, model and prompt identities, raw completions and audit receipts.
No full-scale jobs or paper changes follow automatically.
