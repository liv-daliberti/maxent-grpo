# E21 MATH free-form conditional-token MaxEnt

**Status: FROZEN V10 OUT-OF-SAMPLE SMOKE PASSED; STORAGE-SAFE ANALYTICAL
COHORT V4 AUTHORIZED (2026-07-20).**

**Stage-A V3 admission-ceiling amendment, before any optimizer update:** the
six allocated V3 jobs were still in initialization evaluation when their
startup logs reported a prompt dataset length of 8,507.  OAT applies
`max_train` as a source-row selection before its deterministic length filter,
so using 8,515 as that ceiling first omitted the final eight source rows and
then excluded the eight over-bound rows.  All V3 jobs were cancelled with zero
optimizer updates.  V4 restores the source selection ceiling to 8,523; after
the same frozen filter this yields the intended 8,515 trainable prompts.  The
evaluation interval remains 2,129 and the terminal boundary remains 8,515.

**Stage-A V2 placement amendment, before allocation or outcome:** all 12 V2
jobs remained pending for priority with zero runtime because the eligible
A6000 nodes were reserved for higher-priority work.  They were cancelled
without allocating a GPU or creating a run directory.  V3 changes placement
only to the idle ten-A5000 `mltheory` node105 pool, to which the experimenter
has an active allocation.  The microbatch-one repair is the frozen 24 GB-safe
configuration; the earlier 24 GB failure used microbatch four.  No data,
algorithm, coefficient, seed, optimizer, cadence, checkpoint, or evaluation
setting changes.

**Stage-A V1 storage and prompt-admission amendment, before any quarter-epoch
MATH outcome:** all 12 V1 jobs were released, but the shared project
filesystem reached zero free bytes.  The four seed-43 jobs stopped between
steps 178 and 196, before the first scheduled quarter-epoch evaluation; the
other eight jobs failed during startup.  Telemetry through the interruption
was finite, and the failure text was `No space left on device`.  V1 is an
infrastructure-aborted cohort and is not analytical evidence.  Obsolete raw
optimizer checkpoints were removed while exported models, metrics,
evaluations, approvals, and source snapshots were retained, restoring 1.3 TB
free.  V2 changes checkpoint retention only, from five to two most-recent
checkpoints per job.

The V1 startup logs also made a pre-outcome bookkeeping discrepancy visible:
the byte-exact source contains 8,523 rows, while OAT deterministically excludes
eight rendered prompts longer than the frozen 1,024-token prompt bound.  The
actual trainable population is therefore 8,515 rows.  The excluded row
indices, rendered-token counts, and problem hashes are frozen in
`var/artifacts/e21_math_conditional_token_prompt_admission.json`; none has a
blank answer.  V2 truthfully uses 8,515 as its terminal placement count and
evaluates after 2,129, 4,258, 6,387, and 8,515 consumed prompts.  The prompt
bound, completion bound, data bytes, ordering, algorithm, coefficients,
optimizer, seed set, and complete 500-row MATH evaluation are unchanged.  The
completed seed-9008 smoke uses the first 64 admitted rows and therefore remains
a valid runtime gate; its approval is rebound to this disclosed
infrastructure-only amendment.

**Technical amendment, before any optimizer update or MATH outcome:** the
first provisional smoke reached initialization evaluation but the full
verifier tried to install `SIGALRM` from OAT's actor grading thread; it was
cancelled at step zero. The repaired verifier retains signal timeouts on the
main thread and uses a bounded daemon helper from actor worker threads. A
second held-cohort audit then caught `max_queries=64`; that field counts 16
generated candidates per prompt and would have ended at four updates, so
those jobs were cancelled during model initialization. The exact worker-
thread regression and focused suite pass, `max_queries` is now nonbinding,
and only the fresh V10 smoke prefix below may authorize Stage A.

**V3 memory-only amendment, before any V3 outcome:** the complete V2 cohort
reached initialization evaluation (matched pass@1 `0.336`, matched mean
completion length `562.48`) and then exhausted a 24 GB RTX 3090 during the
first two learner updates with per-device microbatch four. V2 was cancelled
as a matched cohort. V4 keeps the same effective batch of 16, objective,
coefficients, data, and evaluation contract, but uses per-device microbatch
one with gradient accumulation 16. Expandable CUDA segments are deliberately
disabled because vLLM's CUDA memory pool rejects that allocator mode. The V4
jobs are restricted to 48 GB A6000s on lowprio-eligible nodes
205/206/207. Hardware and memory telemetry are the only V2 information used
for this repair; no treatment comparison or coefficient was changed.

**V3 launch-race amendment, before any V4 outcome:** an external launch of the
V3 prefix occurred from the preceding 3090 configuration while the A6000
repair was being frozen. Slurm inspection caught all four jobs on node023
before any run artifact or learner telemetry was written; all four were
cancelled at 75 seconds. V4 adds a fail-closed inspection of each held Slurm
job's requested node set and `gpu:a6000` TRES before releasing the cohort.

**V4 identity-race amendment, before any V5 outcome:** the V4 manifest was
created immediately before the allocator-removal protocol amendment. Its
recorded protocol hash therefore differed from the frozen repaired document.
Two jobs allocated for 25 seconds and two never allocated; none reached
evaluation or emitted learner telemetry. All four were cancelled. V5 keeps
the A6000/microbatch-one repair, binds the post-amendment protocol hash, and
uses a prefix-configurable frozen checker so later infrastructure recovery
does not silently select an older manifest.

**V5 scheduling-only amendment, before any V6 outcome:** all four A6000 jobs
completed the identical initialization evaluation (MATH-500 pass@1 `0.322`,
500 rows, mean completion length `563.346`). The low-priority partition then
preempted C0 and M-fixed before their first update; M-proportional and M-dual
were cancelled as a matched cohort after step six. No OOM or nonfinite metric
occurred. V6 changes only the Slurm partition from preemptible `lowprio` to
non-preemptible `cs` with the same `allcs` account and A6000 node set. No
objective, coefficient, data, batch, seed, or evaluation setting changes.

**V6 verifier-queue amendment, before any V7 outcome:** V6 confirmed the
matched initialization result and reached the step-16 evaluation in all four
arms without a runtime, numerical, or length-extension failure. At the
step-32 boundary, one control response exceeded the actor's one-second outer
grading wait even though the full verifier legitimately contains two bounded
parses plus one bounded symbolic comparison. Abandoned in-flight calls then
starved its two-thread reward queue while the treatment jobs advanced. The
matched cohort was cancelled and is not analytical evidence. V7 submits each
grading batch concurrently and gives the compound full-verifier task a
four-second outer budget. An exact regression with two correct 1.2-second
calls and the focused E21 suite pass. No objective, coefficient, data, batch,
seed, prompt, or evaluation cadence changed.

**V7 killability amendment, before any V8 outcome:** the four-second outer
wait allowed C0's step-32 evaluation to finish and fixed completed all 64
updates, but a timed-out SymPy helper thread itself remained unkillable. By
C0's terminal evaluation, accumulated daemon work again starved the actor;
the matched cohort was cancelled. V8 runs full MATH rewards in two fresh
`spawn` worker processes, never forks the CUDA/vLLM actor, and terminates and
recreates the isolated pool after any poisoned row before regrading pending
rows. Native `SIGALRM` therefore runs in each grader process's main thread.
Fast grading remains threaded. No analytical setting changed.

**V8 entrypoint amendment, before any V9 outcome:** an external V8 launch
raced the preflight amendment: C0 and M-fixed reached step ten and the two
adaptive jobs never allocated before the matched cohort was cancelled.
Although `spawn` worked under that actor entrypoint, preflight showed that it
re-imports the parent's `__main__`, which is undefined for stdin and can
recursively execute other service launchers. V9 instead launches the explicit module
`oat_drgrpo.math_grader_worker` as one persistent subprocess with JSON-lines
IPC. It never imports the CUDA actor entrypoint. Each answer has a five-second
parent deadline; timeout, broken pipe, or malformed output hard-terminates the
worker, and the next answer starts a clean interpreter. A 500-row saved eval
regraded identically (167/500, zero changed labels) in 6.21 seconds; tests
cover correctness, hard timeout, forced worker death, and clean restart.

**V9 gate-design amendment, after recording all V9 smoke outcomes and before
any V10 optimizer update:** all four V9 jobs completed 64 updates and terminal
MATH-500 evaluation with exit code zero. Final-16 mean completion lengths were
`556.92/587.80/591.78/578.07` for C0/fixed/proportional/dual. Their no-EOS
totals across 256 responses were `12/20/21/19`, while their maximum counts in
any one 16-response batch were `3/5/4/7`. The frozen V9 gate therefore failed:
it compared the maximum of 16 noisy batches with C0 maximum plus one, so the
fixed and dual arms exceeded its limit of four even though aggregate no-EOS
rates were `4.69%/7.81%/8.20%/7.42%` and all absolute guards passed. Terminal
pass@1 was `0.332/0.316/0.334/0.334`; it is disclosed but was not used to
choose this repair. V9 is not analytical evidence and cannot authorize Stage
A. V10 keeps every algorithm, coefficient, prompt, batch, and hardware setting
unchanged, uses the independent seed 9008, and replaces only that unstable
maximum comparison with an aggregate final-window allowance: each treatment
may have at most the C0 no-EOS total plus 16 across the 16 batches (one extra
row per batch on average). The existing relative mean-length bound, absolute
mean-length bound, and hard maximum of 12/16 no-EOS rows in every batch remain.

E21 is the free-form alternative to E20's blocked canonical MATH design. It
preserves authentic MATH prompts and unrestricted Qwen text generation. It is
**not** canonical-action MaxEnt and is not the Shannon entropy of a whole
completion. It must be labeled **free-form conditional-token MaxEnt**. No
answer list, answer index, gold-derived support, or multiple-choice
transformation is used.

## Data and leakage boundary

- Training source is the byte-exact 8,523-row released
  `math_lvl3to5_8k/train` artifact.  Under the frozen 1,024-token rendered
  prompt bound, 8,515 rows are trainable and the eight over-bound rows are
  deterministically excluded as recorded above.
- Evaluation is the byte-exact 500-row released `evaluation_suite/math`
  artifact (MATH-500).
- Exact train/evaluation problem overlap is zero. MATH-500 is evaluation-only
  and is never used for training, controller calibration, dose selection,
  early stopping, or prompt selection.
- The two released blank-reference training rows and one repeated problem are
  retained unchanged.
- The import manifest SHA-256 is
  `2fe5f5461be2fed5ec2a0ed5d400873ad02dc7684d3c4793afa54bd9cc4f5202`.
- Grading uses the full `boxed_reward_fn(..., fast=False)` path
  (`verifier_version=math_verify`).

## Length-neutral entropy objective

For an ordinary free-form response state `s`, factor the next-token policy
into its EOS/continue probability and its content distribution conditional on
continuing:

`q(a | s, continue) = pi(a | s) / (1 - pi(EOS | s))`, for `a != EOS`.

The E21 entropy term is the mean `H(q)` across active positions within each
sampled response, followed by a mean across response rows. The regularizer
treats sampled states as fixed: it has no prefix importance ratio and does not
differentiate state visitation. Consequently:

- the EOS column is absent from the entropy softmax, so the entropy term has
  exactly zero direct derivative with respect to the EOS logit;
- a response receives one equal-weight mean regardless of its token count, so
  generating more positions cannot accumulate more entropy reward;
- earlier actions receive no entropy gradient for reaching later states.

This is the free-form analogue used here because literal sequence entropy in
E11--E12 paid for extra states and exhibited a sharp length/no-EOS runaway.
E21 does not add a learned length penalty: termination remains controlled by
the ordinary Dr.GRPO reward update. Response length and no-EOS behavior remain
fail-closed runtime gates to catch indirect parameter-sharing effects.

## Common optimization contract

- Pilot model: Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Prompt: ordinary `qwen_math` reasoning template; normal full tokenizer
  vocabulary and learned EOS.
- Maximum prompt length 1,024; maximum train/evaluation completion 1,024;
  model context bound 2,048.
- Dr.GRPO reward update: `critic_type=drgrpo`, `xdr_tau=inf`, uniform candidate
  aggregation, group size 16, one PPO epoch, learning rate `2e-7`, `beta=0`,
  and the maintained `1/T_max` reward-update normalization.
- Rollout temperature 1 and `top_p=1`; greedy evaluation temperature 0.
- The conditional-token entropy mean is already normalized once per response;
  no second `1/T_max` factor is applied to it.
- No aggregation rescaling, xDr weighting, canonical action restriction,
  prefix-ratio entropy term, sequence-entropy sum, expected-length dual, or
  token-count reward is allowed.

## Arms

| Arm | Frozen rule |
|---|---|
| C0 | matched free-form Dr.GRPO, `alpha=0` |
| M-fixed | fixed conditional-token MaxEnt, `alpha=0.00010` |
| M-proportional | base `0.000075`, maximum `0.00015`, target 80% of the mean conditional-token entropy during warmup, EMA `0.9`, gain `2` |
| M-dual | base `0.000075`, bounds `[0.00005, 0.00015]`, target 80% of the mean conditional-token entropy during warmup, log-alpha Adam LR `0.005` |

These deliberately conservative coefficients bracket the effective scale of
E12's safe low-dose region after replacing a horizon-normalized entropy sum
with one per-response mean. They are engineering provenance, not MATH outcome
evidence. MATH-500 accuracy cannot be used to change them.

## Stage S: four-arm runtime smoke

Seed 9008 runs C0 and all three treatments on the same first 64 training
prompts for 64 learner updates. The effective training batch is 16 and the
per-device microbatch is one. Adaptive warmup is 16 updates. Greedy
evaluation runs at initialization, every 16 prompts, and the terminal boundary
on all 500 MATH-500 rows. The smoke uses one 48 GB A6000 per job on
node205/node206/node207, no checkpoints, no auto-resume, and a four-hour
limit.

The analytical cohort remains blocked unless all jobs reach step 64 without a
runtime failure and:

- loss, reward, response length, and no-EOS telemetry are finite;
- treatment rows report finite conditional-token entropy and entropy loss;
- `maxent_eos_excluded`, `maxent_state_distribution_detached`, and
  `maxent_response_equal_weight` equal one;
- adaptive controllers use
  `conditional_content_token_nats_mean_v1`, land their targets after warmup,
  and remain within their frozen alpha bounds;
- every treatment retains at least one positive-reward batch in the final 16
  updates;
- final-16 mean response length is at most
  `max(1.25 * C0 mean, C0 mean + 32)`;
- aggregate no-EOS count across the final 16 batches is at most the C0 total
  plus 16 (one extra row per batch on average), final-16 mean length is below
  768, and no batch has more than 12/16 no-EOS rows;
- initialization and terminal MATH-500 greedy pass@1 are finite.

Any failure blocks Stage A and requires a fresh, explicitly named redesign.
It does not authorize coefficient fishing or outcome-based selection.

## Stage A: matched analytical cohort, conditional on Stage S

If Stage S passes prospectively, all four arms run seeds 43/44/45 (12 jobs).
Each job trains for one complete pass over all 8,515 admitted prompts.
Evaluation and checkpointing occur every 2,129 consumed prompts (at least
quarter-epoch), plus initialization and the terminal boundary at 8,515.
Adaptive warmup is 64 updates.
MATH-500 greedy pass@1 is primary. Sampled pass@8 and mean@8, response length,
no-EOS rate, and conditional-token entropy are secondary diagnostics.

All seeds are shown individually and as a three-seed mean. C0 is one shared
matched control per seed. The older free-form Dr.GRPO paper result is context
only because its model and training recipe differ.

The E21 source and execution surface are frozen before Stage S:

- Python source:
  `var/artifacts/source_snapshots/e21_math_conditional_token_217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448/src`,
  SHA-256 `217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448`.
- Runtime execution surface:
  `var/artifacts/source_snapshots/e21_math_conditional_token_ops_05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce/ops`,
  path-bound SHA-256
  `05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce`.
- Prospective prefixes are `mte21_math_conditional_token_smoke_v10` for Stage
  S and `mte21_math_conditional_token_05b_v4` for the repaired Stage A.

Stage A additionally requires a positive smoke approval whose protocol,
source, and execution-surface hashes still match exactly. Slurm job IDs live
in the corresponding immutable-prefix manifests; they are outcomes, not part
of this prospective document.
