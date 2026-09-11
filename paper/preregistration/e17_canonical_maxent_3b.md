# E17 canonical MaxEnt 3B scale continuation

**Status: FROZEN V5 EVAL-BOUNDARY ACTOR-SYNC AMENDMENT AFTER V4 RUNTIME-SMOKE
PERFORMANCE FAILURE, BEFORE V5 SUBMISSION OR OUTCOMES (2026-07-19).**

E17 scales the completed E16 canonical-action comparison from
Qwen2.5-0.5B-Instruct to Qwen2.5-3B-Instruct. It does not revive E11's failed
free-form Standard MaxEnt design. The policy, datasets, action codecs,
objective, coefficients, seeds, and evaluation definitions are inherited from
the repaired E16 V3 campaign; model scale and two-A6000 placement are the only
substantive/runtime changes.

The immutable Python source is the E16 V3 snapshot with SHA-256
`0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec`.
This includes the Countdown canonical-code greedy pass@1 repair. The 3B model
snapshot is Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Its `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, and `merges.txt` are byte-identical to
the E16 0.5B tokenizer, preserving the audited one-token actions 1--6.

## Frozen grid

- Environments: canonical graph coloring and canonical Countdown.
- Methods: Standard MaxEnt fixed, proportional, and Haarnoja dual.
- Paired analysis seeds: 43, 44, and 45 (18 jobs total).
- Group size: 16; one PPO epoch; learning rate `2e-7`; `beta=0`.
- Horizon: five complete prompt-pool epochs.
- Graph pool: 192 train / 96 evaluation prompts; 960 optimizer updates.
- Countdown pool: 384 train / 128 evaluation prompts; 1,920 optimizer updates.
- Evaluation cadence: every 48 graph prompts and every 96 Countdown prompts,
  exactly one quarter of an epoch, plus the initialization evaluation.
- Sampled evaluation: pass@8, mean@8, and coverage@8 at temperature 1;
  greedy pass@1 uses the repaired canonical Countdown decoder.
- Checkpoints: retain the newest quarter-epoch checkpoint only.
- Placement: one 48 GB A6000 per job across idle nodes 103/104/208,
  canonical learner-side rollout batch 1, ZeRO stage 2, and optimizer plus
  activation offload. E17 retains batch 1 rather than changing the audited
  canonical sampling semantics to satisfy a multi-GPU topology.

The canonical policy emits exactly three digit actions. Graph support is 27
actions with entropy target `2.7974740052946436`; Countdown support is 108
actions with entropy target `3.9741470618167156`.

| Method | Coefficient/controller |
|---|---|
| Standard MaxEnt fixed | `alpha=0.10` |
| Standard MaxEnt proportional | base `0.075`, max `0.10`, EMA `0.9`, gain `4`, immediate exact-entropy control |
| Standard MaxEnt Haarnoja dual | base `0.075`, min `0.05`, max `0.10`, log-alpha Adam LR `0.005`, immediate exact-entropy control |

The historical free-form Dr.GRPO/xDr curves are contextual only and are not a
canonical-policy control. All three seeds are reported individually and
aggregated; no single-seed curve may be presented as the E17 result.

## V1 scheduler-only attempt

Jobs 30015667--30015684 were submitted and released, but the cluster's account
policy rewrote the requested `all` partition to `mltheory`. Because the frozen
node constraint named A6000 nodes outside that partition, Slurm held every job
at `BadConstraints`. All eighteen were cancelled with elapsed time `00:00:00`,
no node assigned, and no training process or outcome. V2 changes only the
scheduler account from `mltheory` to the user's valid `allcs` association so
the already-frozen `all` partition and A6000 node constraint are honored. V2
uses fresh `gce17_canonical_maxent_3b_v2` and
`cde17_canonical_maxent_3b_v2` prefixes.

V2 jobs 30015688--30015705 were also cancelled at elapsed time `00:00:00`
with no node assigned. The site job-submit policy preserves an explicit
`lowprio` request but rewrites a long `all` request to the account-specific
partition (`cs` for `allcs`), which excluded the requested idle nodes. V3
therefore requests the documented shared `lowprio` partition directly and
uses the `mltheory` association, which is authorized there. It adds node105 to
the node constraint and requests generic `gpu:2`, allowing the frozen
runtime's already-implemented A5000-only offload rule. V3 uses fresh
`gce17_canonical_maxent_3b_v3` and `cde17_canonical_maxent_3b_v3` prefixes.

V3 jobs 30015706--30015723 allocated successfully across node105 and A6000
nodes, then all exposed the same initialization assertion before any rollout
or optimizer step: OAT's two-GPU actor topology requires rollout batch size to
be divisible by two, while the canonical learner sampler requires rollout
batch size exactly one. The jobs were cancelled after approximately 85
seconds; no training metric was written and no V3 outcome is eligible. V4
uses one A6000 per actor/job with offload, the already-audited batch-one
topology, and fresh `gce17_canonical_maxent_3b_v4` and
`cde17_canonical_maxent_3b_v4` prefixes. A separate seed-9006 graph fixed-arm
runtime smoke must reach a real optimizer record before the 18-job cohort is
submitted.

The V4 runtime smoke, job 30015724, successfully completed canonical sampling
and its first offloaded optimizer update. It then spent more than five minutes
broadcasting 3B weights to the vLLM actor at step 1, and was cancelled before
writing the post-update metric. This broadcast is unnecessary between
evaluations: canonical training rollouts are sampled from the learner itself,
not vLLM. V5 exposes OAT's existing `sync_params_every` option and sets it to
48 graph updates / 96 Countdown updates, exactly matching the frozen
quarter-epoch eval cadence. Thus every inline evaluation still uses current
weights, while no stale actor participates in training. The six-file frozen
V5 execution surface has combined receipt
`a8414bcd1932787ee1cea7e8c0b23868282f45903eeb9d495b23e4649f397f0c`.
V5 uses fresh `gce17_canonical_maxent_3b_v5` and
`cde17_canonical_maxent_3b_v5` prefixes and a fresh runtime smoke.

## Retrospective E16 antecedent note — appended after E16 completion

**This note records the inherited 0.5B outcome after E17 was frozen. It does
not alter E17's grid, coefficients, stopping rule, or eligibility criteria.**

All eighteen E16 Stage-R V3 jobs completed without watchdog alerts. At the
registered terminal sampled evaluation, all three methods improved every
reported metric over their common initialization in both domains in the
three-seed mean. This aggregate statement is not seed-uniform. The most
encouraging adaptive result is on Countdown: proportional control improves
`pass@8`, `mean@8`, coverage, and greedy `pass@1` over paired fixed MaxEnt by
4.2, 9.1, 3.3, and 5.2 points, respectively. Graph coloring is strong under
all three methods, with mean coverage 0.360/0.338/0.352 for
fixed/proportional/dual. These outcomes motivate, but do not modify, the
already-frozen 3B scale continuation. E16's exact held-out full-support
quantities remain pending; see
[`../results/e16_canonical_maxent_replication.md`](../results/e16_canonical_maxent_replication.md).

## Retrospective V5 execution note — appended after launch

**This is an operational snapshot, not an analytical result and not a change
to the frozen protocol.**

As of July 19, 2026 at 15:44 EDT, fifteen V5 jobs are running and the three
Countdown seed-45 jobs are pending for resources or priority. No analytical
job is complete. At the latest checkpoint shared by all three graph-coloring
seeds and all methods (two passes), the sampled trajectories are nearly tied.
Countdown has reached a one-pass common frontier only for seeds 43 and 44; the
third seed is absent rather than imputed. These live trajectories may be shown
only as explicitly interim evidence and must remain outside endpoint,
abstract, and conclusion claims until the complete registered cohort lands.
