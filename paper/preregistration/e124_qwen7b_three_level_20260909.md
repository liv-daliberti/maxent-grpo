# E124: one-seed Qwen7B MaxRL / ReplayMaxRL across three levels

Registered September 9, 2026, before E124 treatment outcomes. The user explicitly
requested one Qwen7B seed across all three levels and five domains, and selected
MaxRL and ReplayMaxRL: 3 levels × 5 domains × 2 arms × 1 seed = 30 fresh runs.

Use Qwen/Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` and training seed **70** throughout.
Each cell starts from pretrained weights with fresh optimizer and replay state.
Existing smaller-model endpoints are not E124 initializations or observations.
One seed provides a descriptive scale extension, not a seed-variability estimate.

## Scientific contract

The domains are Graph Coloring, Countdown, Python Factors, MathIR and PantryPlan.
Level1 uses the frozen E118/E80-R1 interfaces and data; Level2 uses the admitted
`modebench_harder_v2_matched_r5` data; Level3 uses the authenticated completed
`modebench_level3_matched_v3` fixed-reference confirmation. Preserve all split
bytes, native prompts, legal syntax and graders. No refitting or admission
decision uses E124 outcomes. Level3's adaptive fixed-reference provenance remains
as described in its canonical registration; this is not a new equivalence claim.

All cells use 384 training prompts, eight passes and 3,072 prompt-level optimizer
updates; one PPO epoch; group and effective batch size16; learning rate1e-7;
cosine-with-minimum learning-rate scheduling;10% warmup and10% minimum LR;
scheduler-horizon adjustment16; Adam betas(.9,.999), epsilon1e-8, weight decay0;
gradient norm1; KL coefficient0; generation temperature1 and top-p1. This common
scale-aware optimizer lineage is chosen prospectively for all three levels,
rather than inheriting the different0.5B Level2 defaults.

Both arms use the binary finite-rollout MaxRL task objective and the same
verified replay-bank traversal. ReplayMaxRL applies the
`verified_likelihood_per_rollout` derivative with weight0.10; MaxRL executes the
matched traversal with its applied replay derivative zero. Bank capacity is16
observed correct modes per prompt, one deterministic global replay group per
update and zero bootstrap. Entropy, counterfactual, DAPO, RLEP, DIAYN and other
auxiliary objectives are disabled. Evaluation information never enters training.

Level1 Pantry retains its legitimate six-action support-mask interface.
Level2/3 Pantry use ordinary generation and the native Level2 Pantry prompt;
canonical action, learner and fixed-shape sampling are explicitly disabled.
The resolved30-cell manifest seals every domain-specific interface and length.

Evaluate all128 held-out prompts initially and every192 updates, with the final
endpoint required. Retain greedy evaluation plus four fixed sampled draws,
K=8, temperature1, top-p1 and inherited fixed evaluation seeds. Evaluation
microbatch32 is explicit. `ALLOW_SPARSE_EVAL=1` prevents the generic wrapper from
silently replacing the registered192-update cadence with96. This common E124
cadence is coarser than E123's96-update cadence; all endpoints remain aligned.
Save full resumable model, optimizer, RNG, prompt-traversal and replay state every
96 updates, retaining one rolling checkpoint and one terminal export. Preserve
an old checkpoint until the new one commits; prune resumable state only after
successful terminal export under the existing runtime policy.

## Runtime qualification and release

The initial candidate is one48GiB A6000,256GiB host RAM and8CPUs; ZeRO2,
CPUAdam offload, activation offload, physical microbatch1, four optimizer/BLAS
threads, one actor and rollout batch1; vLLM memory ratio0.40 and default level1
sleep. The single-GPU topology preserves the established update path. Historical
7B failures and initialization-only successes do not qualify this configuration.
CUDA, Python and cache roots must resolve to the actual workspace, independently
of the immutable source-snapshot location.

An isolated systems suite must exercise the production MaxRL and ReplayMaxRL
gradient/update at full tensor shape; actual actor synchronization, generation
and complete evaluation on all15 level/domain interfaces; its own complete
checkpoint save, restore and subsequent update; and whole-job host/GPU memory.
The detailed suite manifest records representative arm coverage and exact short
smoke boundaries. Short smoke runs retain the3,072-update scheduler horizon and
are excluded from scientific counts and outcomes. Synthetic tensor stress is
never represented as discovered correct responses. Score improvements cannot
select hardware or admit the campaign. Required observed headroom is20% or8GiB
host RAM, whichever is larger, and at least2GiB on the48GiB GPU.

Submit science jobs held with exact resource/export audits and unique ownership
comments. Their release requires the fully qualified, hash-bound systems result,
unchanged model/data/runtime/recipe identities and fresh storage admission.
Initially permit at most one E124 GPU allocation, including its systems suite.
No existing E118–E123 allocation is cancelled or altered by this controller.

Reserve220GiB for one E124 rolling-checkpoint peak plus64GiB shared headroom
and conservative outstanding reservations for other owned checkpoint writers.
The estimate uses14.19GiB pretrained weights and the observed smaller-model
model-plus-optimizer ratio, pending the measured7B checkpoint. Release later
cells only when live free space permits; existing campaigns' normal successful
checkpoint pruning may provide that space. Do not delete existing results.
Thirty terminal exports alone require approximately426GiB; no claim is made
that every cell can finish within today's free-space envelope.

The CPU controller records every submit/release/requeue intent before mutation,
never repeats an ambiguous submission, audits the exact owned held job before
release, and treats unexpected writers, hashes, failures or counters as requiring
review. Normal completed boundaries may advance the staged queue. Timeout or
preemption recovery must retain the same cell, source, model, data, resources
and scientific horizon and require a validated advancing checkpoint. Controller
operation is bounded by its recorded deadline and retry limits. A queued systems
suite or held30-cell ledger is not reported as qualified or running training.
