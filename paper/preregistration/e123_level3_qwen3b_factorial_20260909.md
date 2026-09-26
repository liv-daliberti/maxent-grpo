# E123: Qwen2.5-3B Level-3 Dr.GRPO / ReplayDr.GRPO / MaxRL / ReplayMaxRL factorial

Prepared September 9, 2026, before E123 treatment training or treatment outcomes.
The user explicitly requested testing the node302 efficiency improvements and
then launching this complete new Level-3 factorial at Qwen 3B scale as E123.
This document registers the scientific comparison and separately defines the
systems acceptance needed to select its physical execution settings. A draft
or queued benchmark does not constitute a passed systems gate or a launched
training campaign. The final plan and measured systems report record those
states without retrospectively changing this protocol.

## Scientific design and optimizer lineage

Use Qwen/Qwen2.5-3B-Instruct, frozen model revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, for every cell. The full campaign is
Graph Coloring, Countdown, Python Factors, MathIR and PantryPlan × four arms
× paired training seeds 70, 71, 72, 73 and 74 = 100 fresh runs. Each run starts
from the same pretrained model with fresh optimizer and replay state. Existing
E80-R1 or E118-Q3 endpoints are descriptive comparators, not E123 arm substitutes.

The common optimizer recipe follows the established Qwen3B E80-R1/E118-Q3
lineage: AdamW, peak learning rate 1e-7, `cosine_with_min_lr`, ten-percent linear
warmup, minimum learning rate 1e-8, Adam betas (0.9, 0.999), epsilon 1e-8, weight
decay zero, PPO clip range 0.2, gradient norm bound 1, KL beta zero, and BF16.
The 3,072-update horizon and its warmup/decay must be verified in the actual
runtime; `OAT_ZERO_MAX_STEP_ADJUSTMENT=16.0` corrects the inherited native
192-step accounting while preserving one scheduler step per prompt update.

This choice preserves the current 3B optimization contract across difficulty
levels. E122 uses its separately registered 0.5B constant-2e-7 recipe. A comparison
between E122 and E123 therefore changes the registered optimizer recipe as well
as model scale and must not be described as a strictly model-only intervention.
No E123 outcome selects a learning rate or schedule.

Every domain uses 384 training prompts and eight passes, giving 3,072 updates
per run and 307,200 planned campaign updates. Each update has one prompt, 16
fresh responses, train batch size 16, and one PPO epoch. Rollout temperature and
top-p are 1. Physical microbatch size may be 1, 4 or 8 only under the systems
selection below; gradient accumulation remains 16 divided by the microbatch.
Changing physical batch shape must preserve the effective objective, replay
coefficient, update count and sampling/request-stream contract.

| Arm | Task objective | Applied verified-replay derivative |
|---|---|---|
| Dr.GRPO (`drgrpo`) | Ordinary Dr.GRPO | Zero; execute compute-only replay traversal |
| ReplayDr.GRPO (`replay_drgrpo`) | Ordinary Dr.GRPO | Weight 0.10 |
| MaxRL (`maxrl`) | Binary finite-rollout MaxRL | Zero; execute compute-only replay traversal |
| ReplayMaxRL (`replay_maxrl`) | Binary finite-rollout MaxRL | Weight 0.10 |

Replay is `verified_likelihood_per_rollout`, capacity 16 observed correct modes
per prompt, one deterministic global replay group per update, zero bootstrap,
and no adaptive weighting. Bank insertion, scheduling and traversal remain
shared across arms. Semantic-MaxEnt, semantic Shannon shaping, canonical-bank
entropy, token entropy, xDr, DAPO, RLEP, DIAYN and counterfactual/proposal
objectives are disabled. Exhaustive support, evaluation outcomes and desired
mode counts cannot influence training or replay scheduling.

## Frozen Level-3 data and admission

Train and treatment evaluation use only the corresponding `train` and `eval`
splits under `var/data/modebench_level3_matched_v3`; each domain has 384 train,
128 development and 128 evaluation rows. The development split is excluded from
E123 treatment training and treatment evaluation. Data spelling maps
`pantry_plan` to the `pantry` directory. No recipes, rows or confirmation receipts
are altered for E123.

| Frozen source | SHA-256 |
|---|---|
| `var/data/modebench_level3_matched_v3/identity.json` | `890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d` |
| `var/data/modebench_level3_matched_v3/frozen_recipes.json` | `acad4298da776476717753fab0ab8e97f106c8fcafeea08e5ad8af028fe7247c` |
| `var/artifacts/modebench_level3_v3/registration.json` | `12a426f944acfa5a27cca8eafe2736f51c0589dfd6f6e00460d3142ead1d728f` |
| `var/artifacts/modebench_level3_v3/fixed_reference_amendment.json` | `2537d45996e304c6f9880cff030332267ce5981c6674ce17f0129ffb1474fdab` |
| `var/artifacts/modebench_level3_v3/confirmation/confirmation_report.json` | `b685cfa6b20d6f9b714239546a7946947af95af3d4e83ed8ab1b7ad96c453195` |

The canonical report already records `matched_fixed_reference`, all five
comparisons complete, no errors or missing domains, and passed absolute
pass@1/pass@8 differences within 0.04/0.08. Its underlying original-grader replay,
receipts, registration, dataset files and inventories must be reauthenticated
by `audit_modebench_level3_v3.validate_confirmation_report` at preparation and
submission, with frozen inputs rehashed before release. Merely reading a saved
PASS string is insufficient. The identity retains its original historical
`pending_fresh_candidate_confirmation` metadata; the separate completed
canonical report provides admission. No data-gate waiver is needed or implied.

This is adaptive second-round matching against fixed, measured historical
Level-1 values. Graph and Python have fresh V3 candidate development fits and
fresh confirmation; Countdown, MathIR and Pantry retain V2 split bytes and
completed evidence. Passing the numerical tolerances does not establish
statistical equivalence, eliminate fixed-reference sampling uncertainty, or
make all five comparisons fresh observations from one round. No E123 treatment
outcome informs data admission.

## Interfaces, evaluation and estimands

Preserve Level-3's native Qwen Level-2 r5 prompt interfaces and target-blind legal
syntax: Graph `qwen_boxed`/`none`; Countdown
`qwen_level2_countdown`/`countdown_legal_v3`; Python
`qwen_level2_python_factors`, MathIR `qwen_level2_mathir` and Pantry
`qwen_level2_pantry`, each with `domain_legal_v1`. Ordinary generation is explicit:
`canonical_action_task=none`, canonical graph actions disabled, canonical learner
sampling disabled, and fixed-shape sampling disabled. This incorporates the
E119/E122 correction of an accidentally inherited Pantry support-mask policy.
Raw prompt lengths and tokenization must satisfy the runtime input contract.

Input, response and model limits are 1024, 192 and 2048 tokens. This is the
Level-3 envelope, including Pantry's ordinary 192-token response budget; short
Level-1 Graph/Pantry traces alone cannot establish its memory safety.
Evaluation uses all 128 evaluation prompts, greedy accuracy, and four fixed
sampled draws at k=8 with temperature/top-p 1. Inherited evaluation seeds stay
fixed across paired training seeds: Graph 610100, Countdown 610200, Python
610300, MathIR 610400, Pantry 76299. Physical evaluation batch size is 32,
following the established 3B memory recipe; all logical evaluation work is
preserved.

Evaluate every 96 updates on the full pass grid 0, 0.25, 0.5, ..., 8, matching
E122's effective evaluation cadence. Primary terminal endpoints are sampled
pass@8 and distinct correct semantic modes@8. Primary contrasts within each
domain are ReplayDr.GRPO minus Dr.GRPO, MaxRL minus Dr.GRPO, ReplayMaxRL minus
MaxRL, and `(ReplayMaxRL - MaxRL) - (ReplayDr.GRPO - Dr.GRPO)`. Show every paired
seed and paired-seed uncertainty. Secondary endpoints include greedy accuracy,
mean sampled correctness and excess multiplicity (`distinct@8 - pass@8`);
trajectory summaries use trapezoidal AUC over the complete registered grid.
Report domains separately and retain failed/incomplete cells visibly. No best
checkpoint, early scientific stopping or pooled confirmatory effect across
domains is selected.

## Prospective A100 systems selection

The user authorizes running the previously prepared node302 benchmark and
launching E123 once its selected execution profile is supported by measurement.
The earlier audit's `prepared_only_not_submitted` status describes that earlier
turn; it is not a continuing permission requirement. This authorization permits
isolated benchmarks and the new campaign. Performance results and scientific
outcomes must remain distinct.

Use one allocated A100-80GB per independent learner, ZeRO-2, BF16, gradient
checkpointing, existing eager attention, vLLM ratio 0.25 and sleep level 1. Do
not infer eight-job feasibility from nominal GPU capacity or Slurm requests.
The measured current 3B CPU-optimizer recipe used about 84 GiB noncache host RAM
per job and is unsuitable for an unmeasured 56 GiB request. The activation-offload
flag exists but the inspected GRPO path does not use it; do not attribute a
measured performance improvement to activation offload without execution evidence.

Compare settings separately before combining them:

| Candidate | Explicit change from matched baseline | Invariant |
|---|---|---|
| Physical microbatch 4 | `OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4` | Train batch 16, accumulation 4 |
| Physical microbatch 8 | `OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=8` | Train batch 16, accumulation 2 |
| CPU Adam with four threads | `OMP_NUM_THREADS=4` at baseline microbatch | Same CPU optimizer and logical work |
| GPU-resident Adam | `OAT_ZERO_ADAM_OFFLOAD=0` at baseline microbatch | Same AdamW hyperparameters and logical work |

The baseline uses microbatch 1 except Pantry's established 4 and
`OMP_NUM_THREADS=1`. GPU Adam changes the installed implementation from
DeepSpeedCPUAdam to OAT's FusedAdam interface, which the current production
`fused_adam_shim` binds to `torch.optim.AdamW(fused=True)` to avoid a CUDA JIT
build. Pin and report that actual class and shim setting in both the benchmark
and production runtime; numerical and recovery validation remain required.
For E123, which starts from the base model, successful fresh GPU initialization
and GPU save/reload continuity are required. CPU-to-GPU checkpoint restoration
is additionally required before applying this change to any existing CPU-Adam
campaign continuation; resetting existing optimizer state is not permitted.

Benchmark outputs use isolated directories and cannot enter E123 endpoints.
Choose examples by predetermined prompt order, input/response length and replay
coverage, not accuracy or mode-count gains. A populated replay bank and identical
materialized fresh/replay work must exercise both replay-live and compute-only
paths. Compare a repeated baseline with each candidate for finite loss, gradient
and parameter-update differences, optimizer steps/moments, scheduler horizon,
replay coefficients/counts, bank state and request-stream continuity. Report
floating-point deviations and fail on an objective, scaling or update-count
mismatch. Observe the real learner; a synthetic matrix multiplication cannot
establish the training contract.

Use eight warmup and 32 timed predetermined updates for each matched learner
throughput comparison. All five domains share the 1024-input/192-response tensor
capacity envelope, so one designated domain may supply the full timing window,
with one additional full-capacity stress update for each of the other four.
These synthetic tensor stress inputs exercise the production loss and replay
path; they are not validator-positive discoveries or scientific observations.
Separately exercise actual actor/learner execution for all five frozen Level-3
interfaces, full evaluation and a complete temporary checkpoint save/reload.
Measure the first optimizer update because its moments allocate lazily. Restore
validation must demonstrate that model/master parameters, optimizer state,
scheduler and step counters are recovered after an intervening state change;
reloading unchanged state alone cannot establish recovery correctness. Actual
runtime recovery also preserves replay bank, data cursor and request-stream state.
Existing Level-1 timing windows may motivate candidates but cannot substitute
for these Level-3 capacity and execution checks.

Record completed-update wall times, token/replay work, learner/generation/sync
phases, GPU peak memory, whole-job cgroup anonymous/shared/cache/kernel memory,
`memory.events`, pressure and actual CPU thread use. Base a reduced host-memory
request on the peak anonymous/shared/kernel footprint plus dirty/writeback
pages, with at least twenty-percent or 8 GiB headroom, whichever is larger,
and retain full-checkpoint-save/load evidence. Clean file cache from earlier
CPU-optimizer benchmark workers can be reclaimed; cumulative clean page-cache
occupancy is recorded but does not by itself define the per-job reservation.
Reduced-request concurrency still requires aggregate pressure observation.
Select by completed useful
updates per node-hour subject to numerical/recovery correctness and safe host/GPU
peaks, independent of efficacy outcomes. Fix the selected microbatch, optimizer
residency, threads, CPU/RAM request and their evidence hashes prospectively in
one production profile shared across arms; if a domain needs a different
physical profile, fix that map across all four arms before release.

Eight simultaneous independent jobs is the target, subject to measurements and
available resources. A candidate eight-CPU/56-GiB allocation would reserve
64 CPUs and 448 GiB for eight jobs; it is not an observed safe profile. Admit
concurrency using measured peaks plus explicit headroom and check aggregate
pressure as concurrency rises. Use a lower supported concurrency if GPU, host
RAM, CPU, storage or the scheduler limits it. Failed performance candidates do
not justify changing scientific batches, evaluation coverage or stopping rules.

## Launch, storage and recovery contract

Publish a new hash-bound E123 runtime and separate 100-cell plan/ledger; do not
modify frozen E122 launchers, plans, source snapshots, evidence or its existing
release controller. The new scheduler auditor must accept Slurm 25.11's exact
single-node spellings `NumNodes=1` and `NumNodes=1-1`, preserve raw scheduler
records, and reject broader ranges. E122's actual additive compatibility
entrypoint was `e122_slurm_2511_compat.py`; its later unused alternate adapter is
not the provenance of the actual E122 launch.

The immutable E123 plan uses the node302 owner partition and account
`mltheory`, one A100 per job, 72-hour allocations, and Nice 100. Preserve
QoS, exclusion and allocation evidence from actual scheduler readback.
The measured resource profile sets CPU and host-memory requests. Long
allocations reduce expensive optimizer reloads. Submitting a held cohort
and releasing runnable jobs are separate audited operations.

For each new cell record a durable exclusive intent before `sbatch --hold`, keep
the raw result before parsing it, and audit the complete environment/resources.
Ambiguous submission/release results stop automatic retry until exact IDs and
state are reconciled. All 100 unique allocations must pass complete identity
and held-job checks before the actual ledger is published. Release only exact
E123 ledger IDs through one persistent controller with durable release journals,
source/plan/evidence authentication and a live storage budget. Failed, requeued,
preempted or unresolved released cells retain their concurrency/storage
reservation until verified success or explicit reconciliation. Report queued,
held, running and completed cells separately in campaign tracking.

Recovery checkpoints occur every 192 updates except Pantry every 96. Retain one
complete rolling checkpoint and its predecessor until replacement completes;
export one terminal model and retire resumable state after verified normal
success. Preserve model/master parameters, optimizer moments, scheduler, data
cursor, replay bank and RNG/request-stream state across recovery. Preserve the
two-hour inactivity limit, one-hour startup grace, twelve-restart watchdog budget
and job-specific watchdog log path. Storage policy must not select checkpoints
or alter the scientific horizon.

Recorded Qwen3B files total approximately 40.2 GiB per resumable checkpoint
(6,172,096,696 model bytes plus 37,031,303,168 optimizer bytes). The existing
conservative reservation is 84 GiB per released unfinished cell for atomic
replacement, including its terminal export, with 64 GiB additional
shared-filesystem headroom. One hundred retained terminal exports would
eventually occupy approximately 625 GiB; eight active replacement reservations
total 672 GiB. The controller checks live free space before each release,
reserving active replacement peaks and concurrent campaign writes. It does
not preallocate all 100 future terminal exports. Completed exports remain
accounted for through measured free space. Use these peak reservations until
complete measured saves justify a separately recorded profile. GPU optimizer residency does not
itself remove durable optimizer checkpoint bytes. No existing artifact cleanup
is required or authorized by this protocol.

At publication, record exact plan, runtime, dataset, benchmark evidence,
launcher/test/controller hashes, real scheduler IDs, the selected resource
profile, and the initial release/heartbeat verification. Until those receipts
exist, describe the corresponding step as prepared, queued or awaiting its
specific gate rather than completed.
