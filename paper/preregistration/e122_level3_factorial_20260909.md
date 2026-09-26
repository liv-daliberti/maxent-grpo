# E122: Level-3 Dr.GRPO / ReplayDr.GRPO / MaxRL / ReplayMaxRL factorial

Prepared September 9, 2026, before E122 treatment training or treatment outcomes.
E122 uses Qwen2.5-0.5B-Instruct for all 100 cells, frozen revision
7ae557604adf67be50417f59c2c2f167def9a775. This preserves the model in the user's
latest request to repeat E119 under the same conditions on Level3. It records
the literal-request interpretation stated before training, not a separate
explicit model-choice answer. Qwen2.5-3B remains the Level3 calibration model.
No treatment outcome informs model selection. The CLI requires an explicit
model argument, and preparation/submission/verification enforce the registered
0.5B choice; the generic 3B interface remains available for read-only previews.

## Scientific design

This is a new complete factorial: Graph Coloring, Countdown, Python Factors,
MathIR and PantryPlan × four arms × paired seeds 43–47 = 100 new runs.
Every arm starts from the same base model. No prior treatment endpoint is reused.

The arms are ordinary Dr.GRPO, ReplayDr.GRPO, binary finite-rollout MaxRL and
ReplayMaxRL. Replay uses verified_likelihood_per_rollout at alpha 0.10,
capacity 16 and one deterministic global group per update, with no bootstrap.
Both arms without a replay derivative traverse the same replay machinery in
compute-only mode. Semantic-MaxEnt, canonical-bank entropy, token entropy, xDr,
DAPO, RLEP, DIAYN and counterfactual proposal objectives are disabled.

The schedule matches the **effective** E119 runtime: 384 training prompts,
eight passes, 3,072 optimizer updates; 16 fresh samples per prompt, one PPO
epoch, constant learning rate 2e-7, zero warmup, Adam betas (0.9, 0.95), weight
decay zero, gradient norm bound 1, KL beta zero, temperature 1 and top-p 1.
Rollout batch is one prompt, train batch 16 responses; learner microbatch is
one response except Pantry's inherited four. Evaluation batch remains 64.
Model scaling does not import E80's different learning rate or cosine schedule.

Input, response and model-length limits remain 1024, 192 and 2048 tokens.
Native Qwen Level-2 r5 prompt profiles and target-blind legal syntax are the
same interfaces used by Level-3 calibration: Graph uses qwen_boxed/no grammar;
Countdown uses qwen_level2_countdown/countdown_legal_v3; Python, MathIR and
Pantry use their qwen_level2 profiles/domain_legal_v1. Ordinary generation is
explicit in all domains: canonical_action_task=none and canonical graph,
learner action and fixed-shape sampling are disabled. This incorporates E119's
documented correction of an accidentally inherited Pantry support-mask policy.

Evaluation uses greedy accuracy plus pass@8 and correct semantic modes@8 with
four fixed draws, temperature/top-p 1. Evaluation seeds, held constant across
paired training seeds, are graph 610100, Countdown 610200, Python 610300,
MathIR 610400 and Pantry 76299. Evaluations occur every 96 updates on the
pass grid 0, 0.25, 0.5, …, 8. E119's written launcher requested 192, but its
runtime capped that interval to 96; E122 preserves the realized cadence.

Optimizer/model recovery checkpoints occur every 192 updates except Pantry's
operationally amended 96. Retain one complete rolling checkpoint, preserving
its predecessor until replacement finishes. Export one terminal model and
retire resumable state normally after successful completion. Evaluation is
independent of storage; no checkpoint or outcome selection is permitted.

Primary terminal endpoints are sampled pass@8 and distinct correct modes@8.
Report all five paired seeds separately by domain. Contrasts are ReplayDr.GRPO
minus Dr.GRPO, MaxRL minus Dr.GRPO, ReplayMaxRL minus MaxRL, and the factorial
difference in differences. Secondary endpoints and full-trajectory AUC retain
E119's definitions. Do not pool domains into one confirmatory effect.

## Benchmark provenance and admission

Use only train and eval under var/data/modebench_level3_matched_v3; never load
development rows for training or treatment evaluation. Identity SHA-256 is
890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d.
The original identity remains byte-identical with its historical
pending_fresh_candidate_confirmation decision. A separate completed canonical
V3 confirmation report supplies admission; no metadata is relabeled.

Preparation and submission require the canonical auditor to authenticate the
complete all-five report, its underlying receipts, original-grader replay,
registration, dataset inventory and frozen files. Every domain must pass the
registered absolute pass@1/pass@8 tolerances of 0.04/0.08.

V3 is adaptive second-round matching against fixed, measured historical
Level-1 benchmark values, not five fresh comparisons from one round. Graph
and Python use fresh V3 candidate development fits and fresh confirmation;
Countdown, MathIR and Pantry retain V2 split bytes and completed evidence.
Historical L1 measurements remain fixed references. Passing these numerical
gates does not establish statistical equivalence or remove reference sampling
uncertainty. No E122 outcome informs admission or model selection.

## Operational resources, integrity and release

All arms share the same resource route: non-PVL A6000 nodes205/206/207 and
compatible A100 node302, account allcs/partition lowprio, one GPU, 36-hour
walltime and nice=0. The site-assigned QoS is medium and is audited; E122
does not request a QoS override. Node208 is excluded while drained for overheating.
The lower-priority partition permits owner preemption and same-job requeue;
no uninterrupted allocation or predicted start time is guaranteed. The same
checkpoint, replay-state and watchdog recovery contract remains enabled.
The release controller reserves a slot and full peak storage for every released
unfinished cell, including preempted, requeued and failed cells, until verified
successful completion or explicit reconciliation. Failed cells pause new release.

Read-only sbatch --test-only comparisons on September 9 accepted this exact
route and predicted September 10 for both model resource profiles, compared
with September 17–18 on cs. These are scheduler estimates, not submissions.
The site submission hook routes allcs jobs longer than 60 minutes to cs unless
exactly lowprio is requested. The earlier owner-borrowing amendment
(paper/preregistration/campaign_owner_borrowing_20260908.md) remains unapplied:
its multipartition request is unsupported; E122 uses the supported single
lowprio route. This prospective operational choice changes no model, objective,
batch, learning rate, evaluation cadence, checkpoint interval or endpoint.
The registered Qwen-0.5B campaign uses eight CPUs, 128 GiB for Countdown,
96 GiB for Pantry and 64 GiB otherwise. These resource choices incorporate
observed E119 host-memory throttling and Pantry failures on 24-GB GPUs; they
change no objective or batch. The generic read-only 3B preview uses 16 CPUs,
128 GiB and optimizer/activation offload, while preserving evaluation batch 64;
it does not authorize a 3B E122 submission.

A reviewed draft binds current learner/source bytes and E119's healthy frozen
operational scripts. Preparation publishes a new snapshot, changing only its
operational E122 run guards and correct job-specific watchdog log path. Existing
snapshots, calibration files and campaigns remain untouched. Two-hour inactivity,
one-hour startup grace and a twelve-restart watchdog budget are preserved.

Submission is separate from release. Each of 100 jobs is submitted held, with
an exclusive claim and per-cell intent recorded before sbatch. Every result is
preserved before checking it. An ambiguous result or interruption forbids
automatic retry; known jobs remain held for evidence-based reconciliation.
The full environment, scheduler resources and identity are audited twice before
the actual-job ledger is published. This launcher never releases a job.

The separately armed controller must authenticate admission, the registered
0.5B model, immutable plan/snapshot, all 100 held allocations and a live storage
budget. Recorded 0.5B model/optimizer files total approximately 6.44 GiB per
rolling checkpoint and 12.89 GiB during atomic replacement. Reserve 16 GiB
peak per unfinished released cell and 1.25 GiB per terminal run, or 125 GiB
for all 100 final runs. The controller cap is four released unfinished cells,
with 64 GiB additional filesystem headroom. Pending/requeued/failed released
cells retain their peak reservation and slot until verified success or explicit
reconciliation. Account for concurrent campaign writes before each release.

The size measurements are immutable recorded observations, not persistent pins
to old checkpoint files; prior campaigns may retire their own checkpoints
normally. The generic 3B preview records 84 GiB peak and 6.25 GiB terminal
reservations, but no 3B campaign is authorized here. No existing artifact
cleanup is authorized by this protocol. Normal successful-run retention
remains enabled; resource admission must never change scientific stopping
or select on efficacy.
