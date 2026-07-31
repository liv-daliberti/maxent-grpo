# Clean eight-environment 0.5B verified MaxEnt versus Dr.GRPO cohort

Status: E70 Stage A running (40 jobs); three new rows stopped; ConstructiveCode v2 admission running
Date: 2026-07-29

## Prospective staged-execution amendment

After the frozen atomic 80-job design stopped, the user explicitly authorized
launching every already-ready row without waiting for the new-domain gates.
This does not retroactively revive the stopped design or turn an ineligible
row into a result.

The separately preregistered E70 Stage-A cohort therefore launched the four
established domains as 40 fresh jobs: two arms, seeds 43–47, and 12 passes.
Its control is `grpo_compute_matched`, not literal historical GRPO, and all
jobs run on the `mltheory` A5000/A100 pool. ConstructiveCode, PantryPlan,
PointMaze, and AntMaze remain independently gated; their cells may be added
only through a prospective, identity-bound Stage B. The final eight-row
renderer remains fail-closed and cannot present Stage A alone as a completed
all-environment result.

E70 Stage A was released as Slurm jobs `30184722`–`30184761`. Its protocol is
`paper/preregistration/e70_clean_verified_maxent_vs_compute_matched_drgrpo_stage_a_05b.md`
and its identity receipt is
`var/artifacts/e70_clean_stage_a_05b_identity.json`. The fail-closed monitor is
Slurm job `30185011`; it refreshes
`var/artifacts/e70_clean_stage_a_05b_audit_latest.json` and the four scaling
curves every minute.

## Admission outcome

The clean surface is active as a staged experiment, but the original atomic
80-job cohort remains unlaunchable:

- E70 Stage A launched the four established rows as 40 fresh jobs: both arms,
  seeds 43--47, and 12 passes. Jobs `30184722`--`30184761` are running or
  queued under the immutable Stage-A identity.
- PantryPlan v2 passed its repaired three-way data audit with 384 train, 64
  development, and 128 evaluation prompts, 8--45 exact modes per prompt, and
  zero split overlap. Its frozen initial and one permitted prompt-prefill
  viability probes both produced zero verified completions, so the row stops
  before training.
- PointMaze passed executable admission, including 32 real route executions,
  3,200 perturbations, and 5.56 executions/s. Its frozen initial and one
  permitted prompt-prefill viability probes both produced zero verified
  completions, so the row stops before training.
- Ant controller v5 passed the prospective eight-heading controller gate. The
  subsequently frozen route slate failed on its first fresh map, so AntMaze is
  ineligible and no language-model sample was taken.
- ConstructiveCode v1 remains ineligible because its Python-2/Python-3 replay
  slate failed released-checker equivalence. Its prospective Python-3-only v2
  materialization is running as job `30184636` and does not yet constitute
  admission or a paper result. Dependency job `30185057` will snapshot and
  launch the 1,600-replay dual-suite gate only after materialization succeeds.

No Stage-B training cell has been launched. Failed new rows stay visibly
ineligible; they are not silently replaced, dropped, or folded into E70.

## Purpose

Produce one clean, MATH-free comparison of the best validated online MaxEnt
method against objective-equivalent plain Dr.GRPO at 0.5B scale. The reporting
surface contains every requested environment:

1. graph coloring;
2. Countdown;
3. executable Python factors;
4. executable MathIR action menus;
5. ConstructiveCode;
6. PantryPlan;
7. PointMaze; and
8. AntMaze.

This prospectively supersedes
`clean_05b_maxent_vs_drgrpo_all_domain_plan.md` before that design received an
experiment ID or launched a job. It removes MATH-500 rather than interpreting
endpoint variants as modes, and it admits the three new rows only after their
frozen feasibility gates pass.

PointMaze and AntMaze are environment rows controlled by a 0.5B language
model. They are not conventional PPO baselines. The LM emits a bounded action
program; a trusted MuJoCo worker executes it and returns environment success
plus the route identity of successful trajectories. AntMaze uses a frozen
low-level locomotion controller shared by both arms and is labeled as such.

## Exact cohort size

- 8 environments x 2 arms x 5 seeds = 80 training jobs;
- seeds 43, 44, 45, 46, and 47;
- 12 complete passes through each frozen training pool; and
- separate fail-closed evaluation, execution, integrity, and figure audits.

The 80-job count is immutable after the experiment ID is assigned. A failed
admission gate produces an explicitly ineligible figure row and prevents the
80-job cohort from launching; it does not permit a post-outcome substitution.

## Arms

### Dr.GRPO control

Use `grpo_compute_matched`. The task objective is plain Dr.GRPO. It performs
the same passive verifier-bank bookkeeping, one-group replay scoring, and
backward traversal as treatment, while
`online_canonical_replay_compute_only=1` makes the replay derivative exactly
zero. Entropy, novelty, balance, and route coefficients are zero.

### Online verified MaxEnt treatment

Use the frozen E58 `verified_first_global_replay_canonical` treatment:

- semantic Shannon coefficient 0.10;
- online canonical novelty beta 0.50;
- replay alpha 0.10;
- one persistent-hash round-robin replay group per optimizer update;
- replay capacity 16;
- open-set, verified-mass, and known-mode-balance warmup 64;
- no coefficient projection;
- no token-entropy objective;
- no E68 proposal actuator and no E69 route-successor replay; and
- no gold support, target entropy, target mode count, reference solution,
  evaluation output, or held-out route catalogue enters training.

This is the best currently validated explicit online MaxEnt recipe. E68 is
retained as mechanism evidence, not substituted for the primary treatment.

## Policy and executable interfaces

Every row uses a pinned 0.5B checkpoint, and both arms within a row use the
identical checkpoint.

- Graph, Countdown, Python, MathIR, PantryPlan, PointMaze, and AntMaze use the
  pinned Qwen2.5-0.5B-Instruct snapshot.
- ConstructiveCode uses Qwen2.5-Coder-0.5B-Instruct only if its preregistered
  viability gate passes. The different 0.5B checkpoint is labeled and kept as
  a model stratum rather than hidden in a pooled scalar.
- PointMaze maps language tokens to fixed quantized 2-D forces.
- AntMaze maps language tokens to fixed high-level heading commands executed
  by one hash-pinned frozen locomotion controller.

ConstructiveCode, PointMaze, and AntMaze execute in hash-pinned networkless
workers, never in the trainer process. Their execution counts and timeout
budgets are matched across arms.

## Fixed run contract

- Training rollouts per prompt: 16.
- Optimizer, learning rate, group size, prompt order, request seeds, token
  limits, verifier calls, and evaluation draws are matched within each row.
- Evaluation: greedy plus four deterministic temperature-one `K=8`
  replicates every quarter pass.
- Registered anchors: passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12.
- Report terminal pass 12 and trapezoidal AUC over registered anchors.
- No best checkpoint, peak selection, carry-forward, seed substitution,
  result-dependent extension, or missing-cell averaging.

The four shared outcome columns are greedy success, `mean@8`, `pass@8`, and
`distinct@8`. In maze rows, `distinct@8` counts topology-bound successful
route keys. Additional domain diagnostics do not replace the shared surface.

## Admission gates before an experiment ID

### Existing four domains

Re-audit frozen data identities, verifier source hashes, model snapshot, and
the complete E58/control compute-match contract. No historical outcome is
copied into the new cohort.

### ConstructiveCode

- select and freeze four witness families;
- pass released-checker equivalence on known correct and incorrect replays;
- pass adversarial canonicalization tests;
- pass networkless Apptainer isolation and throughput gates;
- freeze nonoverlapping train/dev/evaluation tasks; and
- pass the frozen Qwen2.5-Coder-0.5B viability criterion.

### PantryPlan

- freeze and manually audit the USDA-derived ingredient table;
- enumerate exact feasible supports for every prompt;
- require at least two and at most the registered maximum support modes;
- freeze nonoverlapping train/dev/evaluation prompt families and ingredients;
  and
- pass the frozen Qwen2.5-0.5B viability criterion.

### PointMaze and AntMaze

- pin simulator, environment, map, codebook, gate, and controller hashes;
- pass dependency-free parser and topology identity tests;
- certify at least two successful route programs per admitted prompt;
- pass perturbation, collision, near-miss, timeout, and simulator-failure
  audits;
- measure networkless worker throughput under the exact rollout boundary;
- freeze map/start/goal train/dev/evaluation splits;
- pass frozen 0.5B viability criteria on development maps only; and
- for AntMaze, independently pass frozen-controller command-following and
  stability audits.

## Frozen interpretation

The all-environment method gate requires:

- treatment minus control is positive for terminal and AUC `pass@8` and
  `distinct@8` in at least six of eight environments;
- no environment loses more than 0.03 terminal `pass@8`;
- each new environment is positive in terminal `distinct@8` in at least three
  of its four preregistered families or map strata; and
- every execution, identity, cadence, resume, separation, and information
  firewall audit passes.

Report per-environment paired seed deltas regardless of this aggregate gate.
Do not pool AntMaze with non-hierarchical rows without an explicit interface
stratum. A negative or mixed result remains part of the clean cohort.

## Figure contract

The stable requested basename is
`paper/figures/e68_e58_vs_grpo_05b_12ep_terminal_provenance.png`, retained for
continuity even though the new figure is a clean cohort rather than an E68
result. Its visible title and sidecar provenance must name the new comparison.

- Rows are the eight environments above, with no MATH-500 band.
- The first four columns are greedy, `mean@8`, `pass@8`, and `distinct@8`.
  The requested historical-format progress view appends the registered E58
  mechanism columns; those diagnostics do not replace the four shared
  outcomes.
- The empty design rendering says `AWAITING CLEAN COHORT` in every panel and
  contains no copied historical value.
- The progress rendering distinguishes pending, running, failed, ineligible,
  and terminal cells.
- Live means use only seeds present at the exact x-coordinate, visibly annotate
  their contributing `n`, and never carry a missing seed forward.
- The final renderer refuses `TERMINAL` unless all 80 expected jobs and all
  expected evaluation cells are terminal and integrity-clean.
- Show all seed trajectories and paired seed deltas in the terminal figure.
- Bind protocol, source, data, model, environment, controller, result, audit,
  and plotting-source hashes in a machine-readable sidecar.
- The live wide renderer is
  `ops/exp_scaling/plot_e70_clean_05b_wide_live.py`; its stable sidecar is
  `var/artifacts/clean_05b_eight_environment_figure_provenance.json`.

## Recorded stopping point

1. Running: E70 Stage A, 40 jobs over the four established domains, two arms,
   and five seeds.
2. Passed then stopped: PantryPlan data admission and PointMaze executable
   admission passed, but both rows failed their frozen initial and one-time
   repaired 0.5B viability gates.
3. Passed then stopped: Ant controller v5 passed, but frozen route admission
   failed on the first fresh map.
4. Running: prospective ConstructiveCode v2 materialization and executable
   replay admission.
5. Not assigned or launched: the full 80-job eight-environment cohort and all
   new-row Stage-B training cells.
6. Required reporting: E70 remains a prospectively staged four-row result; the
   final eight-row renderer stays fail-closed and shows failed rows explicitly.
