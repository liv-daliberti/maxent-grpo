# Clean 0.5B verified MaxEnt versus Dr.GRPO cohort

Status: superseded before experiment ID; no jobs launched
Date: 2026-07-29

This 36-job design is retained as a provenance record. The user subsequently
requested a MATH-free figure containing PantryPlan, PointMaze, and AntMaze as
0.5B language-policy environments. The prospective 48-job replacement is
`clean_05b_maxent_vs_drgrpo_eight_environment_plan.md`. No outcome from this
unlaunched design was inspected before the replacement, and its job count must
not be presented as the current plan.

## Purpose

Produce the clean paper comparison requested after the current successor gate:
explicit online verified MaxEnt versus objective-equivalent plain Dr.GRPO at
0.5B scale, over every admitted ModeBench domain, with one additional
ConstructiveCode row and a separately labeled held-out MATH-500 transfer row.

This is a new cohort. It does not retrofit E69, reuse interrupted E61-R1
weights, or combine post-outcome replacements. E61-R1 supplies the treatment
definition and historical evidence; E68 remains the separated-support causal
ablation; E69 remains the verifier-bound route-transfer successor.

`PantryPlan` is now a separate candidate under
`pantry_plan_modebench_extension_plan.md`. It is not included in this frozen
36-job design. If it passes its source, support-identity, and 0.5B viability
gates before this cohort is assigned an experiment ID, a prospectively frozen
superseding design may contain six ModeBench rows and 42 jobs. Do not mutate
this job count or add PantryPlan after seeing a pilot outcome.

## Reporting surface

The five ModeBench training domains are:

1. graph coloring;
2. Countdown;
3. executable Python factors;
4. executable MathIR action menus; and
5. ConstructiveCode, only after its executable-identity and 0.5B viability
   gates pass.

MATH12K to MATH-500 is a sixth training/evaluation area but not a sixth
ModeBench domain. Its figure row must remain visually separated and must not
report endpoint variants as reasoning modes.

The complete cohort is therefore:

- 5 ModeBench domains x 2 arms x 3 seeds = 30 training runs; and
- 1 MATH transfer area x 2 arms x 3 seeds = 6 training runs;
- total: 36 exact-manifest jobs, plus fail-closed evaluation/audit jobs.

## Arms

### Dr.GRPO control

Use `grpo_compute_matched`, described in the paper as compute-matched Dr.GRPO.
Its task objective is plain Dr.GRPO. It passively tracks the same verified bank
and executes the same one-group replay scoring and backward traversal as the
treatment, but `online_canonical_replay_compute_only=1` makes the replay score
derivative identically zero. All entropy, novelty, balance, and route
coefficients are zero.

This control is preferable to the older literal `grpo` launcher because it
holds auxiliary mechanism compute constant without changing the Dr.GRPO
optimizer update.

### Verified MaxEnt treatment

Use the frozen E58 `verified_first_global_replay_canonical` treatment:

- semantic Shannon coefficient 0.10;
- online canonical novelty beta 0.50;
- replay alpha 0.10;
- one persistent-hash round-robin replay group per optimizer update;
- replay capacity 16;
- open-set, verified-mass, and known-mode-balance warmup 64;
- no coefficient projection;
- no direct token-entropy objective;
- no gold support, target entropy, target mode count, evaluation feedback, or
  reference solution enters training.

ConstructiveCode must use the validator-emitted witness tuple as its canonical
key. Program source, AST, formatting, and compiler trace are ineligible.

No E68 counterfactual proposal actuator and no E69 route-successor replay is
enabled in either primary arm.

## Model strata

Every run uses a 0.5B checkpoint, pinned by repository revision and local tree
hash.

- Graph, Countdown, Python, MathIR, and MATH use the existing
  Qwen2.5-0.5B-Instruct snapshot.
- ConstructiveCode uses Qwen2.5-Coder-0.5B-Instruct only if the preregistered
  0.5B viability gate passes.

The coder row is a separate model stratum. It is never pooled into a scalar
effect with the general-model rows, and the checkpoint difference is named in
the figure and table. If the 0.5B coder fails viability, ConstructiveCode does
not enter this requested 0.5B cohort. A 1.5B feasibility retry may be reported
separately but cannot replace it post hoc.

Within every row, control and treatment use the identical checkpoint.

## Fixed run contract

- Seeds: 43, 44, and 45.
- Training rollouts per prompt: 16.
- Budget: 12 complete passes through each frozen training pool.
- Optimizer, learning rate, group size, prompt order, request seeds, token
  limits, verifier calls, and evaluation draws are matched within each row.
- Evaluation: greedy plus four deterministic replicates of temperature-one
  `K=8` sampling every quarter pass.
- Registered reporting anchors:
  passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12 for ModeBench;
  passes 0, 2, 4, 6, 8, 10, and 12 for MATH-500.
- No best-checkpoint, peak, carry-forward, seed substitution, or
  result-dependent extension.

For ConstructiveCode, execution counts and candidate token budgets must match
across arms in addition to optimizer steps. Candidate programs execute in the
hash-pinned networkless kernel sandbox, never in the trainer process. This is
the pre-sampling 2026-07-29 amendment registered in
`modebench_constructive_code_extension_plan.md`: immutable SquashFS identity,
fresh worker-local extraction, Landlock ABI 5+, explicit seccomp escape-surface
denials, clean environment, and parent-enforced resource limits. The original
Apptainer path failed at infrastructure setup before candidate or model
execution and is not used as evidence.

## Preconditions

Do not assign an experiment ID or submit jobs until all of the following are
frozen and clean:

1. E69 Gate 2 reaches a terminal audited decision and its fail-closed handoff
   behaves as registered. E69's outcome does not change this cohort's arms.
2. The pinned count-only ConstructiveCode source audit is complete.
3. Four constructive witness schemas and canonicalizers pass equivalence and
   adversarial alias tests.
4. Released checker replay has zero wrapper false accepts and preserves the
   registered per-task quality threshold.
5. The amended kernel-isolation smoke passes on a compute node with zero
   violations, median launch latency at most 0.25 seconds, and p95 at most
   0.50 seconds; full released-checker throughput also fits the online-RL
   budget.
6. Train/development/evaluation problem IDs and artifacts are hash-frozen with
   zero overlap.
7. Qwen2.5-Coder-0.5B-Instruct passes the frozen viability gate without using
   evaluation prompts.
8. Configuration-only expansion produces exactly 36 unique arm/seed/area jobs
   under one source and execution snapshot.

## Frozen interpretation

Primary ModeBench summaries are terminal pass 12 and trapezoidal AUC over the
registered checkpoints for `pass@8` and `distinct@8`.

The five-domain method gate requires:

- treatment minus control is positive for terminal and AUC `pass@8` and
  `distinct@8` in at least four of five ModeBench domains;
- no ModeBench domain loses more than 0.03 terminal `pass@8`;
- ConstructiveCode is positive in `distinct@8` in at least three of its four
  frozen witness families, with no family losing more than 0.03 terminal
  `pass@8`; and
- every execution, identity, cadence, resume, separation, and information
  firewall audit passes.

The held-out transfer gate remains separate: terminal MATH-500 greedy and
`mean@8` must each be no more than 0.02 below control and at least one must be
directionally positive. `distinct@8` is structurally ineligible on this
endpoint-only track.

These thresholds must be copied verbatim into the eventual preregistration
before the new experiment ID is assigned.

## Figure contract

Create a new provenance-bound figure rather than overwriting the historical
E61/E68/E69 panels.

- Rows: Graph, Countdown, Python, MathIR, ConstructiveCode, then a separated
  held-out MATH-500 band.
- Core columns: greedy, `mean@8`, `pass@8`, `distinct@8`, and excess
  multiplicity; MATH-500 omits ineligible mode metrics.
- Show all seed trajectories and paired seed deltas, not only the mean.
- Label the ConstructiveCode coder checkpoint directly on its row.
- Embed the audit, identity, protocol, source, data, and result hashes in the
  figure-side provenance record.
- Refuse to render a final figure if any expected cell is missing, nonterminal,
  or integrity-failed.

The eventual stable basename should describe the scientific comparison, not a
historical experiment number, for example
`verified_maxent_vs_drgrpo_05b_12pass_all_domains`.

## Immediate next work

1. Verify the live E69 Gate-2 decision and Gate-3 handoff.
2. Implement ConstructiveCode witness schemas and checker equivalence tests
   over the sealed 395-row feasibility pool.
3. Measure isolated execution throughput and freeze the split.
4. Run the 0.5B coder viability gate.
5. Only then assign the new experiment ID, preregister, config-expand, and
   submit the 36-run cohort.
