# E98-R1 preregistration: sparse prompt-matched RLEP-Dr feasibility repair

Date frozen: 2026-08-13, after the original E98 collection audits failed and
before any E98-R1 learner smoke or scientific training cell.

## Why this is a new cohort

E98 collected the registered 64 candidates for every one of 384 training
prompts, but the terminal seed policies were effectively all-or-none by
prompt. In the completed seed-43 audits, 253 Graph, 163 PantryPlan, and 303
Python Factors prompts had no verified trajectory; the other completed
collections show the same qualitative pattern. Repeating the identical
collector cannot satisfy E98's requirement of two successes for every prompt.

E98 therefore remains a failed feasibility result. E98-R1 is a prospective
repair with a separately named estimand and ledger. No E98 dependency is
released or rewritten.

## Frozen repair

E98-R1 reuses the immutable, seed-specific E98 collection sidecars exactly as
written: four draws of 16, temperature 0.7, top-p 0.95, collected from the
paired terminal E78 Dr.GRPO control. No new candidates are generated and no
trajectory is borrowed across prompts, seeds, or domains.

For each training prompt:

- if its own frozen pool contains at least two verified trajectories, sample
  two without replacement using the deterministic E98 sampler and train on the
  registered RLEP-Dr 16-fresh-plus-2-replay mixed Dr.GRPO baseline;
- otherwise, perform the unchanged 16-row E78 Dr.GRPO update with no replay
  rows and no altered advantage normalization.

All 384 prompts remain in every pass. Ineligible prompts are not dropped or
reweighted. Empirical response frequency is preserved; there is no
canonicalization, deduplication, or mode balancing.

## Cohort and estimand

- Model: Qwen2.5-0.5B-Instruct from the same base initialization as E78.
- Domains: Graph Coloring, Python Factors, PantryPlan.
- Seeds: 43--47, paired to E78 control.
- Schedule: 384 prompts x 8 passes, evaluations/checkpoints every 192 updates.
- Primary estimand: paired terminal E98-R1 minus E78-control difference within
  domain and seed.
- Interpretation: this estimates sparse, prior-policy-supported success replay,
  not the infeasible all-prompts intervention registered as E98.

The fixed replay dose is two rows on eligible prompts and zero otherwise.
Report pool eligibility, realized replay-update fraction, replay-row count, and
replay gradient diagnostics beside accuracy and breadth.

## Hard gates

Before submission, every reused pool must pass a content-bound audit requiring
one complete E98 sidecar, the original 16 x 4 / T=.7 / top-p=.95 settings,
exactly 384 prompt references on every draw, and at least one prompt eligible
for the two-row replay dose. The audit records eligible and ineligible counts;
there is no outcome-dependent minimum eligible fraction.

A non-scientific 32-prompt Graph/s43 smoke runs first. Those frozen first 32
pool rows contain both replay-eligible and fallback prompts. A CPU smoke audit
must observe a terminal receipt, both values of `rlep_replay_eligible`, exactly
two replay rows on eligible updates, zero on fallback updates, and finite
RLEP loss/advantage telemetry. All 15 scientific cells depend on that audit.

A missing receipt, pool hash drift, wrong replay dose, canonical replay,
non-finite training value, traceback, or missing terminal endpoint fails
closed.

## Scheduling and reporting

E98-R1 smoke, smoke audit, and scientific jobs may be prioritized ahead of
other not-yet-running comparator work. Existing running jobs are not
preempted. Scheduling changes do not alter model, data, pool, objective,
placement, seed, horizon, or evaluation.

The amendment is disclosed as post-E98-feasibility repair. E98's original
failure remains reported and is never pooled with E98-R1 as though the
all-prompts gate had passed.
