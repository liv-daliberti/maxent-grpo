# E116 preregistration: sparse RLEP-Dr direct-comparator completion

Date frozen: 2026-08-19, before submission of any E116 collection, audit,
smoke, or scientific cell.

## Question and estimand

Does generic prompt-matched verified-success replay preserve verified-answer
breadth relative to Dr.GRPO across every registered model/domain block? The
estimand is the paired terminal-pass difference, sparse RLEP-Dr minus Dr.GRPO,
within model, domain, and seed. This is the E98-R1/E100 sparse treatment—not
the infeasible E98 all-prompts treatment and not canonical mode-balanced replay.

## Frozen cells

- Qwen2.5-0.5B: Countdown and MathIR, seeds 43--47 (10 cells), paired to E78.
- Qwen2.5-3B: all five ModeBench domains, seeds 70--74 (25 cells), paired to
  E80-R1.
- Together with E98-R1 and E100, these 35 cells target five seeds for all five
  domains at all three scales.
- Training inherits the exact paired control model revision, data, native
  prompt/action surface, optimizer, rollout group of 16, decoding, 384 prompts
  x 8 passes, and 192-step checkpoint schedule.

## Frozen pool and treatment

For each cell, its paired terminal Dr.GRPO policy generates four independent
draws of 16 candidates for every one of the 384 training prompts at temperature
0.7 and top-p 0.95. Pools are never shared across prompts, seeds, domains, or
models, and empirical trajectory frequency is preserved.

On a prompt with at least two verified frozen trajectories, the learner adds
exactly two frequency-preserving replay rows to the ordinary 16-row Dr.GRPO
update. On an ineligible prompt it performs the unchanged 16-row Dr.GRPO
update. Canonical keys, mode balancing, prompt dropping, and prompt reweighting
are disabled. Pantry allocation witnesses are converted to the already-audited
six-decision `pantry_support_mask` action surface without filtering or
deduplication.

## Hard gates

Each collection job depends on its paired Dr.GRPO job, so an unfinished control
cannot be used as a seed policy. Every pool must pass the sparse pool audit and
contain at least one eligible prompt. The fixed Qwen2.5-0.5B Countdown/s43 and
Qwen2.5-3B Graph/s70 learner smokes must each reach 32 updates; their audits
must observe both eligible and fallback
branches, the registered replay doses, and finite RLEP telemetry. A scientific
cell depends on both its own pool audit and its family smoke audit.

All jobs are submitted held, identity-audited, recorded atomically, and then
released. Failed feasibility remains a reported outcome and cannot be repaired
by selecting a different seed after inspecting pools. Infrastructure retries
may preserve only this frozen configuration.

