# E112: corrected verified-support MaxEnt plus ReplayDr full evaluation

Date frozen: 2026-08-18, before inspecting any E111, E105, or E109 task-outcome endpoint and before E111 terminal gate release.

## Question

Does the implementation that realizes the intended verified-support entropy mechanism improve retained correct-mode breadth when added to ReplayDr.GRPO at Qwen2.5-0.5B, Falcon-1B, and Qwen2.5-3B?

E112 is the efficacy successor to the structurally inactive E105 v6 treatment. It is conditional on a passing terminal E111 mechanism audit. PointMaze is excluded.

## Frozen treatment

Every E112 cell uses exactly the E111 treatment identity:

- Dr.GRPO task updates with beta zero;
- uniform verified-likelihood ReplayDr with coefficient 0.10;
- the v7 persistent verified-support semantic score-function estimator with coefficient 0.10, pseudocount one, surprisal clip five, and no structural unseen bucket;
- verified replay/proposal keys enter predictor support membership with count zero and never enter neutral-policy frequency counts;
- one target-free original-prompt proposal group of 16 at temperature 1.2 and one attempt per eligible update;
- validator-positive novel proposals enter only the replay bank, never PPO;
- replay-priority visits zero and priority multiplier one, so ReplayDr mass remains uniform;
- diagnostic retention tracking enabled and adaptive retention priority disabled; and
- v5, v6, RMS control, token entropy, UCPO, RLEP, transformed proposals, exact grammar transforms, gold support, desired-mode feedback, and evaluation feedback disabled.

No coefficient, domain, scale, seed, or horizon may be selected from E111 task outcomes. The only inspected E111 fields before this freeze were the preregistered mechanism fields needed to verify discovery, external support, support size, v7 pressure, replay gradient, uniform mass, and leakage guards.

## Paired design

The full grid is 3 model scales x 5 natural domains x 5 seeds = 75 treatment cells:

- Qwen2.5-0.5B seeds 43-47;
- Falcon-1B seeds 55-59; and
- Qwen2.5-3B seeds 70-74.

Domains are Graph coloring, Countdown, Python factors, MathIR, and PantryPlan. Each cell uses the first 384 training prompts for eight passes, for 3,072 optimizer updates with group size 16. Checkpoints and neutral sampled evaluation occur every 192 updates. The evaluation protocol, four registered sampled draws, prompt surfaces, verifier, optimizer, learning rate, and model-specific action interface remain those of the matched released scale campaign.

Each cell is paired by model, domain, and seed to the existing ReplayDr.GRPO comparator. Python uses the parser-repaired E109 comparator; non-Python uses the corresponding released E78, E79, or E80-R1 replay arm. Qwen-3B treatment placement follows the already frozen paired A100/A6000 comparator placement record so treatment and comparator hardware class match cell by cell.

## Release gate

Submission must freshly run the E111 outcome-blind auditor and fail closed unless:

- all 15 E111 jobs are terminal and each reached 64 updates without a failure marker;
- the E111 ledger, runtime snapshot, unit evidence, and scheduler-amendment evidence validate;
- the outcome-blind proposal-retention checkpoint-deserialization recovery evidence passes, with no optimizer-update or treatment change;
- the Pantry partial-checkpoint quarantine is verified after requeue, and auto-resume selects only complete model+optimizer ZIP checkpoints;
- the recorded post-freeze E111 training-reward exposure is disclosed and remains unused by the mechanism gate or treatment selection;
- every frozen safety and leakage invariant passes; and
- each of the three model scales contains at least one same-cell discovery-to-support-to-v7-to-uniform-ReplayDr causal chain; and
- the exact 75 E105 jobs are no longer active, with the prospective E105 supersession record present.

All 75 E112 jobs are submitted held. Their exact environment, source snapshot, treatment flags, paired comparator identity, and scale-specific scheduler placement are audited before atomic release. Any failure cancels only the newly submitted held E112 jobs and writes no released ledger.

Evaluation remains on the common 192-update half-pass grid. For the ten Qwen-3B cells whose paired comparator placement is the preemptible low-priority A6000 pool, storage-only save/resume checkpoints occur every 64 optimizer updates, beginning at update 64; all other cells retain the 192-update save/resume cadence. This prospective recovery rule does not change training data, sampling, losses, optimizer updates, evaluation times, treatment identity, or the paired estimand. It limits discarded work after preemption and is frozen before any E112 job is submitted or any E112 outcome exists.

## Outcomes and analysis

No E112 or E109 task outcome is inspected before the full paired cohort is ready. The primary paired endpoint is correctness-adjusted breadth at pass 8,

    (distinct@8 - pass@8)_E112 - (distinct@8 - pass@8)_ReplayDr,

with pass@8 reported beside it. The secondary trajectory summary is the paired eight-pass area under the adjusted-breadth curve. Seed-level paired points are always shown; a mean and paired two-sided 95% Student-t interval are shown only for complete five-seed blocks. Missing or failed cells remain explicit and are never silently dropped or replaced.

E105 artifacts remain audit-only and are never pooled with E112. E109 remains the valid Python ReplayDr comparator. No paper efficacy claim is released from E111 mechanism telemetry alone.
