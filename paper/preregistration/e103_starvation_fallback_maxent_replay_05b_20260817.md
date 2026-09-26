# E103: bounded explorer-starvation fallback for replay-side MaxEnt

Date frozen: 2026-08-17, before any E103 GPU smoke or full training cell.

## Motivation and treatment

E102 completed all 25 registered Qwen2.5-0.5B cells. It produced large paired
breadth gains over E78 replay, but correctness uncertainty remained and one
Python cell (seed 43) had only eight proposal admissions, followed by a terminal
stall of 551 eligible proposal updates. E103 tests a discovery-reliability
intervention, not a different learning objective.

Every E103 cell retains E102's task-only Dr.GRPO PPO loss, verified replay-mass
loss, retention-safe whole-bank balance loss, open-bank validator, and fresh-mode
priority settings exactly:

\[
L = L_{\mathrm{PPO,task}} + 0.1 L_{\mathrm{replay\ mass}}
    + 0.1^{\mathrm{safe}} L_{\mathrm{whole\ bank\ balance}}.
\]

Proposal rows are never appended to the PPO batch. Proposal-only support remains
separate from the on-policy objective support. Priority affects only normalized
replay mass; MaxEnt balance continues to compare the complete prompt bank.

The sole full-run intervention is a restart-safe proposal scheduler:

- base search: one original-prompt group at temperature 1.2, as in E102;
- trigger: 64 eligible proposal updates with no newly admitted verified mode;
- fallback burst: 16 eligible updates with up to four groups, using the existing
  1.2, 1.4, 1.6, and 1.8 temperature sweep and isolated request seeds;
- cooldown after an unsuccessful burst: 48 eligible updates at the one-group
  E102 budget, after which the burst may rearm;
- reset: only an actual new validator-positive bank admission.

The scheduler reads no evaluation result, exhaustive support, desired mode
count, target entropy, or gold answer. An update is eligible only when the
ordinary E102 explorer has a verified anchor and reaches original-prompt
sampling. The fallback does not create an anchor and cannot relax validation.

The fixed 64/16/48/4 schedule was selected before E103 execution. Under a
read-only replay of E102 admission streams it would have activated on 0% of
eligible Pantry updates, 0--4.1% in Countdown, 4.8--8.2% in Graph, 12.0--14.5%
in MathIR, and 0--19.8% in Python. This establishes a bounded, selectively
triggered compute intervention rather than a four-attempt-everywhere arm.

## Design and comparators

- Model: Qwen2.5-0.5B-Instruct.
- Domains: Graph Coloring, Countdown, Python Programs, Math Intermediate
  Representation, and Pantry Planning.
- Seeds: 43, 44, 45, 46, 47.
- Training: 384 rows, eight passes, 3,072 optimizer steps.
- Registered checkpoints: every 192 steps, or every half pass.
- New cells: one E103 arm, 25 cells total.
- Primary comparator: the completed, paired E102 full-open-bank cells.
- Secondary comparators: the completed E78 replay and control cells.

E103 does not rerun E102 or either E78 comparator.

## Pre-full verification gate

Before full release, one 32-step seed-43 smoke must complete in every registered
domain from the exact frozen source snapshot used for the full campaign. Smokes
compress only the scheduler timing to patience 2, burst 2, cooldown 2; all loss,
validation, replay, balance, and priority settings remain the full treatment.

The gate is mechanistic and does not inspect task accuracy or breadth:

- each smoke reaches step 32 with finite training loss;
- fallback activation and at least one extra fallback proposal group are seen;
- proposal rows sent to PPO and proposal objective-support delta remain zero;
- retention-safe balance is active and applied harmful replay gradient is at
  most `1e-7`;
- target/gold/evaluation feedback telemetry remains zero.

The full campaign is released if and only if all five mechanism smokes pass.
No smoke outcome metric may be used to change the registered full schedule.

## Confirmatory outcomes

The primary scientific contrast is the equal-domain macro paired difference
E103 minus E102 across seeds. Report raw terminal and half-pass-anchored terminal
estimates, normalized AUC, per-domain paired effects, and uncertainty for:

- greedy accuracy;
- sampled pass@8;
- sampled mean correctness at 8;
- sampled distinct correct modes at 8;
- sampled correct-mode coverage at 8.

Mechanism reporting includes admissions, eligible proposal updates, fallback
activations and duty cycle, extra generated groups, discoveries during fallback,
admission-to-priority replay, proposal-to-PPO leakage, objective-support delta,
retention-safe scale, and applied harmful replay gradient. Proposal generation
groups and realized token/time costs are reported explicitly; optimizer steps
remain the training-budget alignment variable.

Python seed 43 is a prespecified stress-case trace because it motivated the
reliability hypothesis, but it is not a release gate and is not the primary
effect estimate. No single seed or domain may be silently dropped.

## Operational amendment: effective Slurm partition label

After all five smokes passed, the first held full submission (job 30634215) was
canceled before execution because Slurm represented the requested `all`
meta-partition as the effective `mltheory` partition for a two-day job. The
launcher audit had required the literal short-job label `Partition=all`. No full
cell ran and no E103 ledger was created. The held-job audit now accepts only
`Partition=all` or `Partition=mltheory`, while continuing to require
`Account=mltheory`, the exact node allowlist, GPU/memory/time contract, frozen
source, and every scientific environment variable. No treatment, schedule,
metric, seed, domain, or analysis rule changed.

On release of jobs 30634313--30634337, Slurm again assigned the effective
`mltheory` label. That partition contains only nodes 105, 302, and 915--917, so
it was incompatible with the preregistered healthy-node allowlist and all 25
jobs remained pending with `BadConstraints`; none had begun. Each existing job
was updated in place to `Partition=all`, matching the successful E102 execution
placement and its original `sbatch --partition=all` submit line. No job ID, run
directory, source snapshot, environment, treatment, or budget was replaced.
