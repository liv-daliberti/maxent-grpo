# E101m: open-bank fresh-admission mechanism smoke

Date frozen: 2026-08-14, before submission.

## Purpose

E101m is a single-cell fail-fast diagnostic while the registered E101 A6000
ladder waits for scheduler placement. It asks only whether the clean open-bank
path executes end to end on Countdown:

1. a sample from the untouched original prompt is attempted;
2. the ordinary validator and canonicalizer either reject it or admit a novel
   verified sequence to replay-only support;
3. proposal rows sent to PPO and feedback remain exactly zero;
4. if admission occurs, a later replay update actuates the enlarged bank.

It is not an additional E101 comparison arm. Accuracy, pass@K, and diversity
are descriptive only and cannot be compared with the pending 32-prompt cells.

## Frozen cell

- Model and immutable source snapshot: identical to E101,
  Qwen2.5-0.5B-Instruct at snapshot
  `var/artifacts/source_snapshots/e76_tuned_scale_0d87aba8529891ef`.
- Objective: task-only Dr.GRPO plus replay mass 0.10 and known-bank balance
  0.10; every semantic/PPO-advantage coefficient is zero.
- Discovery: one singleton-only fresh proposal attempt at temperature 1.0;
  deterministic transforms are off and objective support is separate.
- Data: the frozen E101 Countdown split, first 8 training prompts and all 32
  disjoint evaluation prompts; seed 102; one pass, hence 8 optimizer updates.
- Evaluation: steps 0 and 8, batch size 8, K=4, one fixed draw. These readouts
  exist for runtime integrity, not treatment comparison.
- Placement: `all`/`mltheory`, one generic GPU from the frozen healthy
  >=24 GB whitelist `node[007,020,022-023,101,103,202,204-206,302,403,805]`,
  8 CPUs, 22 GB host memory, and a hard 30-minute wall-time. At the sixth
  replacement freeze, Slurm's non-submitting test selected node805 immediately.
  Prior
  short 0.5B E56 smokes peaked at 15.2--15.3 GiB RSS; the much longer Countdown
  sentinel peaked at 20.9 GiB. The 22 GB cap is therefore a measured fail-fast
  choice for this eight-update micro-run. An OOM is a failed attempt, not
  objective evidence. GPU type is not used for treatment comparison or
  performance interpretation.

Initial job 30580872 requested node805 only. Before it allocated any GPU or
materialized a run directory, a new node805 workload consumed the remaining
scheduler memory. That elapsed-zero job is superseded by a replacement whose
only change was a same-family node-list expansion.

First replacement job 30581047 requested node206 or node805. It also remained
pending at elapsed zero with no run directory after both nodes became planned
for other work. The second replacement moves this mechanism-only smoke to the
then-available 48 GB A40 placement; the data, objective, seed, and registered
interpretation are unchanged.

Second replacement job 30581116 requested node101's A40. It likewise remained
pending at elapsed zero with no run directory after node101 became planned for
other work. The third replacement uses the valid `mltheory` association and the
measured-memory A100 placement above. The data, objective, seed, wall-time, and
registered interpretation remain unchanged.

Held mltheory attempt 30581503 was accepted but mapped to partition `all` by
the current job-submit policy; the launcher's held audit failed closed and
canceled it before release. Held diagnostic 30581540 then established that an
explicit pending-job partition update restores `mltheory`; that probe was also
canceled without release. The launcher now performs and records that held-only
update before its normal audit and release. No job is released on a partition
different from the registered placement.

Released third replacement 30581575 remained pending at elapsed zero and never
materialized a run directory. A live resource audit showed why: node302's six
active jobs reserved all 480 GB of schedulable host memory, stranding two
otherwise unallocated A100s. The fourth replacement moves only this
mechanism-only smoke to the available node403 L40 placement above. Its data,
objective, seed, decoding, registered interpretation, memory cap, and wall-time
are unchanged.

Fourth replacement 30581868 used the valid but lower-priority `allcs` account
on node403. It remained pending at elapsed zero and never materialized a run
directory. A cluster-wide non-submitting audit then isolated account fair-share
as the bottleneck: 15-, 20-, and 30-minute requests and 16--22 GB memory caps
all received the same sliding estimate, while the existing `mltheory`
association on the same `all` partition and node403 received an immediate
estimate. Partition `all` explicitly allows `mltheory`. The fifth replacement
therefore changes only the billing/fair-share account; node, GPU type, data,
objective, seed, decoding, memory cap, wall-time, and interpretation are
unchanged.

Fifth replacement 30582087 used `mltheory` correctly but retained the explicit
node403 pin. It remained pending at elapsed zero, never materialized a run
directory, and reported that the requested node was reserved. The sixth
replacement removes that avoidable singleton-host constraint: Slurm may choose
one GPU only from the frozen legal whitelist above. Every member has at least
24 GB VRAM, the class on which prior short 0.5B runs completed. Data, objective,
seed, decoding, memory cap, wall-time, and interpretation remain unchanged.

Sixth replacement 30582102 started immediately on node202 but failed the frozen
argument validator before model training: counterfactual proposals require
replicated free-form sampling. It consumed 65 seconds, reached zero optimizer
steps, and produced no scientific metrics. The repaired `r1` run adds the same
one-GPU replicated sampling and local actor weight-sync flags used by prior
valid proposal experiments. E101r1 applies those flags to every comparison arm;
the smoke changes no objective, data, seed, decoding, or interpretation.

## Interpretation

- No proposal attempts: implementation or eligibility failure.
- Attempts but no admissions: discovery is the immediate bottleneck.
- Admission with any proposal-to-PPO or proposal-to-feedback row: invalid.
- Admission without later replay actuation: scheduling failure.
- Admission, zero leakage, and later replay actuation: the new-mode mechanism
  works and the pending E101 ladder may evaluate its comparative effect.

No coefficient, decoding, data, or resource change is allowed after metrics
materialize. Any repair receives a new run stamp and amendment.
