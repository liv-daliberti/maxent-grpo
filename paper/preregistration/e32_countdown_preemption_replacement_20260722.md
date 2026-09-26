# E32 Countdown preemption-safe replacement

**Status: FROZEN — prospective, before submission (2026-07-22).**

## Reason for replacement

The six `cde32_freeform_05b_ema_10ep_v2` jobs were preempted together on the
`lowprio` partition before their first optimizer-resumable checkpoint. Their
evaluations through 0.75 prompt passes remain diagnostic evidence, but the
jobs have no learner state from which a scientifically valid continuation can
be made. Those jobs remain held and their curves must not be combined with
this replacement cohort.

## Frozen replacement

- Fresh prefix: `cde32_freeform_05b_ema_10ep_v4_preemptsafe`. The v3
  submission was cancelled while held, with zero elapsed runtime, after its
  audit failed solely because Slurm compressed `node203,node204` to
  `node[203-204]`; it contains no scientific run.
- Qwen2.5-0.5B-Instruct, Countdown easy3, free-form Dr.GRPO and
  EMA-Haarnoja conditional-token MaxEnt, seeds 43/44/45.
- Ten prompt-pool passes from initialization; G=16, learning rate 2e-7,
  response length 192, and all other scientific settings match E32.
- MaxEnt target 1.347109432487438 nats, alpha range 0.000075--0.00060,
  controller learning rate 0.010, and entropy EMA decay 0.7.
- Deterministic pass@1 plus four fixed K=8 evaluations seeded
  1001--1004 at initialization, every quarter pass, and the endpoint. Every
  raw response and draw statistic is retained.
- Optimizer-resumable checkpoints are written at the same quarter-pass
  boundaries (96 updates), retaining the newest two. Restored learner weights
  synchronize to actors before evaluation or rollout.
- Run on the non-preemptible `cs` partition under account `allcs`, restricted
  to nodes 203 and 204, with one A5000, four CPUs, and 32 GiB per job.
- The execution audit requires both the requested prompt-epoch budget and the
  per-attempt ceiling to equal ten; this avoids the superseded wrapper's
  hard-coded five-epoch cap.

The replacement is the only eligible newest Countdown 0.5B cohort. The E32
graph-coloring cohort is unaffected and retains its original identity.
