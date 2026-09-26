# E67 zero-step partition amendment

Recorded 2026-07-27 after all 12 E67 jobs were pending with zero optimizer
metrics. Low-priority reservations and preemption prevented progress even
after the compatible-capacity node amendment.

Read-only `srun --test-only` probes then found immediate capacity on partitions
already advertised by the same GPU nodes:

- A6000 and RTX 3090 on `pvl-lowprio`;
- A100 on `mltheory`.

This non-outcome amendment may change only Slurm partition and, for Graph, add
the already registered A6000 `node103` back to the compatible node list.

- Graph `30128500--30128502`: `pvl-lowprio`, A6000 nodes
  `node103,node104`.
- Countdown/Python `30128503--30128508`: `pvl-lowprio`, RTX 3090 nodes
  `node020,node021,node022,node024,node026`.
- MathIR `30128509--30128511`: `mltheory`, A100 `node302`.

Every job must be held and have no `train_metrics.jsonl` before mutation. No
SubmitLine environment, source, execution snapshot, model, data, seed,
optimizer, objective, controller, coefficient, proposal, replay, evaluation,
checkpoint, or recovery setting may change. This document and the mutation
script are SHA-256-bound before release.
