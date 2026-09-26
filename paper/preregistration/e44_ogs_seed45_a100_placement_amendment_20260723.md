# E44-OGS seed-45 A100 placement amendment

**Status: FROZEN BEFORE SEED-45 ALLOCATION (2026-07-23 18:10 EDT).**

The matched Countdown seed-45 jobs remained pending with zero runtime:

- `30072670`: `cde44_ogs_canonical_maxent_05b_v1_grpo_s45`;
- `30072671`: `cde44_ogs_canonical_maxent_05b_v1_online_canonical_maxent_s45`.

Their original A5000 pool was saturated: node105's ten A5000s were occupied by
the other ten E44-OGS jobs, and every A5000 on nodes202--204 was allocated to
external workloads. Node302 had eight idle A100 GPUs in the same `all`
partition and `mltheory` account.

At the user's direction, both pending Countdown seed-45 arms change placement
together from one A5000 on `node105,node202,node203,node204` to exactly one A100
on `node302`. This is an operational placement amendment, not a treatment
change. The two arms remain paired on the same GPU class.

The following remain immutable:

- run stamps and Slurm job IDs;
- model and tokenizer snapshot;
- source and execution-surface snapshots;
- task data, prompt order, training seed, and evaluation seeds;
- group size, optimizer, learning rate, update count, and ten-pass budget;
- Dr.GRPO control objective;
- E44-OGS validator-bound admission, bank update, entropy coefficient, novelty
  coefficient, and post-task-centering advantage placement;
- evaluation cadence and neutral K=8 protocol.

All graph-coloring runs and Countdown seeds 43 and 44 retain A5000 placement.
Countdown seed 45 therefore differs in hardware class across training seeds,
but the scientific contrast remains paired within seed because both arms move
together. Results must disclose this placement amendment.
