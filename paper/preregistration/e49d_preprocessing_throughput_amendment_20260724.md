# E49D preprocessing-throughput amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

The first three durable v4 records showed that the second temperature-zero
direct proposal and second temperature-zero rescue proposal repeated the same
certification outcome. The rain and fixed-doubling rows remained certified
singletons after all repeats; the modular-arithmetic row found its certified
two-route clique in the first new direct proposal.

The answer-blind search budget is therefore one direct proposal plus one
route-ideation rescue proposal, in addition to any imported proposal that is
re-audited. The worker count is raised from four to eight, exactly matching
the frozen Qwen72 server's `max_num_seqs=8`.

This amendment changes only preprocessing latency. Audit roles, seeds,
temperature, answer isolation, maximum-clique selection, singleton fallback,
retained-menu hashes, policy data, runtime execution gate, matched arms, E46
controller, data cohorts, training schedule, and advancement criteria are
unchanged. The first three v4 records remain valid because their durable
certifications are recomputed independently of proposal-attempt count or
worker scheduling.
