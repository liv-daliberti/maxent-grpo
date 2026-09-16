# E111 scheduler-only partition amendment

Recorded prospectively on 2026-08-18, before any E111 optimizer step or outcome endpoint was produced or inspected.

All 15 E111 jobs were submitted with `--partition=all --account=mltheory` and passed the held environment, node-list, GPU, and time-limit audit. Slurm interpreted the special partition token `all` by selecting the higher-priority `mltheory` partition. That partition contains only nodes 105, 302, and 915-917, none of which intersects the preregistered healthy node pools, so every released E111 job remained `PENDING (BadConstraints)` at optimizer step 0.

The repair is scheduler-only: update `Partition=lowprio` for the same 15 jobs. The `lowprio` partition permits account `mltheory` and contains all nodes in both frozen E111 node lists. No job is canceled or replaced; job IDs, environment, source snapshot, model, data, seeds, treatment flags, target steps, node lists, GPU constraints, memory, CPU count, and time limits are unchanged. Preemption/requeue is already supported by the frozen runtime.

- E111 ledger SHA-256 before amendment: `570f2a46562069e660b529276a043b5ef6fde0f3348588997635e7b69d6a9362`
- Exact job IDs: `30674728`, `30674729`, `30674730`, `30674731`, `30674732`, `30674733`, `30674754`, `30674755`, `30674756`, `30674757`, `30674758`, `30674759`, `30674760`, `30674761`, `30674762`.
- Pre-amendment state: all 15 pending with `Reason=BadConstraints`, runtime zero, campaign progress 0/960 optimizer steps.
- Outcomes inspected: none.
- PointMaze: excluded.
