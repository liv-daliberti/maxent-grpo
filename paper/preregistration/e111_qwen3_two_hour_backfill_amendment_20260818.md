# E111 Qwen-3B two-hour backfill amendment

Recorded on 2026-08-18 before any E111 Qwen-3B job started or produced an optimizer step. No E111 efficacy endpoint was inspected. The already-running smaller-scale mechanism telemetry is not used to select this scheduler change.

The five Qwen-3B E111 jobs remained pending for priority after the valid `lowprio` partition repair. Slurm estimated starts on August 24-25 because their frozen 12-hour time requests do not fit current A6000 backfill windows. The directly matched completed E104 Qwen-3B 64-update mechanism jobs took 27:26 to 34:03 each. E111 also performs target-free proposal sampling, so the amended limit is a conservative two hours rather than the historical one-hour envelope.

The repair updates only `TimeLimit`, from `12:00:00` to `02:00:00`, for job IDs `30674758` through `30674762`. All five were pending, had zero runtime and zero optimizer steps, and retain the same job IDs, lowprio partition, mltheory account, A6000 node list, A6000 GRES, CPU/memory allocation, environment, source snapshot, data, seed, and treatment. PointMaze remains excluded.
