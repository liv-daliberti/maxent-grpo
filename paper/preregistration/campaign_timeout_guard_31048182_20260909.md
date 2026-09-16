Operational amendment: timeout guard after Pantry31048182 host-memory recovery

Date:2026-09-09. The user authorized recovery of broken E118/E119/E120 workloads and faster completion. This amendment concerns only existing E119 Qwen0.5B Level2 Pantry ReplayMaxRL seed44, job31048182. Scientific outcomes were not used to select the recovery.

The previous96GiB allocation exhibited more than21minutes of weight synchronization per update. Same-ID recovery resumes the validated checkpoint at480 with116GiB host memory and unchanged36-hour allocation limit, A6000-only node205/node207 pool, seed, source snapshot, training exports, optimizer, replay bank, evaluation and3072-step budget. Host-memory throttling is a strong inference; direct cgroup evidence was unavailable.

The previous timeout guard remains immutable and must first record manual_stop for31048182 because MinMemoryNode changed. This separate guard adopts only that job after verifying that stop and the116GiB amendment. The repaired allocation may still be pending; the guard observes its queue state without changing priority or placement. It watches for at most48hours and allows at most one same-ID checkpoint retry after an inactive, accounted TIMEOUT. It never interrupts a running or pending training allocation. A valid newer checkpoint, sole-writer check, unchanged frozen resources and runtime, and an owned requeue hold are required before release. Resource changes, another failure state, checkpoint problems or an uncertain transaction stop automatic action.

No new scientific cell, seed replacement, additional training pass, shortened evaluation, optimizer change or publication claim is authorized by this operational amendment. The CPU-only monitor may run on node915 under the existing authorized account.
