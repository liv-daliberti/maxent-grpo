# Read-only continuation startup observer — September9,2026

The user requested recovery and accelerated completion of E118/E119/E120.
Observe only the five already submitted continuations31158503–31158507 and
the automatically requeued existing job31151411. This CPU-only observer
never holds, releases, cancels, requeues, submits or changes science jobs;
it does not change any experiment ledger or inspect evaluation outcomes.

Freeze their exact run directories, full exported environments, frozen
launchers and starting checkpoint thresholds from the completed capacity
transaction. The thresholds are768,960,960,960,576 and1920, respectively.
Scheduler node-pool changes are allowed; scientific/runtime exports must
remain identical. A single-instance lock prevents duplicate observers.

Watch every60seconds for at most24hours from first observation, retaining
the deadline across restarts, and exit when all six cells have current
post-checkpoint optimizer progress or a valid terminal training receipt.
Persist pending/running scheduler records, selected checkpoint and loaded
model/optimizer messages, latest optimizer record, current-attempt fatal
errors and timestamps. For a same-ID restart, use the latest appended
record and require its file timestamp after the new scheduler StartTime;
never use the historical maximum step as startup proof. Use the actual
selected resume checkpoint when the launcher reports it.

Collect one optional read-only cgroup snapshot per successfully advancing
attempt using a one-CPU overlap step with an8second client timeout. Failure
to obtain this optional diagnostic does not block startup verification.
The observer itself needs one CPU,256MiB and no GPU on non-PVL owner node915
under mltheory. Fatal errors, changed identity, pending capacity and missing
telemetry are reported, never repaired automatically by this observer.
