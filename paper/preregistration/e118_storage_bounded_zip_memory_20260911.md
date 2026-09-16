# E118 storage observer bounded ZIP reads and CPU memory amendment

The existing CPU observer31220594 stalled while reading a concurrently written
optimizer ZIP. Its process held the shared storage and ledger locks, consumed
2.21GiB RSS, and recorded73098 memory.high events. A ZIP footer read without an
explicit byte count can read newly appended tensor data after seeking relative to
an earlier EOF. No GPU release intent was outstanding and all owned holds stayed
intact. The original seven-day deadline remains2026-09-18T01:09:01.650458+00:00.

Install a process-local read-only metadata wrapper in the observer. Capture file
identity,size,andmtime when opening; bound every read,including read(-1),to the
captured EOF,at most8MiB per call and16MiB cumulative; bound member count to65536;
and require stable fstat before granting metadata credit. Member/tensor reads are
forbidden. The existing storage helper treats invalid or growing ZIPs as lacking
current-checkpoint credit and reserves two future copies. Both existing storage
helper files and all scientific sources remain byte-identical.

Archive the original observer source,plan,registration,transaction and batch
script before applying this operational amendment. Prepare a pinned replacement
source and plan before mutation. Requeuehold only the exact CPU job once,verify
it inactive with all GPU holds/receipts intact,then change only its requested
memory2GiB to8GiB. Preserve its original submitted command,eligible pool and915
exclusion,Nice,CPU count,walltime and absolute deadline. While the CPU is stopped,
promote the bounded source and updated plan/registration/transaction bindings
under the existing shared locks,recording the archived lineage. Release only the
same CPU ID and verify fresh completed observation passes. Preserve every GPU ID,
resource,checkpoint,dependency and scientific outcome. No GPU is requeued,held,
released or canceled by this amendment transaction.
