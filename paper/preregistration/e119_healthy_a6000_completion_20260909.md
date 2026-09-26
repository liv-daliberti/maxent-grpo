# E119 healthy A6000 completion routing amendment — September 9, 2026

The user requested faster E119 completion and use of other eligible capacity.
The earlier node208 opening closed when that node was drained again at19:16 EDT
for overheated GPUs. This prospective route uses only healthy public A6000
nodes205 and207 through lowprio/mltheory. At approximately19:20 EDT, exact
116GiB36-hour requests forecast September10 at22:30, ahead of the comparable
cs/allcs route forecast on September17. These are scheduler forecasts, not
promises of immediate execution.

This is an operational placement amendment for eight existing pending Pantry
cells: MaxRL seed46 (31048187), MaxRL seed47 (31048191), Dr.GRPO seed44
(31048180), ReplayDr.GRPO seed45 (31037836), ReplayMaxRL seed45 (31048185),
ReplayMaxRL seed46 (31048188), Dr.GRPO seed47 (31037843), and ReplayMaxRL
seed47 (31037846). Each requests one A6000, eight CPUs,116GiB host RAM, and
its existing36-hour allocation. The additional host memory increases the
previous96GiB headroom without changing physical microbatch or offload settings.
Previously staged MaxRL seeds44/45 and all separately guarded cells are outside
this controller's scope. No existing128GiB request is reduced.

Preserve every training export, seed, method, frozen source and shell entry
point, model and dataset, run stamp and output directory, evaluation settings,
full-state automatic resume, A6000-compatible vLLM ratio0.25, and target3,072
updates. No outcome or score selects the route. An existing committed checkpoint
must validate model/optimizer/counter state and remain unchanged before release.
Cells with no committed checkpoint retain their original automatic fresh-start
behavior; a newer partial checkpoint requires review.

Site policy requires a new scheduler job for account/partition changes. Recheck
that each exact predecessor is pending and authoritative, persist mutation
intent, hold it, and submit one held continuation. Audit the full submission,
all scientific exports, source/script/data/model identities, resource request,
and the exact normalized node set before promoting one continuation row under
the shared ledger lock. Preserve the100-cell scientific denominator and75-row
continuation ledger. Keep each predecessor as an owned dormant hold until its
successor is established. Interrupted acknowledgements must reconcile recorded
identities; never submit twice or duplicate an active writer. If a predecessor
starts during the hold race, preserve its allocation and skip that clone.

Both candidate nodes must remain healthy, A6000-equipped, and accessible through
lowprio immediately before release. Node206 and node208 are excluded while
drained; private and PVL nodes remain excluded. The scheduler enforces capacity;
this routing change does not interrupt any running E119 or other campaign job.
The lowprio partition may preempt and requeue allocations, so preserve full-state
automatic resume and the existing96-update Pantry checkpoint cadence. Memory or
hardware failures require evidence-based review, not blind retries.
