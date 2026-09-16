# E123 A100 benchmark harness repair, September 11, 2026

The user asked whether more work could be started. Allocation 31170074 failed
before qualification: the benchmark shell did not put the existing Python
runtime's bin directory on PATH, so four offloaded candidates could not find
Ninja; GPU Adam reached optimizer-moment sampling and failed an index bounds
assertion. Float32 linspace can round a large tensor's final valid index up to
its element count. The two combined GPU candidates separately recorded CUDA
out-of-memory failures; this amendment does not reinterpret those failures.

Archive the failed plans, launcher, harness, logs, seven candidate receipts,
suite status, and scheduler failure. Preserve every original file and all
frozen scientific snapshot files. A versioned benchmark harness replaces only
diagnostic sample-index construction with exact integer arithmetic: at most
256 evenly spaced indices, including both endpoints. The relative-RMS moment
comparison and its tolerances remain unchanged. A new launcher prepends the
same paper310 Python bin directory to PATH after sourcing the same frozen
repo_env.sh and requires its installed Ninja to be discoverable.

Use a fresh output directory and plan, rerunning all seven original profiles
with eight warmup and 32 measured updates. Preserve all model/data/seed/loss,
full-shape stress, all-five-domain actor/learner evaluation, checkpoint restore,
fixed-update equivalence and memory gates. The original combined GPU profiles
retain first preference, ordered by the same measured update time. Prospectively
admit the already registered gpu_adam profile (physical microbatch 1, one
optimizer thread, GPU Adam) only if neither combined profile qualifies. Record
both earlier failures in an immutable, hash-bound fallback receipt. This is an
execution-profile eligibility amendment, not a new scientific arm. The fallback
must pass the same full qualification; passing an offloaded baseline alone
never admits a profile. The suite may still fail because of memory limitations.

CPU verification must reproduce the float32 boundary defect without allocating
large optimizer states; test integer indices including zero, singleton, above
2**24, and int64 limits; check moment-comparison mathematics and rejection;
verify runtime PATH/Ninja discovery; and rerun the complete production-loss
contract on the immutable E123 source. Compare old and new plans and harness
ASTs to prove that scientific inputs, seven measured profiles, and numerical,
checkpoint, evaluation and memory gates are unchanged. Test combined-profile
preference, failed and unproved fallback rejection, and the exact permitted
candidate/environment maps in the versioned launcher validator.

Prepare one reviewable benchmark command with the original node302 owner
resources: one A100 80GB, eight CPUs, 116 GiB, 24 hours, Nice0, no requeue.
Preparation and CPU tests do not submit it. This packet intentionally exposes
no submission or watcher phase. Before a subsequent launch, separately review
shared-storage registration and admission: the benchmark is currently an unknown
writer, the observed storage margin is below its conservative checkpoint peak,
and the existing E123 science controller does not include all released pending
external writers, full E122 terminal reserves, or the shared admission mutex.
Do not arm that controller unchanged. A future handoff must also authenticate
the explicit operational source-pin transition while preserving all 100 cells.
A passing selected profile and fresh capacity proof remain prerequisites.
