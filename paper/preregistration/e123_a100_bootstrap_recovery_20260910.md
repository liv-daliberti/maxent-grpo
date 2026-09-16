# E123 benchmark bootstrap recovery, September 10, 2026

The user requested starting E123 as soon as possible. Benchmark allocation
31160003 failed before qualification because its shell sourced the frozen
snapshot's repo_env.sh without binding the real repository root. DeepSpeed
looked for nvcc under snapshot/var/cuda124_toolkit, which does not exist, and
Python created unregistered caches under snapshot/var/pycache. The launch
watcher correctly stopped without submitting scientific jobs.

Preserve the original failed allocation receipts and reviewed 100-cell plan.
Verify all registered snapshot files byte for byte, then move only its
unregistered var cache tree into the recovery audit directory. Keep the exact
snapshot identity, benchmark algorithm, seven runtime candidates, all-five-domain
end-to-end qualification, checkpoint restoration, fixed-update equivalence,
model/data/recipe, admission rules, and automatic measured-plan launch gate.

The replacement benchmark shell explicitly exports the real repository root,
var root, CUDA toolkit, extension cache, Python bytecode cache and temporary
root before sourcing the same frozen repo_env.sh. First run its actual CPU
production-loss contract to verify the repaired bootstrap. Then submit one
held benchmark with the same node302 owner resources: one A100, 8 CPUs, 116 GiB,
24 hours, Nice0, no automatic requeue. Audit and release through the existing
once-only submission/release helper. Original exclusive intents remain intact;
new intents live in the separate bootstrap recovery directory.

The recovery watcher invokes the original benchmark-success-to-launch path.
Only a passing selected profile can publish and submit the 100 held scientific
jobs and arm the existing storage-aware release controller. Preserve all
E118/E119/E120 allocations. Actual measurement and science progress must be
reported separately from prepared or queued state.
