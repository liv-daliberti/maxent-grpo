# E118-R2 explicit runtime-variant repair

Frozen after two E118-R1 jobs reached configuration validation but before any
optimizer step. The borrowed verified-support wrapper forcibly enabled semantic
advantages at coefficient zero. Register explicit MaxRL and ReplayMaxRL runtime
variants that set semantic objectives to zero and differ only in whether the
verified replay derivative is compute-only or active. Preserve every other E118
setting, use fresh run paths, one node302 A100 per job, and 12-hour limits.
