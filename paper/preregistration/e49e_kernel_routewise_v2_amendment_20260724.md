# E49E finite-kernel route-wise v2 amendment

**Status: FROZEN BEFORE ANY V2 REQUEST OR POLICY TRAINING — 2026-07-24**

The first finite-kernel run revealed a proposal-admissibility failure before
policy training: a bank was discarded in full when any one proposed route
contained a forbidden derived numeral, repeated a kernel label, or repeated
an operation-code sequence. This behavior is safe but unnecessarily discards
independently admissible routes and is especially harmful on original
zero-support rows.

V2 changes only proposal construction and deterministic selection:

- it uses schema `math_finite_kernel_bank_v2`, proposal seed `492202`, and an
  explicit final scan instructing the proposer to replace every derived
  numeral with a symbolic reference;
- each route is checked independently under the unchanged numeric-leakage,
  finite-kernel, finite-operation, length, and action-reference rules;
- an invalid route is dropped, never repaired or exposed;
- among admissible routes, the first route in frozen proposal order is kept
  for each unique kernel and unique ordered operation-code combination, with
  the same twelve-action bank cap;
- a surviving singleton may fill a coverage gap but cannot create diversity;
  multi-route support still requires both independent execution audits and
  both independent equivalence attacks; and
- for each row, the final selector compares the original trace bank, the V2
  bank, and any strictly larger V1 finite-kernel bank. It retains the largest
  independently certified support, preferring the current/original selection
  on ties.

V1 must be terminal with exact row coverage before V2 is launched. Its
records, the raw 100-row trace evidence, source cohorts, controls, endpoint,
V2 implementation, launcher, and Slurm wrapper are frozen and hashed before
the first V2 request. The five invalid controls, three equivalent controls,
manual blinded pruning, and post-pruning 20-overall/10-evaluation support
thresholds are unchanged.
