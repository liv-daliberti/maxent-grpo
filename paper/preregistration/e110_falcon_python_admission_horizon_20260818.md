# E110: Falcon Python admission-horizon replacement gate

Frozen on 2026-08-18 after E106 Falcon Python job `30640330` completed its
registered 64 optimizer steps and failed only the verified-admission/replay
criteria. The diagnosis uses training-mechanism telemetry and previously
completed campaign configuration only; no E104/E106 post-update evaluation
outcome was inspected. PointMaze is excluded.

## Why the 64-step cell is not a valid mechanism test

The completed E106 cell used the repaired parser, exact v6 snapshot, seed 55,
Falcon3-1B-Instruct, the Python-factor train split, `falcon_boxed`, 16 samples,
and the same 512-token generation surface as the matched Falcon ReplayDr
campaign. It reached step 64 with v6 active and legacy/controller paths off,
but admitted no verified row; therefore its replay bank and semantic history
were empty and the proposed mechanism had no eligible support on which to act.

The already-completed matched Falcon ReplayDr seed-55 Python run `30269053`
used the same model, seed, data, prompt, group size, verifier surface, and
512-token generation budget. Its first nonempty verified replay bank occurred
at optimizer step 179. This evidence predates E106 and is a training-mechanism
quantity, not an evaluation endpoint. A 64-step admission requirement therefore
tested early discovery luck rather than whether the repaired MaxEnt objective
acts correctly once ReplayDr has verified support.

## Replacement cell

Run one fresh Falcon3-1B Python cell from the unchanged E106 snapshot and base
initialization with:

- seed 55, the same Python-factor train/eval data and `falcon_boxed` prompt;
- the exact E106 repaired parser, v6 sampled-group-centered semantic objective
  at coefficient 0.1, and verified ReplayDr objective at weight 0.1;
- 16 online samples, the inherited 512-token train/eval generation budget,
  learning rate, optimizer, verifier, sampling, and batching settings;
- 192 training rows for one prompt epoch, with evaluations and joint
  checkpoints at steps 64, 128, and 192; and
- one A6000, 8 CPUs, 64 GiB, partition `all`, account `allcs`, node pool
  `node[103-104,205-208,805]`, and a three-hour limit.

The new run must use a distinct E110 output directory. The failed E106
directory remains immutable and auditable. E110 supersedes job `30640330` only
for the effective Falcon/Python cell in the combined mechanism gate; it does
not erase or reinterpret the registered E106 failure.

The first held submission attempt, job `30647351`, had runtime `00:00:00` and
was cancelled transactionally before ledger creation or release because the
site submission plugin canonicalized `--partition=all` to `Partition=cs`.
Subsequent application must submit held on `cs`, update that still-held job to
`Partition=all` with the same A6000 node pool, audit the effective record, and
only then persist and release it. This is scheduler bookkeeping only; the
scientific command and environment are identical.

## Frozen gate

E110 passes only if its scheduler job exits successfully at step 192, v6 is
active on every training row, legacy v5 and the semantic controller are off,
verified Python admission populates both the replay bank and semantic history,
verified replay produces an applied gradient, the centered semantic mean stays
within the existing numerical tolerance, and no runtime failure marker occurs.
Nonzero semantic pressure is required only if at least two verified modes have
become available; singleton support remains an exact-zero case by theory.

The combined E104/E106/E110 gate remains locked until Qwen2.5-3B Python also
finishes and all fifteen effective cells pass. Only then may E109 and E105 be
released.
