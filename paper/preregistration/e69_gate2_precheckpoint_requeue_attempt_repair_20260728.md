# E69 Gate 2 pre-checkpoint requeue attempt repair

Date frozen: 2026-07-28, while all Gate 2 runs were nonterminal and before
either affected accepted attempt reached pass 1.

## Infrastructure event

The live Gate 2 audit originally took the maximum optimizer step observed in a
run's append-only `train_metrics.jsonl`. Raw sequence inspection showed that
two low-priority jobs had been requeued before their first optimizer
checkpoint and restarted from initialization:

- Graph compute-matched Dr.GRPO job `30160592` recorded steps `0--174`, then
  restarted at step `0`;
- Python verified-route successor job `30160205` recorded steps `0--203`, then
  restarted at step `0`.

Both current Slurm attempts report watchdog restart count `1/8` and
`auto_resume=no checkpoint found; starting from initialization`. Neither
abandoned prefix produced an optimizer checkpoint or any evaluation beyond
the fixed step-0 baseline. In both jobs, the two step-0 evaluation records
from the abandoned and accepted attempts are byte-identical.

This is a placement/preemption event, not a scientific outcome. It was
detected from optimizer-step regression and checkpoint absence. No pass-5,
pass-6, or terminal result existed.

## Frozen attempt selection

The abandoned prefixes are excluded wholesale:

- job `30160592`: training-metric lines `1--175`, ending at step `174`;
- job `30160205`: training-metric lines `1--204`, ending at step `203`.

The accepted attempts begin at line `176` and line `205`, respectively, each
at optimizer step `0`. The exact abandoned-prefix SHA-256 values and duplicated
step-0 evaluation SHA-256 values are recorded in
`var/artifacts/e69_gate2_precheckpoint_requeue_attempt_repair_identity.json`.

The audit must:

1. verify the exact recorded prefix hash and reset boundary;
2. require that the accepted attempt begins at step `0`;
3. reject any later step regression not covered by a prospective amendment;
4. count training compute and terminal route telemetry only from the accepted
   attempt;
5. continue requiring the complete fixed evaluation surface through pass 6.

The physical job IDs, source snapshot, model, data, seeds, variants,
coefficients, evaluation seeds, and terminal checkpoint remain unchanged.
Discarded pre-checkpoint attempts are not spliced into accepted training.
Repeated step-0 evaluations are retained only as a byte-identical
repeatability check.
