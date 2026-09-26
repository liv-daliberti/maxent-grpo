# E117 Stage-1 S7 checkpoint-resume repair

Date: 2026-09-01

## Trigger

Jobs `30980499`, `30980502`, and `30980505` used the prospective five-hour
pre-maintenance A100 backfill window, wrote coherent DeepSpeed checkpoints, and
ended by Slurm timeout before the terminal registered step.  Their job records
then aged out of the live controller, leaving the first successor in each block
with `DependencyNeverSatisfied`.

## Repair

Submit exact frozen-run replacements with no scientific-field changes.  The
runner's frozen checkpoint validator must automatically select the highest
coherent checkpoint below each unchanged `SAVE_PATH`: Python seed 203 F at step
2816, Falcon MathIR seed 201 C at step 1216, and Falcon MathIR seed 202 P at
step 1216.

Keep all three replacements on `node302` A100 under the `mltheory` account and
partition, matching the hardware used before timeout.  Replace the temporary
five-hour backfill bound with a nine-hour limit sufficient for the registered
remaining updates.  Preserve source snapshot, model/data identities, arm,
seed, optimizer, proposal/replay/semantic settings, checkpoints, evaluations,
and all token budgets.

Reconnect jobs `30980500`, `30980503`, and `30980506` respectively with
`afterok` dependencies on the replacement jobs.  Preserve every downstream
within-block dependency.  This repair is based only on scheduler state and
checkpoint integrity, not outcomes.
