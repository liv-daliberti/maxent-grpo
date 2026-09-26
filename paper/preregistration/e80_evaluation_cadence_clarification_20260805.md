# E80 evaluation-cadence clarification

**Recorded on 2026-08-05 immediately after the first job's startup log and
before any E80 evaluation outcome or optimizer update was available.**

The frozen E80 protocol and scheduler environment request evaluation and
resumable saving every 192 prompt updates (one half-pass). The shared
`ops/run_experiment.sh` safety policy independently caps evaluation to one
quarter-pass, or 96 prompt updates for a 384-prompt pool. It does not cap the
separate save/resume interval. Therefore the realized lifecycle is:

- evaluation artifacts at passes 0, 0.25, 0.5, ..., 8.0;
- resumable checkpoints at passes 0.5, 1.0, ..., 8.0; and
- registered E80 reporting and AUC only on passes 0, 0.5, 1.0, ..., 8.0.

The extra quarter-pass evaluations are monitoring diagnostics. They may not be
used for early stopping, checkpoint selection, cell replacement, changing the
optimizer recipe, or changing the registered estimand. This clarification
changes no submitted environment, training data, objective, sampling draw,
optimizer update, checkpoint interval, or job identity. The same hash-bound
wrapper enforces the behavior in both arms and all 50 cells.

