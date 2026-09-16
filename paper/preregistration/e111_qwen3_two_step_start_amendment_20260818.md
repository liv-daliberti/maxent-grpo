# E111 Qwen-3B two-step checkpoint-start amendment (2026-08-18)

## Scope

This prospective operational amendment applies only to E111 jobs 30674758,
30674759, 30674760, 30674761, and 30674762. PointMaze is excluded.

## Trigger

After installing the preregistered two-step durability amendment, configuration
telemetry from a post-amendment restart showed `resume_steps=2` but
`resume_from=32`. No endpoint outcomes were inspected. The separate start
threshold was materialized by the frozen `run_experiment.sh` before the frozen
`train.sh` override ran, so setting `OAT_ZERO_SAVE_FROM=2` alone did not permit
an early resumable checkpoint.

## Amendment

For the five exact jobs above, the runtime-ops durability block additionally
exports `OAT_ZERO_RESUME_FROM=2`. The marker is extended to report
`checkpoint_start=2`. The submitted interval and start threshold remain part of
the fail-closed identity checks.

This is storage-only recovery behavior. It does not change the model, data,
seed, training horizon, optimizer, semantic coefficient, verified-support
estimator, replay objective, proposal policy, evaluation, hardware, or any
treatment environment. Running attempts are not signaled or reset; the change
takes effect only when a later attempt rereads the frozen runtime script.

## Interpretation

The E111 mechanism gate remains governed by its original terminal audit. This
amendment provides no efficacy evidence and does not authorize E112 release or
any paper claim. The provenance record must chain the live train-script and
durability-block digests from the prior two-step amendment, retain before/after
scheduler records, and state that endpoint outcomes were not inspected.
