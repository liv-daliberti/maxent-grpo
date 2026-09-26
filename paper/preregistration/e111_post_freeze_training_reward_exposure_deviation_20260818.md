# E111 post-freeze training-reward exposure deviation

Date recorded: 2026-08-18.

At approximately 18:31 EDT, after the v7 treatment, proposal-retention
deserialization repair, checkpoint quarantine, ZIP-validation selector, E111
protocol, and E112 full-run protocol had all been frozen, a diagnostic
`tail` of job `30674762` stdout unintentionally displayed the
`xdr/progress/rollout_reward` training telemetry for optimizer step 4.

This was not an endpoint evaluation result and was not requested by any gate,
but it is conservatively recorded as post-freeze task-outcome exposure.  The
value is not used to tune, select, release, stop, or interpret the treatment.
No MaxEnt coefficient, ReplayDr weight/objective, proposal policy, optimizer,
data order, model, seed, verifier, target horizon, or evaluation setting may
change in response to it.  Subsequent E111 decisions use only scheduler state,
checkpoint integrity, step completion, and preregistered mechanism telemetry.

The terminal E111 artifact must report both:

- `post_freeze_training_reward_exposure: true`; and
- `outcomes_used_for_gate: false`.

E112 remains unreleased, its treatment stays frozen, and its task outcomes
remain unseen.  PointMaze remains excluded.
