# E120-R1: shell-forwarding repair before allocation

Recorded 2026-09-03 before any E120 job allocated and before any E120 training
outcome existed.

The original held cohort (Slurm 31033507--31033552) was cancelled while every
job was still pending. Source review after submission found that args.py
defined online_canonical_replay_key_weighting, but ops/train.sh had not
forwarded OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING to the Python CLI.
The smoke would therefore have entered the default uniform branch and failed
its frequency-telemetry audit; all science jobs depended on that smoke and
could not have run.

E120-R1 makes exactly one execution-plumbing repair:

- read the environment value in ops/train.sh;
- pass --online-canonical-replay-key-weighting when the frozen source exposes
  the field;
- fail closed if a non-uniform value is requested against older source;
- print the resolved value in the training configuration log.

A static regression test now requires all three links (environment, shell CLI,
and Python argument), and the launcher's pre-submission test gate includes that
test. E120-R1 uses a new immutable source/ops snapshot, new ledger, run stamps,
and Slurm IDs. The scientific design, estimands, seeds, domains, models,
hyperparameters, smoke design, and no-PVL rule remain exactly those in E120.
No outcome-based decision motivated this repair.

