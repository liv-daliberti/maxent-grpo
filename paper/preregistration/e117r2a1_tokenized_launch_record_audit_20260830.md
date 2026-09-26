# E117-R2-A1 tokenized launch-record audit amendment

Frozen: 2026-08-30 after the terminal E117-R2 mechanism audit job 30977278
failed closed, and before any replacement audit submission. This amendment is
audit-only. It changes no training job, source, configuration, artifact,
endpoint, estimand, or release criterion. PointMaze remains excluded.

## Trigger and evidence boundary

All 12 registered E117-R2 cells completed successfully for exactly 64 optimizer
updates each. The version-3 official audit remained terminal and reported no
training-artifact, mechanism, telemetry, leakage, node, or provenance failure.
Its complete failure list contained only:

- eleven \`scheduler record lacks a bounded export block\` failures, one for each
  clean recovery launch; and
- the consequent \`qwen05b/countdown: launch block lacks C/P/F\` failure.

The original Countdown C launch placed \`--export=...\` before \`--partition=...\`.
The eleven clean recovery launches placed scheduler options first and the
single \`--export=...\` token immediately before the batch-script path. Both are
valid \`sbatch\` argument orders, and every launch receipt retains the exact
export token and its preregistered SHA-256 digest.

Diagnosis used only scheduler identity, audit failures, and mechanism telemetry
that the mechanism-only protocol explicitly permits. No evaluation endpoint,
arm contrast, confidence interval, ranking, or efficacy statistic was
inspected or used.

## Frozen correction

Replace only \`exported_environment_text\` in the E117 auditor:

1. tokenize the durable scheduler record with POSIX \`shlex\`;
2. require exactly one \`SubmitLine\` field;
3. accept exactly one \`--export=value\` or \`--export value\` argument at any
   position after \`SubmitLine\`;
4. return that argument's complete token value for the existing byte-level
   environment digest and key/value checks; and
5. fail closed on malformed quoting, a missing/empty export value, or duplicate
   export arguments.

The environment parser, expected C/P/F differences, common-source comparison,
node check, telemetry requirements, leakage checks, optimizer-row rules,
terminal-sentinel rules, proposal/retention invariants, semantic-pressure
invariants, Stage-1 readiness rule, and efficacy boundary are unchanged.

## Regression and freeze requirements

Before submission, the focused E117 mechanism-audit suite must pass and must
include:

- the original \`--export\`-before-\`--partition\` layout;
- the recovery scheduler-options-before-\`--export\` layout with a following
  batch-script path;
- the separate-token \`--export value\` form; and
- fail-closed cases for absent \`SubmitLine\`, absent, empty, malformed, and
  duplicate export arguments.

Create a content-addressed audit snapshot by copying the immutable E117-R2
training snapshot and replacing only
\`ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py\`. Verify every
other copied runtime byte against the base snapshot. Record the base snapshot
identity, corrected auditor digest, protocol digest, and derived identity.

## Official re-audit and authorization boundary

Preserve the failed audit artifact and job 30977278 as superseded evidence.
Submit one held, no-requeue audit using the derived audit snapshot and the same
12 completed training job IDs and receipts. Validate its held scheduler record,
write the amendment and replacement receipts, then release it. No training cell
is rerun.

Stage 1 remains unauthorized unless this replacement official audit reports
both \`passed=true\` and \`stage1_execution_readiness.ready=true\`. Any additional
failure is a new prospective blocker and is not waived by this amendment.
