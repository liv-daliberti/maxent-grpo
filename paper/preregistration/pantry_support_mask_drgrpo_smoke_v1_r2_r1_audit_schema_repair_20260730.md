# Pantry support-mask smoke r2-r1 audit-schema repair

**Status: FROZEN AFTER R2 AUDIT JOB 30199934 AND BEFORE R2-R1 AUDIT EXECUTION — 2026-07-30**

Training job `30199933` completed all 32 frozen updates. Audit job `30199934`
passed every source, execution, data, model, scheduler, metric-order,
learning-round, six-step sampler, exact-entropy, finite-loss, finite-gradient,
positive-reward, and multi-mode check. Its sole failure was
`submission_contract`: the r2 launcher deliberately wrote fresh schema
`pantry-support-mask-drgrpo-smoke-submission-r2`, while the shared base audit
still hard-coded the earlier `...-submission-v1` name. All hashes and values
inside that submission record matched.

R2-r1 is audit-only. It permits the shared audit to accept an explicitly
configured exact schema set and configures the r2 audit to accept only
`pantry-support-mask-drgrpo-smoke-submission-r2`. It re-audits the immutable r2
identity, submission, manifest, scheduler record, stdout, metrics, draws,
checkpoints, data, source snapshot, and execution snapshot. It may not rerun
training, alter a training artifact, weaken any scientific or telemetry check,
or reinterpret any check other than the literal submission schema name. The
new receipt must also bind the failed r2 audit and verify that its sole error
was `failed check: submission_contract`.
