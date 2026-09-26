# E112-R1 sampler-contract repair and original E112 retirement

Frozen: 2026-08-19T17:14:32-04:00, before canceling any active original
E112 job, changing the launcher, or inspecting any task-evaluation endpoint.

## Observed launch defect

The released original E112 ledger has SHA-256
`ef70759f73788f618a53c8bd513cbb89e4b8be32c293717d26ebe0fc21f46fd7`.
Eleven non-Pantry jobs reached the launcher validation path before any
optimizer update, exhausted six same-ID watchdog restarts, and terminated with
the exact error:

`ValueError: counterfactual canonical proposals require replicated free-form sampling or the fixed-shape Pantry learner sampler`

The held scheduler records prove that all 60 non-Pantry cells omitted
`OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1`. All 15 Pantry cells retained their
registered fixed-shape learner sampler. At this freeze the ledger contained 61
pending, 3 running, and 11 failed jobs. The failed jobs had zero accepted
optimizer updates. The only realized training updates were in correctly
configured Pantry cells. No efficacy endpoint was inspected.

This is a launcher-contract defect, not a treatment result and not a
low-priority preemption. The original E112 cohort is permanently retired as an
efficacy cohort. Preserve its ledger, logs, and partial run artifacts; never
pool them with the replacement.

## Exact repair

E112-R1 must retain the frozen E112 science matrix, optimizer, objective,
coefficients, data, seeds, hardware pairing, horizon, and evaluation schedule.
Only the missing E111 sampler interface is restored:

- non-Pantry cells set replicated free-form sampling and local actor weight
  synchronization to one;
- Pantry cells set both values to zero and retain fixed-shape canonical learner
  sampling; and
- held-job audit and unit contracts fail closed on these domain-specific
  values.

E112-R1 uses fresh run directories, run stamps, job names, and a separate
75-cell ledger. It may be released only from a newly frozen source/ops snapshot
with passing production tests and a passing 75-cell dry run. No original E112
job may remain active when E112-R1 is submitted.

## Scheduler retirement

Cancel only active job IDs bound by the exact original E112 ledger. Existing
failed jobs require no scheduler action. Record the exact before/after state
and cancellation result in
`var/artifacts/e112_sampler_contract_failure_retirement.json` before releasing
E112-R1.
