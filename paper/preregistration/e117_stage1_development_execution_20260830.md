# E117 Stage 1 development execution freeze

Frozen: 2026-08-30 before submission of any Stage 1 job and before inspection
of any Stage 1 response or endpoint.

Status: user-authorized execution amendment to the analysis-only E117
successor effective contract v11. This amendment changes no endpoint,
threshold, uncertainty rule, causal contrast, scope rule, or confirmation rule.
PointMaze remains excluded. The sealed confirmation reserve must not be read by
the launcher, training jobs, terminal audit, or development table builder.

## Authorization and immutable parents

Launch is authorized only because the official E117-R2 replacement audit
`var/artifacts/e117r2_same_plumbing_component_preflight_audit.json` passed with
an empty failure list and `stage1_execution_readiness.ready=true`. The execution
ledger binds that exact artifact and its scheduler job 30978841 by SHA-256.

The launcher binds by SHA-256:

- `var/artifacts/e117_successor_effective_contract_v11.json` and its effective
  contract, analyzer, analyzer tests, reserve identity/tests, and A10 evidence;
- the official E117-R2 ledger and passing audit;
- the base R2 runtime snapshot and a derived Stage 1 snapshot identity;
- this execution freeze, launcher, terminal auditor, and table builder.

## Exact development execution

Run C/P/F on the four v11 sentinels and seeds 201, 202, and 203: 36 jobs,
384 historical training rows, eight passes, and exactly 3,072 optimizer updates
per job. Evaluate K=8 with draw labels 0 through 15 at checkpoints
0,192,...,3072. Draw `d` uses row seed `117900+d`; every prompt records the
single fixed request seed that generates its eight completions. The same seed
projection is used across training seeds, arms, and checkpoints.

The only evaluation paths are
`var/data/e117_evaluation_reserve_v1/development/<domain>/eval`. Each
context/seed C/P/F block uses the same exact A5000 node, one GPU, eight CPUs,
64 GiB RAM, the same frozen source and scientific export, except for output
identity and the two registered treatment factors. Node blocks are balanced:

| Context | seed 201 | seed 202 | seed 203 |
|---|---|---|---|
| qwen05b/countdown | node202 | node203 | node202 |
| qwen05b/graph_coloring | node203 | node202 | node203 |
| qwen05b/python_factors | node202 | node203 | node202 |
| falcon1b/mathir | node203 | node202 | node203 |

Actual start order is enforced with scheduler `after` dependencies: seed 201
C-P-F, seed 202 P-F-C, and seed 203 F-C-P. This constrains start order without
serializing complete training runs. All jobs are submitted under user hold,
their exact scheduler records and export hashes are audited, the complete
ledger and terminal `afterany` audit are installed, and only then are all jobs
released.

## Narrow runtime clarification

The R2 source wrote the correct fixed-draw seed into the sampler but omitted it
from the response-free sidecar projection. The derived Stage 1 snapshot changes
only `src/oat_drgrpo/actor.py` to retain that already-used seed as
`request_seeds_by_prompt`; sampling, responses, rewards, and training are
unchanged. The launcher records the parent/derived byte diff and refuses any
unlisted runtime change.

The v11 prose inherited an A3 phrase that included `answer_keys` in a
"response-free" prompt digest. Answer keys are response-derived. The frozen
builder therefore retains the already-tested E105 response-free projection:
prompt identity is `answer_mode_count`, `option_ids`, `prompt`, `prompt_index`,
and `reference`; request identity is row seed plus `option_ids`, `prompt_index`,
and `request_seeds_by_option`. Metrics, responses, rewards, and answer keys are
excluded from both digests. This closes the contradiction without changing an
estimand or gate.

## Storage, restart, audit, and outcomes

Rolling resume checkpoints are written every 64 updates, at most two are
retained, successful jobs prune them, and automatic resume/watchdog requeue is
enabled. Registered evaluation checkpoints remain every 192 updates. Terminal
model export is disabled (`OAT_ZERO_EXPORT_STEPS=-1`) solely to avoid 36
unneeded model copies on a 97%-used filesystem. No endpoint uses exported model
weights.

The terminal audit must fail closed on scheduler completion, exact job/export/
node/dependency identity, all 3,072 canonical optimizer rows plus one validated
terminal bookkeeping sentinel, C/P/F mechanism invariants, exact 17 by 16
sampled receipt grids, response-free paired identities, and step-zero equality.
It reports per-arm generated rows, charged and realized prompt/response tokens
for neutral/control/proposal/replay paths where recorded, explicit
`unavailable` otherwise, optimizer updates, A5000 count and seconds, node, and
restarts. Cost remains descriptive and never enters a scientific gate.

Only after non-outcome integrity checks pass may the audit materialize the
development table and invoke the frozen v11 analyzer. The resulting Stage 1
decision may authorize only the already-locked confirmation design; it never
authorizes reading confirmation data during development.
