# E113-R3: effective-gate full DAPO direct-comparator relaunch

**Frozen:** 2026-08-19 16:18 EDT, after both original E113-R1 outcomes and
after submission of the Qwen E113-R1-M1 memory-capacity replacement, but while
M1 remained scheduler-pending at zero updates. No M1 outcome and no E113-R3
training result had been observed.

## Immutable history and purpose

The original E113 record remains a failed launch: both Countdown smokes exited
before 32 accepted updates and all 50 dependent science placeholders became
`DependencyNeverSatisfied`. E113-R1 then tested Graph Coloring without any
science cells. Falcon completed 32 unique optimizer updates and wrote its
receipt before a process-teardown allocator abort; Qwen completed six updates
before a valid 16-response batch exhausted a 24 GB A5000. E113-R2 is closed by
its own frozen rule because the original R1 Qwen smoke failed. None of these
operational trajectories is an efficacy datum.

E113-R3 is a fresh, prospective scientific successor. Its estimand remains
DAPO minus the already completed matched plain Dr.GRPO control at the same
model family, domain, seed, data, decoding, and horizon.

## Fail-closed effective gate

No E113-R3 job may be submitted unless the effective-gate auditor passes both:

1. Falcon job 30790112 has receipt outer step 33, exactly policy/global steps
   1 through 32, one duplicate terminal summary, all registered DAPO telemetry,
   at most 5,120 queries, the frozen post-receipt log ordering, and the observed
   `FAILED/137:0` teardown classification; and
2. the fresh Qwen E113-R1-M1 A6000 smoke has the same exact 32-update telemetry
   contract and exits `COMPLETED/0:0`.

Receipt outer step 33 is the runner's target-plus-one summary convention, not a
33rd optimizer update. There is no force flag, partial-family release, or use
of the failed original Qwen trajectory.

## Frozen scientific matrix

- Models: Qwen2.5-0.5B-Instruct and Falcon3-1B-Instruct.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, PantryPlan.
- Seeds: five matched E78 seeds for Qwen and five matched E79 seeds for Falcon.
- Cells: `2 models * 5 domains * 5 seeds = 50`. A 25-cell launch would cover
  only one model family and would not answer the requested cross-model
  comparative.
- Horizon: 384 training prompts for eight passes, or 3,072 accepted optimizer
  updates per cell.
- Sampling: 16 responses per prompt, temperature 1, top-p 1.
- DAPO: GRPO critic, dynamic rejection of constant-reward groups, asymmetric
  clips `.20/.28`, token-level loss aggregation, overlong buffer ratio `.20`,
  overlong penalty factor `1.0`, and at most ten generation batches per
  accepted update.
- Query ceiling: `3,072 * 16 * 10 = 491,520` sampled responses per cell.
- Replay, MaxEnt, UCPO, RLEP, collision, DIAYN, discovery tracking, and entropy
  interventions remain disabled exactly as in E113.
- Runtime: the immutable E113 source/ops snapshot recorded in the original
  ledger.

Each cell receives a new E113-R3 job ID and run directory. The 25 Qwen cells use
one 48 GB A6000 from `node[103-104,205-208,805]`, with fused Adam and activation
offload both unchanged and disabled. This is the capacity-only repair validated
by M1; complete 16-response optimizer batches are retained. The 25 Falcon cells
retain their frozen E79 placement and objective surface.

## Submission and failure semantics

The launcher first submits all 50 jobs held, audits every scheduler-expanded
environment and placement, writes one atomic unreleased ledger, releases every
job, and then marks the ledger released. Any pre-release error cancels the held
set. No dependency placeholders are created.

The ten-generation-batch limit is the scientific DAPO feasibility rule. The
experiment wrapper may not requeue a learner after DAPO exhaustion; such a cell
fails once. Scheduler/node requeue and checkpoint resume remain available under
cluster policy. No failed cell may receive a higher sampling cap, different
temperature, replacement seed, warm start, or method change under E113-R3.

## Reporting

Always report the full denominator of 50. Endpoint effects use only terminal
E113-R3 cells paired with their frozen E78/E79 controls and must state exact
per-domain and per-family `n`. Failed or incomplete cells remain visible. No
pooled efficacy claim is licensed from an incomplete surface; operational
feasibility is separate from endpoint efficacy.
