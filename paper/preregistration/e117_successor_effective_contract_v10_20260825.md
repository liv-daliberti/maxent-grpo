# E117 successor effective contract v10

Consolidated: 2026-08-25T12:35:32-04:00 while all twelve E117-R1 jobs
remained pending, with zero realized optimizer updates and before any successor
seed, job, ledger, sampled response, or outcome existed.

Status: analysis-only development and conditional-confirmation contract. It
consolidates the original Stage-1 contract and A1--A9. It does not authorize a
launch. A passing E117 mechanism audit and activation-readiness gate remain
mandatory. PointMaze is excluded.

## Causal design

Use C/P/F on exactly four registered model--domain contexts:

- `qwen05b/countdown`;
- `qwen05b/graph_coloring`;
- `qwen05b/python_factors`;
- `falcon1b/mathir`.

`C` pays proposal-shaped compute, validates and canonicalizes the proposal,
then discards it before mutation; semantic coefficient is zero. `P` is C with
validator-positive novel admission and uniform replay enabled; semantic
coefficient remains zero. `F` is P with v7 semantic PPO at fixed coefficient
0.10.

The causal contrasts are total effects:

- `P-C`: enabling proposal admission and replay;
- `F-P`: enabling semantic v7 on the proposal/replay system.

Post-treatment proposal consumption, support, retention, and replay are not
held fixed. `F-P` is not a controlled direct semantic effect. If it advances,
the next decomposition requires semantic-without-proposal.

## Exact development grid

Use training seeds `201, 202, 203`, draw labels `0, ..., 15`, K=8, and
checkpoints `0, 192, ..., 3072`. Run all arms on all contexts. Actual start
orders are 201 `C-P-F`, 202 `P-F-C`, and 203 `F-C-P`. The seeds were absent
from all registered job manifests at freeze time.

Within a context/seed block, C/P/F use the same physical node class, resource
envelope, frozen source, data, and runtime plumbing. A future execution
manifest must bind exact request-seed projections, node blocks, output paths,
exports, and audit machinery before submission.

Retain historical training banks. The only permitted development evaluation
paths are the four `multi_answer` splits below
`var/data/e117_evaluation_reserve_v1/development`. Each contains 128 prompts
disjoint from every historical train/evaluation identity. Bind
`var/data/e117_evaluation_reserve_v1/identity.json`, SHA-256
`a5b9eb4289cca8f78d90249fabbd85343c1c3eb3b5c3df6a2235bbd124446289`.

## Fail-closed identity

Require the exact context/seed/arm/checkpoint/draw grid, strict identifier
types, finite bounded endpoints, no duplicate/extra/missing row, and exact
K=8. Every row carries lowercase SHA-256 prompt and request surface digests.
Prompt projection is exact within context across all rows. Request projection
is exact within context/draw across seeds, arms, and checkpoints; all 16 draw
surfaces are distinct. Metrics and responses are excluded from both digests.

At step zero, `(pass@8, raw distinct correct modes@8)` must agree exactly
across every training seed and C/P/F arm within context/draw. This audits the
shared pretrained state as well as arm pairing.

## Endpoints, uncertainty, and scope

The primitive endpoint vector is `(pass@8, raw distinct correct modes@8)`.
Adjusted breadth is the reported decomposition `raw distinct@8 - pass@8`; it
cannot independently veto simultaneous primitive gains.

Compute `P-C` and `F-P` inside every seed/common draw before averaging. Report
terminal effects and normalized trapezoidal AUC across the complete horizon.
Retain every paired effect and report separately:

- training-seed SE over draw-averaged paired seed effects;
- evaluation Monte Carlo SE over seed-averaged common-draw effects;
- per-seed evaluation Monte Carlo SE.

Do not pool axes into one interval, sample size, or p-value. The prompt
estimand is conditional on the registered finite bank; prompt-population SE is
not estimated. The independent confirmation bank is the locked prompt-block
replication.

Model family and domain are not factorially crossed. Evidence spanning both
family labels is replicated multi-context evidence, not an identified
model-family main effect, domain main effect, or interaction.

## Development gate

Within a context, both terminal and normalized-AUC raw-distinct effects must:

1. exceed +0.05;
2. exceed two evaluation Monte Carlo SEs;
3. be positive in at least two of three paired training seeds.

At terminal and normalized AUC, pass@8 must be at least -0.03 on the paired
seed mean and at least -0.10 in every paired seed, both versus the component
denominator and versus C.

A broad candidate needs the same component actionable in at least three of
four contexts and contexts carrying both model-family labels. Exactly
Countdown alone is a Countdown-specific candidate; Graph remains its reported
negative boundary. Every other pattern is `do_not_advance`.

## Conditional confirmation, locked before selection

If any component advances, run the full four-context C/P/F grid on the sealed
`confirmation` paths. Use seeds `301, ..., 306`, the same 16 draws and 17
checkpoints, and two cyclic-order replicates: 301/304 `C-P-F`, 302/305
`P-F-C`, 303/306 `F-C-P`. These seeds were also absent from every registered
job manifest at freeze time.

Retain every development condition. At terminal and normalized AUC,
additionally require raw distinct to exceed two training-seed SEs and be
positive in at least five of six seeds. A broad selection must reproduce the
broad rule. A Countdown-specific selection must reproduce on Countdown. All
contexts remain mandatory and reported, and confirmation cannot upgrade the
scope selected in development. If both components advance, evaluate both by
the same rule.

These are deterministic replication gates, not p-values. Confirmation may use
the selected component names and scopes, but no development endpoint value may
enter confirmation estimates, uncertainty, or gates.

## Decision

- `P-C` confirms while `F-P` does not: retain proposal/replay; retire semantic
  PPO.
- `F-P` confirms: add semantic-without-proposal before direct attribution.
- Mechanisms activate but neither component advances: improve conservative
  retention/replay; do not increase eta or proposal attempts.
- E117 audit or activation fails: repair plumbing; do not launch efficacy.

The executable analyzer is `ops/exp_scaling/e117_stage1_statistics.py`, with
schemas `e117_stage1_paired_vector_statistics_v9` and
`e117_confirmation_paired_vector_statistics_v1`. A future launcher and table
builder must bind the v10 manifest, analyzer/tests, reserve identity/tests, and
latest A9 evidence by SHA-256.
