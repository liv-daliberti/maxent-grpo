# Maze interactive scorer microbatch-4 repair

**Status: FROZEN AFTER POINTMAZE v1 AUDIT FAILURE AND BEFORE ANY REPLACEMENT
ONLINE JOB — 2026-07-30**

## Observed failure

The immutable PointMaze v1 paired jobs `30201821` and `30201822` both
completed four updates, 64/64 verified episodes, and four multimode prompts.
Their independent audit job `30201823` nevertheless failed, correctly, because
every update violated the frozen behavior/live log-probability bound.  The
largest update-1 discrepancy was `0.23097944259643555`, versus the required
maximum `0.0001`.  PointMaze v1 remains failed and does not authorize paper
jobs.

## Frozen causal replay

Read-only jobs replayed the immutable v1 update-1 state/action ledger against
the independently hash-matched initial checkpoint
`26e50c563b180f657aae830325ac46532c1e205ebdfde0f8b6d68bf6c318d251`.
The exact grouped replay establishes:

- microbatch 4: maximum discrepancy `0.0`;
- microbatch 16: maximum discrepancy `0.0`;
- microbatch 8: maximum discrepancy `0.23097944259643555`;
- changing train/eval mode or supplying explicit position IDs has no effect;
- the immutable snapshot's exact backward helper reproduces the same
  microbatch-8 discrepancy.

The machine-readable evidence is:

- `var/artifacts/point_maze_interactive_paired_smoke_v1_grouped_logprob_diagnostic.json`;
- `var/artifacts/point_maze_interactive_paired_smoke_v1_exact_backward_diagnostic.json`;
- `var/artifacts/point_maze_interactive_paired_smoke_v1_microbatch8_diagnostic.json`.

Thus the failure is a batch-geometry numerical incompatibility between the
16-row sampling traversal and the submitted 8-row gradient traversal.  It is
not an environment, map, verifier, reward, model-mode, tokenizer-position, or
checkpoint failure.

## Frozen repair

All subsequent PointMaze and AntMaze online paired/final jobs use policy
microbatch 4.  The rollout count, fixed policy slots, replay slots, optimizer
steps, objective, learning rate, action support, simulator horizon, model,
data, and every admission threshold remain unchanged.  Audits require the
recorded microbatch size to be exactly 4 and continue to enforce
behavior/live discrepancy at most `0.0001`.

PointMaze uses a v2 development gate rather than rerunning v1.  It uses fresh
seed `75302` and the previously unused train-row indices `1,3,5,7`, retaining
one map from each frozen family in the order `bar7`, `block9`, `bar9`,
`asymmetric_block9`.  It has the same four-update, 16-rollout, compute-matched
two-arm contract.  Only a passing independent v2 audit may authorize the ten
PointMaze Stage-B cells.

AntMaze paired jobs `30202665` and `30202666` were held before allocation at
runtime `00:00:00`; audit dependency `30202667` also had runtime `00:00:00`.
They were cancelled without an online outcome once the shared microbatch-8
defect was proven.  Their replacement may retain seed `76313` and the frozen
v13 map slate because no AntMaze paired rollout occurred.  The passing v13
warm-start viability result remains an antecedent, not a paper cell.

No failed receipt may be relabelled, overwritten, or deleted.  No threshold,
map, final seed, evaluation row, model checkpoint, or endpoint is selected in
response to these outcomes.
