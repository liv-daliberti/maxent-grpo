# E117 Stage-1 A10: execution-estimand closure

Frozen: 2026-08-25T13:26:36-04:00 while all twelve E117-R1 jobs were
pending with zero realized optimizer updates and before any successor job,
response, or outcome existed. PointMaze remains excluded.

Status: pre-outcome schema and disclosure correction. It changes no arm,
endpoint, contrast, seed, draw, checkpoint, threshold, uncertainty formula,
scope, or confirmation rule in the v10 effective contract.

## Strict endpoint representation

The v10 analyzer required finite bounded endpoint values but converted each
value with `float(...)`. JSON booleans and numeric strings could therefore
masquerade as measured endpoints. The analyzer must now accept only native JSON
numbers (`int` or `float`, never Boolean), then apply the existing finite,
range, and `raw distinct >= pass` checks. Missing, Boolean, string, container,
nonfinite, or out-of-range endpoints fail closed before any effect is computed.

## Training-budget versus compute target

The primary development and confirmation estimands are aligned at the fixed
registered optimizer-update grid `0, 192, ..., 3072`. C/P/F share the same
proposal-shaped request contract, nominal resource envelope, and update count.
They are not claimed to consume identical realized FLOPs, tokens, or wall
time: response lengths and post-treatment proposal/replay consumption may
diverge after the policies diverge.

A future execution manifest and terminal audit must therefore retain, by arm,
context, seed, and checkpoint interval where available:

- neutral, fixed-control, proposal, and replay generated row counts;
- charged and realized prompt/response-token counts for those paths;
- optimizer updates, GPU class/count, elapsed GPU seconds, restarts, and node;
- any unavailable realized-token field explicitly as unavailable, never
  imputed from a favorable arm.

These quantities are descriptive cost and efficiency diagnostics. They do not
rescale the registered endpoint, enter an advancement gate, create a pooled
benefit-per-token statistic, or convert `P-C` / `F-P` into fixed-FLOP effects.
If realized compute differs materially, report that alongside the fixed-update
effect and preserve the total-effect interpretation.

## Boundary

The strict type check protects the table interface. The compute disclosure
protects the estimand language. Neither can rescue missing data, failed
identity, mechanism nonactivation, or an unfavorable endpoint. Stage-1 launch
remains unauthorized until the E117 mechanism audit and readiness gate pass.
