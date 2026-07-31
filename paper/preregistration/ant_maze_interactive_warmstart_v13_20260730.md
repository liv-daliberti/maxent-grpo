# AntMaze v13 constrained 0.5B warm start and development viability

**Status: FROZEN AFTER V13-R1 TRAIN DATA PASSED AND BEFORE SFT OR ANY V13 DEVELOPMENT SAMPLE — 2026-07-30**

## Immutable antecedents

AntMaze v12 free-form viability remains failed at 0/256. The v12 route and
three-node gates remain passed, and v13-r1 reproduced both certified route
modes on all four training maps under the exact single-threaded v12
controller, yielding 32 public state/action examples from eight routes. No
development or evaluation row was loaded and no language model was sampled
during materialization.

## Separate language-action interface

V13 is explicitly reported as a constrained, closed-loop language policy.
At each high-level decision, Qwen2.5-0.5B receives the public map, current Ant
position, planar velocity, goal, and remaining action horizon. The model's
next-token logits are restricted to the eight one-token compass actions
`N NE E SE S SW W NW`; the model still selects the action. The trusted worker
then passes that adjacent-grid target to the unchanged frozen v11 low-level
controller using the unchanged v12 cumulative targeting rule. No route,
planner output, canonical key, reward, answer, future state, or verifier
feedback enters a nonterminal prompt.

## Frozen train-only SFT

Initialize from the pinned Qwen2.5-0.5B-Instruct snapshot. Train on exactly
the 32 v13-r1 examples for 64 epochs, batch size 4, gradient accumulation 4,
128 optimizer updates, AdamW learning rate 2e-5, weight decay 0.01, warmup 8,
linear decay, max norm 1, BF16, and seed 75313. Cross-entropy is normalized
only over the eight action-token logits. The resulting checkpoint is shared
unchanged by both later online arms; it is a task-specific 0.5B initialization
stratum and is never pooled invisibly with base-checkpoint rows.

## Frozen development gate

After SFT, load only the unchanged four-row v12 `dev/multi_answer` split.
For each map sample 64 closed-loop trajectories at temperature 1/top-p 1;
the first 16 are the frozen training-group viability prefix. Request seeds
are deterministic from base seed 107313, map index, trajectory index, and
decision round. The worker is single-threaded and networkless, and every
positive is revalidated through the exact v12 topology/controller verifier.

The gate passes only if at least two of four maps have a verified route in
their first 16 trajectories and at least one map exposes both canonical route
keys among all 64. All 256 trajectories must terminate with complete request,
action, simulator, identity, and validation records; any worker, mask,
nonfinite, source, controller, data, or evaluation-firewall violation fails.
A pass authorizes one separately frozen paired online mechanism smoke, not
the final seeds 43--47. A failure stops AntMaze Stage B. No setting changes
after this outcome.
