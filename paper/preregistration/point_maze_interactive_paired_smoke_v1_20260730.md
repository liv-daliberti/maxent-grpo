# PointMaze interactive paired online-training smoke v1

**Status: FROZEN AFTER THE V3 VIABILITY PASS AND BEFORE IMPLEMENTATION OR SAMPLING — 2026-07-30**

## Antecedent and authorization boundary

The train-only velocity-state v3 warm start completed as Slurm job `30198474`.
Its untouched development gate passed with 32/256 verified trajectories,
prefix successes on three of four prompts, and multiple verified route keys on
three of four prompts. The immutable receipt is
`var/artifacts/point_maze_interactive_05b_viability_warmstart_v3.json`.

This protocol authorizes exactly two development-only jobs. It is not one of
the final seeds 43--47 and cannot become a paper result. A passing independent
audit authorizes the ten PointMaze Stage-B jobs; it does not authorize any
other environment.

## Frozen pair

- arms: `grpo_compute_matched` and
  `verified_first_global_replay_canonical`;
- common model: the exact completed directory
  `var/models/point_maze_interactive_warmstart_v3`, loaded independently by
  both arms and verified byte-for-byte before the first rollout;
- seed: `75301` for both arms, with arm-separated deterministic request-seed
  namespaces and no seed substitution;
- training rows: indices `0,2,4,6` of the frozen PointMaze v1 train split,
  exactly one preselected row from each of `bar7`, `block9`, `bar9`, and
  `asymmetric_block9`, processed in that order;
- one pass, 16 rollouts per row, four optimizer updates per arm;
- policy interface: the passed v3 public Markov state
  `(maze, position_xy, velocity_xy, goal_xy)` and the fixed nine-label action
  mask; temperature 1, top-p 1, maximum 96 force decisions, action repeat 5;
- optimizer: AdamW, learning rate `2e-7`, betas `(0.9,0.999)`, epsilon `1e-8`,
  weight decay 0, one PPO epoch, clip epsilon 0.2, gradient norm cap 1;
- task advantage: terminal binary reward centered within each 16-rollout
  prompt group without variance normalization;
- loss: selected action tokens only, averaged within episode and then across
  episodes. Environment text, state text, padding, and compute-only rows have
  exactly zero policy loss.

Every rollout round executes a fixed 16-by-96 policy-forward budget. Episodes
that terminate early use public terminal padding prompts and zero decision
masks for the remaining rounds; no additional environment action, verifier
call, canonical key, reward, or bank update occurs. Replay uses one fixed
16-mode-slot budget per update, with zero-masked public padding rows for absent
slots. These pads make policy and replay forward/backward traversal counts
identical across arms without changing either scientific objective.

## Frozen treatment

The treatment is the interactive form of the exact E70 Stage-A method:

- success-conditioned signed semantic Shannon coefficient 0.10, surprisal
  clip 5, pseudocount 1, symmetric advantage cap 0.05;
- open-set inverse coefficient adaptation, warmup 64 eligible observations,
  EMA decay 0.90, no projection and no gold support target;
- online canonical bank direct entropy alpha 0, novelty beta 0.50,
  pseudocount 1, surprisal clip 5;
- retained verified-exemplar capacity 16;
- one persistent-hash round-robin global replay prompt per update;
- split verified-mass and known-mode-balance replay, both base coefficient
  0.10, warmup 64, EMA decay 0.90, no projection; and
- reward-estimator factor 15/16 and per-rollout replay factor 1/16.

Semantic and novelty advantages are detached and added only after task-reward
centering. Only active verifier-positive episodes with non-null unchanged
PointMaze route keys enter either history. Replay mode score is the mean current
restricted log probability of the selected actions across active decisions in
the retained state/action episode; terminal padding never contributes.

## Frozen control and compute match

The control uses plain Dr.GRPO. It performs the same canonicalization, passive
bank updates, retained-episode selection, fixed replay scoring, controller
observations, and replay backward traversal, but semantic, novelty, mass, and
balance derivatives applied to model parameters are exactly zero. Both arms
consume the same four prompts, 64 environment rollouts, 6,144 fixed policy
decision slots, four optimizer steps, four replay group slots, and 64 replay
mode slots. Simulator/verifier call counts may be lower than the fixed policy
slot budget only through real early termination and are reported separately.

## Information boundary

The model never receives certified route programs, directed gate definitions,
canonical keys, reward, distance-to-goal shaping, collision shaping, planner
output, development rows, or evaluation rows. Canonical keys and reward are
used only after terminal execution by the CPU-side objective. The shared warm
start contains only the already-audited train-split behavioral cloning data.

## Pass criteria

The independent pair audit passes only if:

1. both jobs complete the exact four updates from the same initial checkpoint;
2. each arm has at least one verified rollout and at least one training prompt
   with two verifier-distinct route keys;
3. every selected action belongs to the recorded nine-label support and every
   replayed public state/action episode reproduces its stored transition hash;
4. all task, semantic, novelty, PPO-ratio, KL, replay-mass, replay-balance,
   controller, loss, and gradient diagnostics are finite;
5. treatment has at least one nonzero verified exploration advantage and, when
   a replay group is eligible, a nonzero applied replay gradient;
6. control has exact-zero applied semantic, novelty, mass, and balance
   gradients while retaining nonzero raw replay telemetry whenever treatment
   eligibility has a control-side analogue;
7. prompt, rollout, fixed policy-slot, replay-slot, forward/backward traversal,
   and optimizer-step counts match exactly across arms; and
8. source, execution, model, split, admission, warm-start, receipt, metrics,
   state-replay, scheduler, and audit hashes all match their frozen identities.

Failure stops PointMaze before the final five-seed cohort. No threshold change,
map replacement, seed replacement, extra pass, or post-outcome warm-start edit
is permitted.
