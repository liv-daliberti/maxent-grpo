# PointMaze full-scale-v7 384/128 online comparison

**Status: FROZEN BEFORE THE BALANCED-V6 TERMINAL OUTCOME, BEFORE V7 DATA
EXECUTION, AND BEFORE ANY V7 ONLINE JOB — 2026-07-30**

This is the full-scale successor requested after the balanced-v6 online gate
began producing real verified trajectories. Balanced v6 remains immutable and
is reported as the qualification-scale antecedent; it is not silently relabeled
as a 384/128 study.

## Data and information boundary

- Training contains exactly 384 distinct executable PointMaze prompts.
- Development contains exactly 64 distinct prompts and is reserved for
  prelaunch checks.
- Evaluation contains exactly 128 untouched prompts. Evaluation rows, outcomes,
  and certified routes never enter optimization or replay.
- The eight registered geometry strata each contribute 48 train, 8
  development, and 16 evaluation prompts.
- Every split is balanced over all four rotations; no executable task
  fingerprint overlaps another split.
- Every one of the 576 prompts has two independently replayed successful route
  programs with different topology-bound canonical keys. These programs certify
  the benchmark but are absent from model context and online training.

## Frozen comparison

- Shared checkpoint: exact
  `point_maze_interactive_warmstart_v6_balanced_short`.
- Arms: compute-matched Dr.GRPO and verified-first global replay-canonical
  MaxEnt.
- Shared seeds: 76641, 76642, 76643, 76644, and 76645.
- Twelve complete prompt passes: 4,608 optimizer updates per cell.
- Sixteen live interactive trajectories per update, learning rate `2e-7`,
  policy/evaluation microbatch 16, context cap 1536, and decision horizon 96.
- Evaluation occurs at update zero and every 96 updates (one quarter pass),
  for 49 coordinates. Each coordinate evaluates all 128 untouched prompts with
  one greedy trajectory and four deterministic K=8 common-random-number draws,
  totaling 4,224 trajectories per cell and coordinate.

Both arms execute identical live-policy and replay-decision forward slots.
The control computes the same exploration/replay diagnostics but applies
exact-zero auxiliary derivative. Treatment applies the already-qualified
verified-first global replay-canonical objective. Only official terminal reward
and online-discovered canonical identities enter either history.

## Conditional launch and terminal audit

The ten v7 cells may launch only after the immutable balanced-v6 terminal audit
passes with decision `point_maze_balanced_v6_terminal_eligible`. This is an
operational gate, not an efficacy threshold: v6 establishes that the exact
checkpoint, interactive optimizer, simulator, replay path, and compute match
run to completion with real verified outcomes.

The v7 terminal audit requires all ten jobs to complete all 4,608 updates and
49 evaluation coordinates, exact paired compute traversal, finite metrics,
zero support escapes, exact-zero control auxiliary derivatives, applied
treatment derivatives whenever raw signals are nonzero, immutable identities,
and official re-execution of every stored training transition. No efficacy
threshold, seed replacement, early stopping, best-checkpoint selection,
result-dependent extension, or evaluation feedback is permitted.
