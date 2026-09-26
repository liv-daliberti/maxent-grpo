# Ant heading controller v4

Status: frozen before v4 evaluation  
Frozen: 2026-07-29

## Prospective separation

The stopped eight-environment v1 cohort and controller trials v1--v3 remain
failed. Version 4 is a separately versioned controller-admission study.

Trial v1 established that its frozen weights met every navigation criterion
but failed the optional STOP command. Trial v3 established that retraining the
same navigation objective from scratch is not reliably direction-balanced.
Version 4 therefore freezes the exact v1 model bytes as an eight-heading
navigation controller and removes STOP from the language-action alphabet.
There is no weight update, checkpoint selection, ensemble, heading-specific
model choice, or threshold change.

## Frozen identity

- model:
  `var/maze_runtime/controllers/ant_heading_v1.zip`;
- model SHA-256:
  `526b669bb14cf8a07b44a1c725f94411632a3ac8a8e84966db09336cc96e4b0f`;
- observation: native position-excluded Ant-v5 observation followed by the
  two-dimensional unit heading;
- actions: N, NE, E, SE, S, SW, W, and NW only;
- deterministic policy prediction;
- episode horizon: 300 simulator steps;
- reset noise scale: 0.1; and
- runtime: the existing hash-pinned maze Python environment.

## New-seed admission

The earlier v1 evaluation seeds are forbidden. Evaluate 12 deterministic
episodes per heading using:

`2073004 + 1000 * heading_index + replicate`, for replicate 0 through 11.

Admit v4 only if:

- each heading's mean projected displacement is at least 2.0;
- overall mean projected displacement is at least 4.0;
- unhealthy early termination across all 96 episodes is at most 0.10;
- the loaded model SHA-256 is the frozen value above; and
- every episode and aggregate metric is present and finite.

No rerun, seed replacement, command removal, coefficient change, or checkpoint
substitution is allowed after inspection. Failure makes v4 ineligible.

## Information firewall and next gate

Evaluation uses open-plane Ant-v5 only. It may not load a maze map, route gate,
language completion, MaxEnt result, or Dr.GRPO result. Passing admits the
controller only to a separately frozen AntMaze route-identity, perturbation,
and throughput gate. It does not authorize 0.5B sampling or training.

