# Ant heading controller v5

Status: frozen before v5 evaluation  
Frozen: 2026-07-29

## Prospective separation

Trials v1--v4 remain failed. V4's 96-episode development receipt showed that
the frozen v1 weights retained strong aggregate navigation but systematically
mapped the requested east command toward east-northeast. Both its NE and SE
primitives independently had positive east displacement.

V5 freezes a deterministic macro wrapper around the exact v1 model bytes.
Seven desired headings retain their one-command primitive. Desired east is a
fixed 300-step macro: NE for steps 1--150 and SE for steps 151--300. There is
no model training, checkpoint choice, runtime feedback, adaptive switching, or
maze-dependent control.

## Frozen identity

- model:
  `var/maze_runtime/controllers/ant_heading_v1.zip`;
- model SHA-256:
  `526b669bb14cf8a07b44a1c725f94411632a3ac8a8e84966db09336cc96e4b0f`;
- desired headings: N, NE, E, SE, S, SW, W, NW;
- controller macros by desired-heading index:
  `[[N], [NE], [NE, SE], [SE], [S], [SW], [W], [NW]]`;
- each desired-heading episode is exactly 300 simulator steps unless unhealthy
  termination occurs;
- multi-command macro segments divide the horizon equally;
- model prediction is deterministic; and
- STOP is absent.

## New-seed admission

The v1--v4 evaluation seeds are forbidden. Evaluate 12 episodes per desired
heading using:

`3073005 + 1000 * heading_index + replicate`, for replicate 0 through 11.

Admit v5 only if:

- each desired heading's mean projected displacement is at least 2.0;
- overall mean projected displacement is at least 4.0;
- unhealthy early termination across all 96 episodes is at most 0.10;
- the model and macro identity match this preregistration; and
- every episode and metric is present and finite.

No rerun, segment-order swap, duration change, extra macro, seed replacement,
or threshold change is allowed after inspection. Failure leaves AntMaze
ineligible.

## Information firewall and next gate

Evaluation uses open-plane Ant-v5 only. Passing admits the exact model-plus-
macro controller only to a new AntMaze route-identity, perturbation, and
throughput gate. It does not authorize language-model sampling or training.

