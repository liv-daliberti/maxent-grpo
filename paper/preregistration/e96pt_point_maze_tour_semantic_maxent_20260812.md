# E96-PT: semantic MaxEnt on verified replay, PointMaze Tour, Qwen2.5-0.5B

Registered 2026-08-12, before any E96-PT cell was submitted and before any
E96-PT evaluation was read.

## Why this arm exists on this domain

PointMaze was excluded from E81, E82, and E83. The recorded reason was that
distributing an episode-level semantic advantage across a variable number of
decisions creates a length incentive. PointMaze Tour emits exactly `K` scored
decisions per episode, constant inside every prompt group, so the term divides
evenly and cannot reward or punish length. The exclusion no longer applies, and
this is the first semantic arm on an interactive ModeBench domain.

## Objective

E81's objective verbatim, composed on verified replay: `A_sem = .10 * z`, with
`z` the predictor-centered clipped surprisal of the episode's canonical key
under a prompt-local open-set predictor (pseudocount 1, one structural unseen
bucket, surprisal clip 5), applied to validator-positive episodes only and added
**after** Dr.GRPO's own task centering and **before** the per-decision division.
Verified replay stays on at weight .10, capacity 16.

Comparators, both already complete on the identical release: E93-PT `control`
(matched Dr.GRPO, exact-zero replay derivative) and E93-PT `replay`.

## Registered prediction

Verified replay is passive retention: it rehearses banked tours and does not
oppose a gradient that pulls the policy toward cheap ones. Measured, it does not
oppose it -- the share of successful samples choosing the single cheapest
certified tour is .520 for `replay` against .507 for `control` at pass 8, and
mean cost rank is 2.41 against 2.44. Semantic MaxEnt rewards an outcome in
proportion to its surprisal, so it is an active force against concentration.

**If the mechanism works on this domain, rank-1 share at pass 8 falls below the
control's .507 and mean cost rank rises above 2.44. If rank-1 share lands near
.51, the cost attractor defeats both auxiliaries and this domain's near-null is
structural rather than a property of which auxiliary is chosen.**

This is stated before the run because the rank statistic is far more sensitive
than `distinct@8`, whose replay effect here was +.027 against a seed spread of
.044, and a prediction written afterwards would be worth nothing.

## Known alternative if the arm is null

`semantic_advantage_rms` measured .0102 against a task-advantage RMS of .392,
about 2.6%. The coefficient is E81's, inherited unablated. A null result is
therefore consistent with "the dose is too small on this domain" as well as with
"the mechanism does not address a cost attractor", and both readings will be
reported. E96-PT does not search the coefficient.

## Design

- 5 seeds (43--47), one arm, 8 passes = 3,072 updates, `var/data/point_maze_tour_v1r1`.
- Evaluation on the sealed 128-map split every 192 updates.
- `Qwen2.5-0.5B-Instruct`, untouched, no warm start; `gpu:a6000:1` for every cell.
- Executed as 8 chained one-hour chunks per cell, resuming from checkpoints.

## Primary estimands

Paired seed differences at pass 8, semantic minus `replay`, for `distinct@8`
and `pass@8`; semantic minus `control` reports the combined package. Secondary
and pre-specified: rank-1 share and mean cost rank against the certified
cost-ordered tour list, measured by `ops/measure_point_maze_tour_mode_rank.py`.
No pooling with any static domain. No checkpoint selection; pass 8 only.
