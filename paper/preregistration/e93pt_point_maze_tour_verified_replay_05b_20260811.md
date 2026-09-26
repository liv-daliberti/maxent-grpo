# E93-PT: PointMaze Tour, verified replay versus matched Dr.GRPO, Qwen2.5-0.5B

Registered 2026-08-11, before any E93-PT cell was submitted.

## What replaces what

PointMaze v1 (E78-PM/E79-PM) is retained as evidence but is inert: its control
moves `distinct@8` by +.016 over 3,072 updates against a mean within-run
checkpoint SD of .048, so it cannot resolve a treatment effect in either
direction. PointMaze Tour is the redesigned sixth domain. E93-PT does not pool
with, supersede, or re-estimate any v1 cell.

## Admission, and an explicit deviation

The registered stage-2 gate required the matched control to lose at least .50
`distinct@8` on development maps. **The four-landmark release does not meet that
threshold**; at four passes it gave a mean drop of +.438, and the eight-pass
endpoint is reported in the campaign log whatever it shows.

The domain is admitted anyway, on the phenomenon rather than the threshold, and
this is a deviation recorded in advance rather than a reinterpretation after the
fact. The reasons:

1. The phenomenon the benchmark exists to measure is present and unambiguous.
   Accuracy rises while breadth falls, in both seeds: `mean@8` .65 -> .86 while
   `distinct@8` 3.53 -> 3.20 and 3.56 -> 3.11, with modes-per-success falling
   .69 -> .46. Correct samples are becoming redundant copies.
2. The two admission gates are in tension by construction. Stage 1 caps `mean@8`
   at .65 so advantages do not vanish; stage 2 requires a control that has
   saturated enough to shed breadth. A five-landmark release that clears the
   stage-1 ceiling (`mean@8` .381) instead *gains* breadth, because the policy
   spends the window learning to succeed at all. The .50 threshold was
   calibrated against the wrong quantity.
3. `mean@8` = .652 exceeds the stage-1 ceiling by .002. Over 99.8% of
   sixteen-sample groups still contain both a success and a failure, so the
   advantage the ceiling protects is intact.

No result of E93-PT was seen before this decision: the gate ran the control arm
only, on development maps, and the evaluation split is sealed.

## Design

- Data `var/data/point_maze_tour_v1r1`: 384 train, 64 dev, 128 eval maps, four
  landmarks, mean 8.33 execution-certified tours per map. Fingerprints are
  deduplicated across splits, so evaluation maps are disjoint by construction.
- Model: `Qwen2.5-0.5B-Instruct`, untouched. **No warm start**, no
  demonstrations. This retires the warm-start caveat that applies to v1.
- Arms: `control` performs the same bank, schedule, scoring, and backward
  traversal with an exact-zero replay derivative; `replay` applies uniform
  verified likelihood at weight .10, bank capacity 16.
- 5 seeds (43-47) x 2 arms = 10 cells, 8 passes = 3,072 updates, evaluation on
  the sealed 128-map split every 192 updates (the half-pass grid used by the
  static domains), learning rate 2e-7, 16 samples per update.
- Every cell runs on `gpu:a6000:1`. Turing reports bf16 support and emulates it
  at roughly 4x cost; a paired cohort must not straddle architectures, and each
  receipt records `device_name`.

## Primary estimands

Paired seed differences, x-Mode minus matched Dr.GRPO, at pass 8, for
`distinct@8` and `pass@8`. Secondary: trapezoidal AUC over the half-pass grid.
All five paired differences are reported with their mean and range. PointMaze
Tour is not pooled with any static domain and has no greedy `pass@1` endpoint,
its actions being sequential.

## Stopping and exclusions

No checkpoint is selected; pass 8 is the only primary endpoint. No cell is
replaced on an outcome. A cell lost to node failure resumes from its last
checkpoint. The cohort becomes final only when all ten cells reach pass 8 and
pass their audit; until then every value is explicitly interim.
