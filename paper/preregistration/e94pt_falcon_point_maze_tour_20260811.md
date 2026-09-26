# E94-PT: PointMaze Tour, verified replay versus matched Dr.GRPO, Falcon3-1B

Registered 2026-08-11, while the cohort stood at 191 of 30,720 optimizer steps
(0.05 of 8 passes) and before any E94-PT evaluation had been read.

## What this is

The second model family on the **identical** release used by E93-PT
(`var/data/point_maze_tour_v1r1`, 4 landmarks, mean 8.33 execution-certified
tours per map). Model, tokenizer, and prompt surface are the only differences:
`Falcon3-1B-Instruct`, untouched, no warm start, `falcon3` role markers. Data,
budget, arms, dose, seeds-per-cell, pass count, and evaluation cadence are
unchanged, so this is a family replication rather than a second design.

Not pooled with E93-PT. Reported as its own stratum.

## Registered deviation: stage-1 breadth is below the admission floor

Stage 1 on 64 development maps, untouched Falcon3-1B:

| gate | value | threshold | |
|---|---|---|---|
| `pass@8` | .953 | >= .70 | pass |
| `mean@8` | .334 | .20-.65 | pass |
| `distinct@8` | **2.125** | >= 2.5 | **fail** |
| maps with >= 2 modes | .688 | >= .60 | pass |

`distinct@8` misses the floor by .375. The cohort is run anyway, and the reason
is recorded here rather than argued afterwards.

The floor exists so that `distinct@8` is not ceiling-bound at eight samples and
so that breadth exists to be lost. Neither concern binds here: 8.33 tours are
certified per map against 8 samples, and 69% of development maps already expose
two or more modes at initialization. What the low value indicates is that Falcon
starts with **less** breadth than Qwen2.5-0.5B (2.125 against 3.531), which is
a fact about the model rather than a defect of the instrument, and it is
precisely the quantity the replication is meant to compare across families.

The cost of the deviation is stated plainly: with less initial breadth there is
less for the control to lose, so this stratum has **less headroom than E93-PT
and is expected to resolve a smaller effect, not a larger one**. A null here is
therefore weak evidence against the method, and will be reported as such.

## Design

- 5 seeds (55--59) x 2 arms = 10 cells, 8 passes = 3,072 updates.
- Evaluation on the sealed 128-map split every 192 updates (half-pass grid).
- `control` performs the same bank, schedule, scoring, and backward traversal
  with an exact-zero replay derivative; `replay` applies uniform verified
  likelihood at weight .10, bank capacity 16.
- Learning rate 2e-7, 16 samples per update, `gpu:a6000:1` for every cell; each
  receipt records `device_name`, since Turing emulates bf16 at roughly 4x cost
  and a paired cohort must not straddle architectures.
- Executed as 16 chained one-hour chunks per cell, each resuming from the
  previous checkpoint.

## Primary estimands

Paired seed differences, x-Mode minus matched Dr.GRPO, at pass 8, for
`distinct@8` and `pass@8`. Secondary: trapezoidal AUC over the half-pass grid.
All five paired differences reported with mean and range. No greedy `pass@1`
endpoint: the actions are sequential.

## Stopping and exclusions

No checkpoint is selected; pass 8 is the only primary endpoint. No cell is
replaced on an outcome. A cell lost to node failure resumes from its last
checkpoint. The cohort is final only when all ten cells reach pass 8 and pass
their audit.

## Known context, registered in advance

E93-PT (Qwen2.5-0.5B, same release) returned `distinct@8` +.027 at pass 8 with
range [-.023, +.086] over five seeds, which is not distinguishable from zero.
E94-PT is therefore not expected to be confirmatory of a positive effect, and
is run to establish whether the near-null is family-general on this domain.
