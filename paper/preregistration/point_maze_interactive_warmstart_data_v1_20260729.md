# PointMaze interactive train-only warm-start data v1

**Status: FROZEN BEFORE DATA MATERIALIZATION — 2026-07-29**

## Antecedent and purpose

The closed-loop untouched 0.5B gate produced 4/256 verified routes, all on one
of four development maps and all with the same canonical identity. It therefore
failed its frozen breadth threshold and advances to the registered train-only
warm-start rung. The endpoint failure and the later partial success remain
final development records.

This stage materializes supervised state-to-action examples. It does not load,
sample, or update a language model.

## Frozen source

- load exactly `point_maze_modebench_v1/train`, containing eight maps and no
  split overlap;
- select exactly the two certified route programs for each record whose
  certification split is `train`;
- replay all 16 programs through the same persistent, networkless PointMaze
  stepwise worker used by the closed-loop gate;
- before every action, render the exact public online-policy prompt and record
  its single option-label target; and
- require each replay to reproduce its frozen executable canonical key.

Any missing route, non-train map, worker failure, early/late termination,
canonical-key mismatch, or duplicate map ID fails materialization.

## Information firewall

The dev and evaluation dataset directories may not be loaded. The dataset
identity catalog is used only to select records explicitly labeled `train`;
no dev/eval problem text or prompt enters the output. The output records train
map IDs, episode keys, source hashes, and explicit `model_sampled=false`.

Passing authorizes only a separately frozen shared SFT design. The eventual
checkpoint, if admitted, must initialize both verified-MaxEnt and Dr.GRPO
arms identically and must be absent from untouched-base comparisons.
