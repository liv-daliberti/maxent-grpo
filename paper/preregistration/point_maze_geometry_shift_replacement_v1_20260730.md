# PointMaze-GeometryShift v1 replacement configuration

Status: prospective; frozen before model sampling on any new map  
Date: 2026-07-30

The registered ConstructiveCode capacity ladder terminated after the 0.5B and
1.5B coders each accepted 0/192 development programs. Those outcomes remain
reported and the original ConstructiveCode ten-cell row is not run.

This document defines a separate replacement campaign, not a repair or
completion of the original roster. `PointMaze-GeometryShift` reuses the
already qualified closed-loop 0.5B language-policy interface and pinned
MuJoCo PointMaze runtime, but freezes four obstacle geometries absent from the
current PointMaze family set:

- `wide_block9_shift`: a centered 3-by-5 block;
- `cross9_shift`: a centered five-cell cross;
- `upper_offset9_shift`: a 3-by-4 upper-left block;
- `lower_offset9_shift`: a 3-by-4 lower-right block.

All maps are 9-by-9 with a one-cell wall border, start at row 4 column 1, goal
at row 4 column 7, and prospectively use rotations 0 and 2 for train, rotation
1 for development, and rotation 3 for evaluation. The existing `bar7`,
`block9`, `bar9`, and `asymmetric_block9` maps are not members of this row.

## Admission and identity

Before model sampling, require the same hash-pinned simulator, action codebook,
route-gate extractor, and fail-closed verifier as PointMaze. For all 16 split
maps, require two released-simulator route programs with distinct canonical
keys, 100 perturbations per route with zero key changes, rejection of
unsuccessful, near-miss, and environment-hash mutations, and at least two
simulator executions per second.

## Frozen development viability

- checkpoint: the already frozen
  `point_maze_interactive_warmstart_v3` 0.5B checkpoint, unchanged;
- development maps only; evaluation rows are not loaded;
- public Markov state plus velocity interface, nine fixed action letters,
  maximum 96 decisions;
- seed 75121, 64 trajectories per development map, prefix 16, temperature 1,
  top-p 1, and one masked action token per decision;
- pass only if at least two of four maps have a verified trajectory in their
  first 16 draws and at least one map has two distinct verified route keys in
  64 draws;
- require every trajectory to terminate and zero simulator, protocol, action
  mask, or identity violations.

Passing authorizes a new paired online smoke and then five seeds by two arms
for 12 passes. It must be labeled a PointMaze configuration-level replacement,
not an eighth independent semantic domain and not a ConstructiveCode result.
