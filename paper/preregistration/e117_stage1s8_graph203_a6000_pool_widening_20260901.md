# E117 Stage-1 S8 Graph-203 A6000 pool widening

Date: 2026-09-01

## Trigger and amendment

After maintenance, frozen Graph Coloring seed-203 leader job `30980490`
remained pending on node207 with a next-day priority projection while multiple
shared A6000 nodes had compatible capacity.  Change only its scheduler account
from `allcs` to `mltheory` and widen its requested node list to the available
shared A6000 class (`node205`, `node206`, `node207`, `node208`, `node300`, and
`node301`).  Preserve partition `all`, A6000 GRES, time limit, source snapshot,
model/data/seed, F arm, optimizer, evaluation, and all scientific exports.

Downstream jobs `30980491` and `30980492` retain their frozen within-block
ordering.  The amendment uses scheduler state only and does not inspect this
block's outcomes.
